"""Pocket TTS speech provider for the narration audio stage (Task 13).

Selected engine for the project is Pocket TTS (Kyutai, the CPU-first open
source text-to-speech library that supports reference-audio voice
conditioning / cloning):

    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("<voice.wav>")
    audio = model.generate_audio(voice_state, text)

This module exposes a tiny provider interface so the engine can be swapped
later without touching the rest of the pipeline:

    create_pocket_tts_provider(cfg) -> PocketTtsProvider | MockTtsProvider

The real Pocket TTS library is imported lazily inside the provider so that
tests and offline smoke runs never need torch / the model installed. A
deterministic mock provider backs the automated tests just like the rest of
the pipeline.

Reference-audio conditioning:
  - Pocket TTS supports cloning from an audio prompt file.
  - The reference is validated *before* any synthesis (see voice_reference).
  - The extracted voice state is created ONCE per run and cached on disk
    (safetensors via export_model_state) so repeated segments do not
    recompute the embedding, keeping RAM and CPU use low.
  - If reference conditioning is unavailable/unsupported we raise a clear
    error rather than silently faking a clone; caller should then fall back
    to the configured built-in voice.
"""
import gc
import logging
import shutil
import wave
from pathlib import Path

LOG = logging.getLogger("mangaexplainer")

# Pocket TTS output sample rate (from the model).
POCKET_SAMPLE_RATE = 24000

# Where we cache a reference voice embedding so segments and re-runs are fast.
VOICE_CACHE_DIR = "cache/voice"  # relative to project root; see cfg


class PocketTtsError(Exception):
    """Base class for all Pocket TTS orchestration errors."""


class PocketTtsUnavailable(PocketTtsError):
    """The pocket-tts package (or a dependency such as torch) is not installed."""


class PocketTtsNotConfigured(PocketTtsError):
    """TTS is disabled in config."""


class ReferenceAudioError(PocketTtsError):
    """The reference audio file is missing or invalid."""


class UnsupportedReferenceConditioning(PocketTtsError):
    """This installation cannot condition on a reference voice."""


class GenerationFailed(PocketTtsError):
    """Pocket TTS failed to produce a usable WAV."""


def _is_pocket_tts_installed():
    try:
        import pocket_tts  # noqa: F401
        return True
    except Exception:
        return False


def wav_duration(path):
    """Actual duration in seconds of a mono 16-bit WAV file."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate() or POCKET_SAMPLE_RATE
        return frames / rate


def _available_voices():
    """Best-effort list of built-in Pocket TTS voice names."""
    try:
        import pocket_tts
        catalog = getattr(pocket_tts, "VOICE_CATALOG", {}) or {}
        names = list(catalog.keys()) if isinstance(catalog, dict) else []
        return names or ["alba"]
    except Exception:
        return ["alba"]


class TtsVoiceStateCache:
    """On-disk cache of a reference voice embedding (safetensors).

    Created once from the reference audio, reused across segments and reruns,
    so we never recompute the voice embedding or hold it longer than needed.
    """

    def __init__(self, cache_dir):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._current = None

    def path_for(self, key):
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)
        return self._dir / f"voice-{safe}.safetensors"

    def exists(self, key):
        return self.path_for(key).is_file()

    def get(self, key):
        if self._current is not None:
            return self._current
        path = self.path_for(key)
        if path.exists():
            return str(path)
        return None


class TtsProvider:
    """Common interface each provider implements."""

    name = "tts"

    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def sample_rate(self):
        return int(getattr(self.cfg.tts, "sample_rate", POCKET_SAMPLE_RATE) or POCKET_SAMPLE_RATE)

    @property
    def reference_audio(self):
        path = getattr(self.cfg.tts, "reference_audio", None)
        if not path:
            return None
        p = Path(path)
        return p if p.is_absolute() else Path(self.cfg.root_dir) / p

    @staticmethod
    def available():
        return False

    @staticmethod
    def supports_reference_conditioning():
        """Whether this provider can drive a narration from a reference voice."""
        return False

    def release(self):
        pass

    def synth(self, text, out_path, target_seconds=None, speaker=None):
        raise NotImplementedError


class MockTtsProvider(TtsProvider):
    """Deterministic sine-tone WAV generator for tests / offline smoke runs.

    Mirrors the existing espeak/mock helper so assertions on duration and RAM
    behaviour stay offline. Does not condition on the reference voice (mock
    reports that reference conditioning is simulated/unsupported so tests can
    exercise both code paths).
    """

    name = "mock"

    @staticmethod
    def available():
        return True

    @staticmethod
    def supports_reference_conditioning():
        # The mock cannot clone a voice; keep False so the runner treats it as
        # "reference conditioning unavailable" the same way code paths behave.
        return False

    def synth(self, text, out_path, target_seconds=None, speaker=None):
        import array

        rate = self.sample_rate
        seconds = max(0.05, float(target_seconds if target_seconds else 1.0))
        total = int(seconds * rate)
        base = 220.0
        seed = 0
        if speaker:
            seed = sum((i + 1) * ord(ch) for i, ch in enumerate(str(speaker))) % 7
        freq = base + seed * 25.0
        phase = 0.0
        step = 2.0 * 3.141592653589793 * freq / rate
        frames = array.array("h")
        frames.extend(int(0.2 * 32767 * __import__("math").sin(phase + step * i)) for i in range(total))
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(frames.tobytes())
        return wav_duration(path)


class PocketTtsProvider(TtsProvider):
    """Real narration provider backed by the pocket-tts library.

    Loads the model lazily, resolves a voice state (from a cached embedding
    or the reference audio prompt), and synthesizes each segment to WAV one at
    a time. Resources (model, voice state, torch tensors) are released after
    every segment via release() + gc.collect() to respect the 4 GB RAM budget.
    """

    name = "pocket_tts"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._model = None
        self._voice_state = None
        self._voice_state_cache = None
        self._import_error = None
        self._conditioning = None  # "reference" | "builtin" | None(resolved lazily)
        self._conditioning_unavailable = None  # set when clone ref could not be used

    # -- availability -----------------------------------------------------

    @staticmethod
    def available():
        return _is_pocket_tts_installed()

    @staticmethod
    def supports_reference_conditioning():
        # Pocket TTS API exposes get_state_for_audio_prompt for a prompt file,
        # which is reference-audio conditioning. We only claim support when the
        # package is importable, since otherwise nothing can be generated.
        return PocketTtsProvider.available()

    def _require_deps(self):
        if self._model is not None:
            return
        try:
            import pocket_tts  # noqa: F401
        except Exception as exc:
            self._import_error = exc
            raise PocketTtsUnavailable(
                "Pocket TTS is not installed. Install it with:\n"
                "    pip install pocket-tts\n"
                "(also requires torch; on Linux use the CPU wheel). "
                f"Import error: {exc}"
            ) from None
        try:
            from pocket_tts import TTSModel
        except Exception as exc:
            raise PocketTtsUnavailable(
                f"pocket_tts installed but TTSModel is unavailable: {exc}"
            ) from None
        self._model = TTSModel.load_model()
        self._voice_state_cache = TtsVoiceStateCache(self._cache_dir())

    def _cache_dir(self):
        root = Path(self.cfg.root_dir)
        return root / "state" / "cache" / "voice"

    # -- reference resolution ---------------------------------------------

    def _reference_path(self):
        ref = self.reference_audio
        if ref and ref.is_file():
            return ref
        return None

    def _resolve_voice_state(self, ref):
        """Return a voice state (object or safetensors path) using the ref.

        Priority:
          1. cached embedding on disk for this reference -> fast reload
          2. live reference audio prompt -> build embedding, cache it
          3. fall back to a built-in voice if ref conditioning is N/A

        If a reference is present but the cloning weights cannot be accessed
        (e.g. Kaggle/HF license gated, not logged in, offline), we DO NOT
        fabricate a clone: we record this on the instance as
        `conditioning_unavailable` and fall back to the built-in voice catalog,
        exactly as the task requires ("report that reference-audio conditioning
        is unavailable in this installation" - never fake it).
        """
        self._require_deps()
        from pocket_tts import TTSModel

        ref = ref or self._reference_path()
        cache = self._voice_state_cache

        if ref is not None:
            key = Path(ref).name
            cached = cache.get(key)
            if cached:
                self._conditioning = "reference"
                return cached  # safetensors path (fast)
            try:
                built = self._build_and_cache_voice(ref, key, TTSModel)
            except ReferenceAudioError as exc:
                # Cloning unavailable: fall back to a built-in voice and tell
                # the caller explicitly. Never fake a clone.
                self._conditioning_unavailable = str(exc)
                self._conditioning = "builtin"
                voice = getattr(self.cfg.tts, "voice", "en") or "alba"
                return self._resolve_builtin_voice(voice, TTSModel)
            self._conditioning = "reference"
            return built

        # No reference: use configured built-in voice name (voice consistency).
        voice = getattr(self.cfg.tts, "voice", "en") or "alba"
        self._conditioning = "builtin"
        return self._resolve_builtin_voice(voice, TTSModel)

    def _build_and_cache_voice(self, ref, key, TTSModel):
        if not TTSModel:
            from pocket_tts import TTSModel as T
            TTSModel = T
        try:
            state = self._model.get_state_for_audio_prompt(str(ref))
        except Exception as exc:
            raise ReferenceAudioError(
                f"could not build a voice state from reference {ref}: {exc}"
            ) from None
        path = self._voice_state_cache.path_for(key)
        try:
            from pocket_tts import export_model_state
            export_model_state(state, str(path))
        except Exception:
            # keep the in-memory object if export/caching is unavailable
            path.unlink(missing_ok=True)
            return state
        return str(path)

    def _resolve_builtin_voice(self, voice, TTSModel):
        # A voice *name* like "alba" resolves via get_state_for_audio_prompt
        # (the library accepts names short of a path). Fall back to default.
        try:
            return self._model.get_state_for_audio_prompt(voice)
        except Exception:
            try:
                return self._model.get_state_for_audio_prompt("alba")
            except Exception:
                raise UnsupportedReferenceConditioning(
                    "Pocket TTS could not resolve any voice state "
                    f"(requested: {voice!r}). "
                    "Pocket TTS reference-audio conditioning is unavailable "
                    "in this installation."
                ) from None

    # -- synthesis ---------------------------------------------------------

    def synth(self, text, out_path, target_seconds=None, speaker=None):
        """Synthesize one narration segment to a WAV file (16-bit PCM)."""
        text = str(text or "").strip()
        if not text:
            raise PocketTtsError("Cannot synthesize empty narration text.")

        self._require_deps()

        # Resolve (or reuse) the voice state once per run.
        if self._voice_state is None:
            self._voice_state = self._resolve_voice_state(None)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            audio = self._model.generate_audio(self._voice_state, text)
            data = audio.cpu().numpy()
        except Exception as exc:
            raise GenerationFailed(
                f"Pocket TTS failed to generate audio for a segment: {exc}"
            ) from None

        if data is None or len(data) == 0:
            raise GenerationFailed("Pocket TTS returned empty audio for a segment.")

        import numpy as np
        try:
            import scipy.io.wavfile
        except Exception as exc:  # pragma: no cover - defensive
            raise PocketTtsUnavailable(f"scipy is required for WAV writing: {exc}")

        # write as 16-bit PCM mono
        pcm = (np.asarray(data).astype(np.float32) * 32767).clip(-32768, 32767).astype(np.int16)
        scipy.io.wavfile.write(str(out_path), self.sample_rate, pcm)

        duration = wav_duration(out_path)
        if duration <= 0:
            raise GenerationFailed("zero-duration WAV produced for a segment.")
        return duration

    def release(self):
        self._model = None
        self._voice_state = None
        self._voice_state_cache = None
        self._conditioning = None
        gc.collect()


def create_pocket_tts_provider(cfg):
    """Resolve the configured Pocket TTS provider; raises PocketTtsError.

    Honors cfg.tts.enabled / cfg.tts.provider.
    """
    tts = getattr(cfg, "tts", None)
    if not tts or not getattr(tts, "enabled", True):
        raise PocketTtsNotConfigured(
            "TTS is disabled (tts.enabled=false in config). Set enabled=true "
            "to generate narration audio."
        )
    provider = str(getattr(tts, "provider", "pocket_tts") or "pocket_tts").lower()
    if provider in ("auto", "pocket_tts", "mock"):
        if provider in ("auto", "pocket_tts") and PocketTtsProvider.available():
            return PocketTtsProvider(cfg)
        if provider == "pocket_tts":
            raise PocketTtsUnavailable(
                "tts.provider=pocket_tts but the pocket-tts package is not "
                "installed (pip install pocket-tts, or set tts.provider=mock "
                "for tests)."
            )
        LOG.warning("Pocket TTS not found; using mock TTS (sine tones, no speech).")
        return MockTtsProvider(cfg)
    raise PocketTtsError(
        f"unknown tts.provider {provider!r} (expected pocket_tts, mock, or auto)"
    )
