"""Pocket TTS narration runner - ONE segment at a time (Task 13).

Generates per-segment WAV files for a single scene into a per-scene folder:

    output/audio/page_001_scene_001/
        segment_000.wav
        segment_001.wav
        ...
        timing.json

Checkpointing is per segment so an interrupted run resumes where it stopped
without regenerating work:

    state key:  page_001_scene_001_seg_001 with value "tts_completed"

The provider is resolved via pipeline.pocket_tts and the reference voice is
validated once up front (see pipeline.voice_reference). Synthesis is done one
segment at a time and the provider is released (model/voice state dropped, GC
forced) after each segment to stay inside the ~4 GB RAM budget.
"""
import gc
import json
import logging
from pathlib import Path

from .audio_generator import load_script, validate_page_number, validate_scene_number
from .pocket_tts import (
    create_pocket_tts_provider,
)
from .voice_reference import (
    ReferenceAudioValidationError,
    reference_audio_path,
    validate_reference,
)

LOG = logging.getLogger("mangaexplainer")

SEG_WAV = "segment_{0:03d}.wav"
TIMING_JSON = "timing.json"
CHECKPOINT_VALUE = "tts_completed"


def _scene_key(page, scene):
    return f"page_{page:03d}_scene_{scene:03d}"


def _seg_key(page, scene, seg_index):
    # Task 13 checkpoint: page_001_scene_001_seg_001 (1-based segment index).
    return f"{_scene_key(page, scene)}_seg_{seg_index:03d}"


def scene_audio_dir(cfg, page, scene):
    return Path(cfg.output.audio_dir) / _scene_key(page, scene)


def segment_wav_path(cfg, page, scene, seg_index):
    return scene_audio_dir(cfg, page, scene) / SEG_WAV.format(seg_index)


def timing_json_path(cfg, page, scene):
    return scene_audio_dir(cfg, page, scene) / TIMING_JSON


class TtsRunner:
    def __init__(self, cfg, provider=None):
        self.cfg = cfg
        self.provider = provider
        self._owned_provider = provider is None
        self._provider = None

    # -- lifecycle ---------------------------------------------------------

    def _get_provider(self):
        if self.provider is not None:
            return self.provider
        if self._provider is not None:
            return self._provider
        self._provider = create_pocket_tts_provider(self.cfg)
        return self._provider

    # -- main entry --------------------------------------------------------

    def run_scene(self, page, scene, state, force=False, segment=None):
        """Generate narration audio for one scene, one segment at a time.

        segment: optional 1-based segment index; when given, only that segment
                 is (re)generated.
        Returns a result dict for the CLI.
        """
        try:
            return self._run(page, scene, state, force, segment)
        finally:
            self._release()

    def _release(self):
        if self._owned_provider and getattr(self, "_provider", None) is not None:
            try:
                self._provider.release()
            except Exception:
                pass
        self._provider = None
        gc.collect()

    def _run(self, page, scene, state, force, segment):
        validate_page_number(page)
        validate_scene_number(scene)

        script_data, script_path = load_script(self.cfg, page, scene)
        segments = script_data.get("segments") or []
        if not segments:
            raise RuntimeError(f"no segments in {script_path}")

        # Validate the reference voice once, before doing any synthesis.
        ref_info = self._validate_reference()

        # Provider only used when there is at least one segment to generate.
        provider = self._get_provider()

        out_dir = scene_audio_dir(self.cfg, page, scene)
        out_dir.mkdir(parents=True, exist_ok=True)

        timing = {"page": page, "scene": scene, "sample_rate": provider.sample_rate,
                  "segments": []}
        generated = 0
        skipped = 0

        indices = segments_to_process(segments, segment)

        for index in indices:
            seg = segments[index]
            text = str(seg.get("text") or "").strip()
            if not text:
                raise RuntimeError(f"segment {seg.get('segment_id')!r} has empty text")
            seg_id = seg.get("segment_id") or text[:24] or f"seg{index}"
            key = _seg_key(page, scene, index + 1)
            target = float(seg.get("estimated_seconds") or 0)
            out_path = segment_wav_path(self.cfg, page, scene, index)

            if not force and state.item_done(key, CHECKPOINT_VALUE):
                duration = _existing_duration(out_path)
                if duration is not None:
                    skipped += 1
                else:
                    # checkpoint says done but WAV is missing -> regenerate
                    provider.synth(text, out_path, target_seconds=target or None,
                                   speaker=seg.get("speaker"))
                    duration = _existing_duration(out_path)
                    state.mark_item_done(key, CHECKPOINT_VALUE)
                    generated += 1
            else:
                provider.synth(text, out_path, target_seconds=target or None,
                               speaker=seg.get("speaker"))
                state.mark_item_done(key, CHECKPOINT_VALUE)
                generated += 1
                duration = _existing_duration(out_path)
            timing["segments"].append({
                "index": index,
                "segment_id": seg_id,
                "type": seg.get("type", "narration"),
                "text": seg.get("text", ""),
                "estimated_seconds": round(target, 3),
                "duration_seconds": round(duration, 3),
                "wav": SEG_WAV.format(index),
            })

            # Free memory between segments: one at a time under heavy RAM load.
            gc.collect()

        timing_path = timing_json_path(self.cfg, page, scene)
        timing_path.write_text(json.dumps(timing, indent=2, ensure_ascii=False),
                               encoding="utf-8")

        result = {
            "page": page,
            "scene": scene,
            "scene_key": _scene_key(page, scene),
            "summary": "skipped" if (generated == 0 and skipped > 0) else "generated",
            "segments_total": len(segments),
            "segments_generated": generated,
            "segments_skipped": skipped,
            "reference": ref_info,
            "timing_json": str(timing_path),
            "audio_dir": str(out_dir),
        }
        if ref_info.get("missing"):
            result["warning"] = (
                "reference voice file is missing; a default/built-in voice is "
                "used instead (copy the supplied sample to "
                "input/voice_reference.mp3 for a cloned voice)."
            )
        cond_note = getattr(provider, "_conditioning", None)
        cond_unavail = getattr(provider, "_conditioning_unavailable", None)
        result["conditioning"] = cond_note
        if cond_unavail:
            result["conditioning_unavailable"] = cond_unavail
            result["warning"] = (
                "Pocket TTS reference-audio conditioning is unavailable in "
                "this installation (cloning weights are license-gated). A "
                "built-in catalog voice was used instead; no voice was faked."
            )
        return result

    # -- helpers -----------------------------------------------------------

    def _validate_reference(self):
        path = reference_audio_path(self.cfg)
        if path is None or not path.is_file():
            return {"missing": True, "path": str(path) if path else None,
                    "note": "no reference file; using configured/fallback voice"}
        try:
            info = validate_reference(path)
            info["missing"] = False
            return info
        except ReferenceAudioValidationError as exc:
            LOG.warning("reference validation failed (%s); continuing with "
                        "fallback voice", exc)
            return {"missing": True, "path": str(path), "error": str(exc)}


def _existing_duration(path):
    from .pocket_tts import wav_duration

    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    try:
        return wav_duration(p)
    except Exception:
        return None


def segments_to_process(segments, segment):
    total = len(segments)
    if segment is None:
        return list(range(total))
    if not isinstance(segment, int):
        segment = int(segment)
    if segment < 1 or segment > total:
        raise ValueError(
            f"--segment must be between 1 and {total} (got {segment})"
        )
    return [segment - 1]
