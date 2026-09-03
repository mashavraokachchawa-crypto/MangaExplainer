"""Low-RAM housekeeping for the MangaExplainer pipeline.

The pipeline is already structured for a small RAM budget:
  * one process, one item at a time (pipeline.batch_size == 1, sequential -
    no workers are spawned anywhere)
  * work is processed in small chunks (page -> panel -> scene) so only a small
    context is ever in RAM
  * every stage result is written to SSD (pages/, panels/, ocr/, analysis/,
    scenes/, script/, audio/, shots/, crops/, output/) and read back from disk
    instead of being held in memory - "the SSD is the memory"
  * heavy models are loaded lazily and released after each item

This module adds the two runtime pieces that keep RAM and disk in budget while
a run is active:
  * MemoryGuard.start(): a daemon thread that every memory.sweep_seconds
    (default 60) runs gc.collect(), checks RSS against memory.guard_mb and
    clears the safe transient cache dirs. The TTS voice embedding under
    state/cache/voice is a "rest on SSD as memory" cache, so it is protected.
  * MemoryGuard.tick(): called by the orchestrator between stages and between
    page chunks to re-check RSS and reclaim stale files in the configured
    sweep dirs (default [state/cache, output/tmp]). output/tmp is only swept
    here, between stages - never while the renderer is writing it.

Only files OLDER than the safety window (about two sweep cycles) are ever
deleted, so a stage that is actively writing a file is never disturbed.
"""
import gc
import logging
import threading
import time
from pathlib import Path

LOG = logging.getLogger("mangaexplainer.lowram")

# Reusable embeddings that are exactly "SSD as memory": expensive to rebuild,
# so they are never removed by a periodic sweep.
PROTECTED_SUBDIRS = ("voice",)

# Longest time (seconds) we ever leave a file alone; protects files that sit
# idle briefly between chunks of a long-running stage.
_STALE_MIN_AGE = 120


def rss_mb():
    """Resident set size in MB (best-effort, Linux /proc/self/status)."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class MemoryGuard:
    """Periodic cache sweep + RSS guard for one pipeline run.

    Call tick() at stage boundaries and on page chunks; start()/close()
    bracket the run and drive the periodic loop during long stages.
    """

    def __init__(self, cfg, root, log=None):
        self.log = log or LOG
        memory = getattr(cfg, "memory", None) or {}
        self.guard_mb = max(0, int(getattr(memory, "guard_mb", 3072) or 0))
        self.sweep_seconds = max(0, int(getattr(memory, "sweep_seconds", 60) or 0))
        self.root = Path(root)
        render = getattr(cfg, "render", None) or {}
        self._temp_dir = self.root / str(render.get("temp_dir", "output/tmp"))
        configured = getattr(memory, "sweep_dirs", None) or [
            "state/cache", self._temp_dir,
        ]
        self.sweep_dirs = [self._resolve(str(d)) for d in configured]
        self._stop = threading.Event()
        self._thread = None
        self._last_sweep = None

    def _resolve(self, raw):
        path = Path(raw)
        return path.resolve() if path.is_absolute() else (self.root / raw).resolve()

    def _safe_timer_dirs(self):
        # The timer loop can run while the renderer is writing its temp dir, so
        # the timer only ever touches state/cache (never output/tmp).
        temp = self._temp_dir.resolve()
        return [d for d in self.sweep_dirs if d != temp]

    # ------------------------------------------------------------------ run
    def start(self):
        if self.sweep_seconds > 0 and self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="lowram-guard", daemon=True
            )
            self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # --------------------------------------------------------------- public
    def tick(self, sweep_dirs=None):
        """Called between stages / small chunks: gc, check RSS, sweep caches.

        sweep_dirs=None -> configured dirs. Passing an empty tuple restricts
        the tick to gc + RSS (used while a stage is actively writing temp).
        """
        rss = rss_mb()
        gc.collect()
        if rss is not None and self.guard_mb and rss > self.guard_mb:
            self.log.warning(
                "memory guard: RSS %s MB above guard %s MB (gc ran); "
                "reduce batch work or raise memory.guard_mb if it persists",
                rss, self.guard_mb,
            )
        dirs = self.sweep_dirs if sweep_dirs is None else list(sweep_dirs or ())
        self._maybe_sweep(dirs)

    def sweep(self, sweep_dirs, force=False):
        """Delete stale files older than the safety window; prune empty dirs."""
        if not force:
            return
        now = time.time()
        older_than = now - max(self.sweep_seconds * 2, _STALE_MIN_AGE)
        removed = 0
        freed = 0
        prune = []
        for base in sweep_dirs:
            if not base.is_dir():
                continue
            try:
                found = list(base.rglob("*"))
            except OSError:
                continue
            for path in found:
                if self._is_protected(path):
                    continue
                try:
                    if path.is_symlink():
                        if path.lstat().st_mtime < older_than:
                            path.unlink(missing_ok=True)
                            removed += 1
                    elif path.is_file():
                        st = path.lstat()
                        if st.st_mtime < older_than:
                            size = st.st_size
                            path.unlink(missing_ok=True)
                            removed += 1
                            freed += size
                    elif path.is_dir():
                        prune.append(path)
                except OSError:
                    continue
        # deepest first; rmdir only succeeds on empty dirs
        for path in sorted(set(prune), key=lambda p: len(p.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        if removed:
            self.log.info(
                "cache sweep: removed %d stale file(s), %.1f MB freed",
                removed, freed / 1e6,
            )

    # ------------------------------------------------------------- internal
    def _is_protected(self, path):
        return any(part in PROTECTED_SUBDIRS for part in path.parts)

    def _maybe_sweep(self, dirs):
        if self.sweep_seconds <= 0 or not dirs:
            return
        now = time.time()
        if self._last_sweep is not None and now - self._last_sweep < self.sweep_seconds:
            return
        self._last_sweep = now
        self.sweep(dirs, force=True)

    def _run(self):
        while not self._stop.wait(self.sweep_seconds):
            gc.collect()
            rss = rss_mb()
            if rss is not None and self.guard_mb and rss > self.guard_mb:
                self.log.warning(
                    "memory guard: RSS %s MB above %s MB while running",
                    rss, self.guard_mb,
                )
            self._maybe_sweep(self._safe_timer_dirs())