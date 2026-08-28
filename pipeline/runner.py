"""Crash-safe pipeline runner.

Walks the ordered stage list one at a time, checkpointing after every stage.
Only completed stages are retained; anything left "running" by a crash is
re-attempted. In this skeleton every stage.run() is a stub that raises
NotImplementedError, so resume reports the point but processes nothing.
"""
import logging

from pipeline.stages import STAGES

LOG = logging.getLogger("mangaexplainer")


class PipelineRunner:
    def __init__(self, cfg, state, stages=None):
        self._cfg = cfg
        self._state = state
        self._stages = list(stages) if stages is not None else list(STAGES)

    def context(self):
        return {
            "config": self._cfg,
            "state": self._state,
            "batch_size": self._cfg.pipeline.batch_size,
        }

    def next(self):
        pending = self._state.next_pending()
        for stage in self._stages:
            if stage.name == pending:
                return stage
        return None

    def resume(self):
        stage = self.next()
        if stage is None:
            return False, "no pending stage - pipeline is complete"
        ctx = self.context()
        self._state.mark_running(stage.name)
        try:
            stage.run(ctx)
        except NotImplementedError:
            self._state.mark_pending(stage.name)
            return (
                False,
                f"stage '{stage.name}' is a stub; the pipeline engine is not "
                f"implemented yet (skeleton only). Nothing was processed.",
            )
        except Exception:
            self._state.mark_failed(stage.name)
            LOG.exception("stage '%s' failed", stage.name)
            raise
        self._state.mark_completed(stage.name)
        return True, f"stage '{stage.name}' completed"