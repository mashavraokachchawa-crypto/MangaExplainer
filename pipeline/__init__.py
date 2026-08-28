"""Pipeline package: ordered stage stubs plus the crash-safe runner."""

from pipeline.base import Stage
from pipeline.runner import PipelineRunner
from pipeline.stages import STAGES, STAGE_NAMES

__all__ = ["Stage", "PipelineRunner", "STAGES", "STAGE_NAMES"]