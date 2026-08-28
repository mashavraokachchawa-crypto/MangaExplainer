"""Declarative list of pipeline stages (stubs in the skeleton).

Each stage is a name plus its input/output workspace directory. Implementations
arrive later (PDF extraction, OCR, VLM, TTS, rendering); nothing here loads
images, PDFs or AI models.
"""
from pipeline.base import Stage

STAGE_SPECS = (
    ("extract_pages", "input", "pages"),
    ("segment_panels", "pages", "panels"),
    ("crop_panels", "panels", "crops"),
    ("run_ocr", "crops", "ocr"),
    ("analyze_panels", "ocr", "analysis"),
    ("build_scenes", "analysis", "scenes"),
    ("write_script", "scenes", "script"),
    ("generate_audio", "script", "audio"),
    ("build_subtitles", "audio", "subtitles"),
    ("plan_shots", "scenes", "shots"),
    ("pick_music", "shots", "music"),
    ("render_video", "shots", "output"),
)

STAGES = [Stage(*spec) for spec in STAGE_SPECS]
STAGE_NAMES = [stage.name for stage in STAGES]