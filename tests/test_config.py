"""Tests: configuration loads with low-RAM defaults."""

from pathlib import Path

from config.loader import DEFAULT_CONFIG, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_configuration_loads():
    cfg = load_config(str(ROOT))
    assert cfg.input.pdf == (ROOT / "input" / "manga.pdf")
    assert cfg.output.dir == (ROOT / "output")
    assert str(cfg.images.resolution) == "1200x1800"
    assert str(cfg.video.resolution) == "1920x1080"
    assert cfg.video.fps == 30


def test_batch_size_is_one():
    cfg = load_config(str(ROOT))
    assert cfg.pipeline.batch_size == 1
    assert isinstance(cfg.pipeline.batch_size, int)


def test_yaml_requests_batch_size_one():
    text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    assert "batch_size: 1" in text


def test_missing_yaml_falls_back_to_defaults(tmp_path):
    cfg = load_config(root_dir=str(tmp_path))
    assert cfg.pipeline.batch_size == 1
    assert cfg.video.fps == 30
    assert cfg.input.pdf == (tmp_path / "input" / "manga.pdf")


def test_panels_defaults_available():
    cfg = load_config(str(ROOT))
    assert cfg.panels.min_area == 3000
    assert cfg.panels.strategy == "gutter_flood"
    assert cfg.output.panels_dir == (ROOT / "panels")


def test_reading_defaults_available():
    cfg = load_config(str(ROOT))
    assert cfg.reading.direction == "rtl"
    assert cfg.reading.row_overlap_ratio == 0.5
    assert cfg.reading.weights.row_overlap == 2.0


def test_ocr_and_preprocess_defaults_available():
    cfg = load_config(str(ROOT))
    assert cfg.ocr.engine == "auto"
    assert cfg.ocr.psm == 11
    assert cfg.preprocess.grayscale is True
    assert cfg.preprocess.resize_scale == 1.5
    assert cfg.output.ocr_dir == (ROOT / "ocr")


def test_vlm_defaults_available():
    cfg = load_config(str(ROOT))
    assert cfg.output.analysis_dir == (ROOT / "analysis")
    assert cfg.vlm.enabled is True
    assert cfg.vlm.provider == "local"
    assert cfg.vlm.model == ""
    assert cfg.vlm.device == "cpu"
    assert cfg.vlm.max_image_size == 768
    assert cfg.vlm.max_new_tokens == 256
    assert cfg.vlm.timeout_seconds == 120


def test_llm_defaults_available():
    cfg = load_config(str(ROOT))
    assert cfg.output.script_dir == (ROOT / "script")
    assert cfg.output.audio_dir == (ROOT / "audio")
    assert cfg.llm.enabled is True
    assert cfg.llm.provider == "local"
    assert cfg.llm.model == ""
    assert cfg.llm.device == "cpu"
    assert cfg.llm.max_context == 4096
    assert cfg.llm.max_new_tokens == 512
    assert cfg.llm.temperature == 0.7
    assert cfg.llm.timeout_seconds == 120


def test_tts_defaults_available():
    cfg = load_config(str(ROOT))
    assert cfg.tts.enabled is True
    assert cfg.tts.engine == "auto"
    assert cfg.tts.voice == "en"
    assert cfg.tts.sample_rate == 22050
    assert cfg.tts.rate_wpm == 150
    assert cfg.tts.pitch_base == 50
    assert cfg.tts.timeout_seconds == 60


def test_shots_defaults_available():
    cfg = load_config(str(ROOT))
    assert cfg.output.shots_dir == (ROOT / "shots")
    assert cfg.shots.match_weights.character == 0.25
    assert cfg.shots.match_weights.story_relevance == 0.15
    assert cfg.shots.review_threshold == 0.55
    assert cfg.shots.tie_epsilon == 0.02
    assert cfg.shots.direct_match_floor == 0.90
    assert cfg.shots.secondary_panel_epsilon == 0.15
    assert cfg.shots.long_segment_threshold == 9.0
    assert cfg.shots.max_shots_per_segment == 3
    assert cfg.shots.zoom_in_end == 1.12
    assert cfg.shots.zoom_out_end == 0.92


def test_crops_defaults_available():
    cfg = load_config(str(ROOT))
    assert cfg.output.crops_dir == (ROOT / "crops")
    assert cfg.crops.resolution == "1280x720"
    assert cfg.crops.format == "jpg"
    assert cfg.crops.jpeg_quality == 90
    assert cfg.crops.safe_padding == 0.06
    assert cfg.crops.critical_weight == 0.6
    assert cfg.crops.min_blob_area == 80
    assert cfg.crops.max_regions == 24
    assert cfg.crops.debug is True


def test_defaults_are_not_mutated():
    load_config(str(ROOT))
    assert DEFAULT_CONFIG["pipeline"]["batch_size"] == 1