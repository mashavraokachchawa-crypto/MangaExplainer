"""Tests for chapter boundary detection (chapter_detect)."""
import tempfile
from pathlib import Path

import pytest

from pipeline.knowledge_db import open_knowledge_db
from pipeline.chapter_detect import (
    merge_chapter_detections,
    detect_chapters_from_internet,
    apply_chapter_detections,
    _text_hash,
)


def _det(page, conf, source, num=None, title=""):
    return {
        "chapter_number": num,
        "title": title,
        "pdf_page_start": page,
        "confidence": conf,
        "source": source,
        "extra": {},
    }


def test_text_hash_is_stable_and_case_sensitive():
    assert _text_hash("Chapter 1") == _text_hash("Chapter 1")
    assert _text_hash("chapter 1") != _text_hash("Chapter 1")  # case-sensitive


def test_merge_empty_returns_empty():
    assert merge_chapter_detections([[], []], 100) == []


def test_merge_exact_same_page_keeps_higher_confidence():
    merged = merge_chapter_detections([
        [_det(5, 0.9, "ocr", num=1, title="Ch1")],
        [_det(5, 0.7, "internet", num=1)],
    ], 50)
    assert len(merged) == 1
    assert merged[0]["pdf_page_start"] == 5
    assert merged[0]["confidence"] == 0.9
    assert merged[0]["title"] == "Ch1"


def test_merge_adjacent_pages_groups():
    merged = merge_chapter_detections([
        [_det(5, 0.8, "ocr", num=2)],
        [_det(7, 0.9, "internet", num=2)],
    ], 50)
    assert len(merged) == 1
    assert merged[0]["pdf_page_start"] == 7  # higher-conf page wins


def test_merge_separates_distinct_boundaries():
    merged = merge_chapter_detections([
        [_det(1, 0.9, "ocr", num=1)],
        [_det(30, 0.9, "ocr", num=2)],
        [_det(60, 0.9, "ocr", num=3)],
    ], 90)
    assert len(merged) == 3
    # end pages assigned
    assert merged[0]["pdf_page_end"] == 29
    assert merged[1]["pdf_page_end"] == 59
    assert merged[2]["pdf_page_end"] == 90


def test_merge_assigns_sequential_unnumbered():
    merged = merge_chapter_detections([
        [_det(1, 0.8, "transition")],
        [_det(40, 0.8, "transition")],
    ], 80)
    assert merged[0]["chapter_number"] == 1
    assert merged[1]["chapter_number"] == 2


def test_internet_even_distribution():
    dets = detect_chapters_from_internet("m1", {"total_chapters": 3}, 60)
    assert len(dets) == 3
    assert dets[0]["pdf_page_start"] == 1
    assert dets[2]["pdf_page_end"] == 60


def test_internet_no_total_returns_empty():
    dets = detect_chapters_from_internet("m1", {}, 60)
    assert dets == []


def test_apply_chapter_detections_preserves_existing(db_fixture):
    db = db_fixture
    mid = db.upsert_manga({"title": "Berserk"})
    db.add_chapter(mid, {
        "chapter_number": 1, "title": "Existing",
        "pdf_page_start": 5, "pdf_page_end": 40, "confidence": 1.0,
    })
    # New detection at page 5 is the same start -> skipped (existing kept)
    merged = merge_chapter_detections([[_det(5, 0.9, "ocr", num=1, title="New")]], 60)
    apply_chapter_detections(db, mid, merged)
    chapters = db.get_chapters(mid)
    assert chapters[0]["title"] == "Existing"
    assert chapters[0]["confidence"] == 1.0


def test_apply_chapter_detections_new_added(db_fixture):
    db = db_fixture
    mid = db.upsert_manga({"title": "Berserk"})
    merged = merge_chapter_detections([[_det(1, 0.9, "ocr", num=1)]], 40)
    apply_chapter_detections(db, mid, merged)
    chapters = db.get_chapters(mid)
    assert len(chapters) == 1
    assert chapters[0]["pdf_page_start"] == 1


@pytest.fixture()
def db_fixture():
    tmp = Path(tempfile.mkdtemp())
    database = open_knowledge_db(tmp)
    yield database
    database.close()
