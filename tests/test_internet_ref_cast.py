"""Tests: character cast extraction from Wikipedia "List of X characters" pages.

These validate the *pure* helpers behind ``fetch_characters`` against a small
hermetic Naruto-style fixture (cast-group L2 headings + leaf L3 persons), the
faction/TOC gate that routes One Piece-style pages to the wikilink fallback,
and the wikilink fallback itself. ``_get_json`` is stubbed so nothing touches
the network.
"""
import pytest

from pipeline import internet_ref as ir


NARUTO_FIXTURE = """\
==Main characters==
===Naruto Uzumaki===
{{nihongo|Naruto Uzumaki|うずまき ナルト}} is the main protagonist of the series.
===Sasuke Uchiha===
{{nihongo|Sasuke Uchiha}} is Naruto's rival and teammate.
===Team 7===
The members of Team 7 are Naruto, Sasuke and Sakura.
====Sai====
Sai replaces Sasuke in the reformed team Kakashi.
==Supporting characters==
===Kakashi Hatake===
{{nihongo|Kakashi Hatake}} is the leader of Team 7.
===Hokage===
The Hokage are the leaders of Konohagakure.
====Hashrama Senju====
Hashrama was the first Hokage.
==Faction page (should NOT parse structured)==
===Pirates===  # placeholder: the gate keyed on the L2 list exists above
"""

ONEPIECE_FIXTURE = """\
==Pirates==
===Straw Hats===
The protagonists of the One Piece series are all members of the [[Straw Hats]] crew.
[[Monkey D. Luffy]] captains the crew, with [[Roronoa Zoro]], [[Nami]] and [[Sanji]].
[[Monkey D. Luffy]] is the captain, [[Roronoa Zoro]] the swordsman, [[Nami]] the navigator.
[[Sanji]] cooks, [[Monkey D. Luffy]] dreams of the sea, [[Roronoa Zoro]] uses three swords.
==World Government==
===World Economic Journal===
The is a newspaper.
==References==
More text, and [[Netflix]] is a company, [[Usopp]] is voiced by [[Christopher Bevins]].
"""


def _stub(monkeypatch, wikitext: str, page_title: str = "List of Omnibus characters"):
    captures = {}

    def fake(url, timeout):
        if "prop=wikitext" in url:
            return {"parse": {"wikitext": {"*": wikitext}, "sections": []}}
        if "list=search" in url:
            return {"query": {"search": [{"title": page_title}]}}
        return None

    monkeypatch.setattr(ir, "_get_json", fake)
    return captures


# ---------------------------------------------------------------- structured


def test_split_blocks_handles_nested_heading_levels():
    blocks = ir._split_blocks("==A==\naaa\n===B===\nbbb\n")
    assert [(l, h) for l, h, _ in blocks] == [(2, "A"), (3, "B")]


def test_clean_heading_strips_italics_and_links():
    assert ir._clean_heading("''Naruto''") == "Naruto"
    assert ir._clean_heading("Team Guy (Team 9)") == "Team Guy (Team 9)"


def test_first_sentence_prunes_name_and_boilerplate():
    body = "{{nihongo|Zabuza Momochi|桃地}} is a former member of the swordsmen. He later joins."
    assert ir._first_sentence(body, "Zabuza Momochi").startswith("a former member")
    va = "voiced by [[X]]. He is the protagonist."
    # the "voiced by" sentence is skipped -> the next real sentence is returned
    assert ir._first_sentence(va, "Someone") == "He is the protagonist."


def test_cast_from_sections_structured_layout(monkeypatch):
    _stub(monkeypatch, NARUTO_FIXTURE)
    cast = ir._cast_from_sections("List of Omnibus characters", timeout=5)
    names = [c["name"] for c in cast]
    # individuals only; Team 7 & Hokage have member subsections -> excluded
    assert "Naruto Uzumaki" in names
    assert "Sasuke Uchiha" in names
    assert "Kakashi Hatake" in names
    assert "Sai" not in names        # under a grouped (non-leaf) section
    assert "Team 7" not in names     # group w/ members
    assert "Hokage" not in names     # group w/ members
    role = next(c for c in cast if c["name"] == "Naruto Uzumaki")["role"]
    assert "main protagonist" in role


def test_cast_from_sections_skips_faction_toc(monkeypatch):
    _stub(monkeypatch, ONEPIECE_FIXTURE)
    # no cast-group L2 heading -> structured pass refuses
    assert ir._cast_from_sections("One Piece", timeout=5) == []


# ------------------------------------------------------------------- fallback


def test_wikilink_cast_returns_crew_names(monkeypatch):
    _stub(monkeypatch, ONEPIECE_FIXTURE)
    names = ir._wikilink_cast("One Piece", timeout=5)
    assert "Monkey D. Luffy" in names
    assert "Roronoa Zoro" in names
    assert "Straw Hats" not in names
    # tail "+ voice actor credits" dropped
    assert "Netflix" not in names
    assert "Christopher Bevins" not in names


def test_fetch_characters_structured_first_then_fallback(monkeypatch):
    _stub(monkeypatch, NARUTO_FIXTURE)
    cast = ir.fetch_characters("Omnibus", timeout=5)
    assert len(cast) >= 2
    # structured path yields roles
    assert any(c["role"] for c in cast)


def test_fetch_characters_network_failure_empty():
    # blank title -> []
    assert ir.fetch_characters("") == []


def test_fetch_characters_with_portraits_returns_verified_image(monkeypatch):
    w = """\
==Main characters==
===Naruto Uzumaki===
[[File:Naruto portrait.jpg|thumb|right]]
{{nihongo|Naruto Uzumaki}} is the main protagonist.
===Sasuke Uchiha===
{{nihongo|Sasuke Uchiha}} is Naruto's rival.
==Supporting characters==
===Kakashi Hatake===
{{nihongo|Kakashi Hatake}} is the leader of Team 7.
"""
    _stub(monkeypatch, w, page_title="List of Naruto characters")
    cast, portraits = ir.fetch_characters_with_portraits("Naruto", timeout=5)
    assert any(c["name"] == "Naruto Uzumaki" for c in cast)
    assert portraits.get("Naruto Uzumaki") == "Naruto portrait.jpg"


def test_fetch_characters_with_portraits_skips_shared_image(monkeypatch):
    w = """\
==Main characters==
===Naruto Uzumaki===
[[File:Berserk cast roster.png|thumb|right]]
{{nihongo|Naruto Uzumaki}} is the protagonist.
"""
    _stub(monkeypatch, w, page_title="List of Naruto characters")
    cast, portraits = ir.fetch_characters_with_portraits("Naruto", timeout=5)
    # "cast roster" is site furniture, never a character portrait
    assert "Naruto Uzumaki" not in portraits


def test_fetch_characters_with_portraits_no_image_casts_still_returned(monkeypatch):
    w = """\
==Main characters==
===Guts===
{{nihongo|Guts}} is the protagonist.
===Griffith===
{{nihongo|Griffith}} leads the Band of the Hawk.
"""
    _stub(monkeypatch, w, page_title="List of Berserk characters")
    cast, portraits = ir.fetch_characters_with_portraits("Berserk", timeout=5)
    assert {c["name"] for c in cast} == {"Guts", "Griffith"}
    assert portraits == {}


def test_fetch_characters_with_portraits_blank_title():
    assert ir.fetch_characters_with_portraits("") == ([], {})


# ------------------------------------------------------------ volume scoping


def _cast(n):
    return [{"name": "C%d" % i, "role": "r%d" % i} for i in range(n)]


def test_scope_cast_vol1_keeps_early_block_cuts_late_arc():
    cast = _cast(40)
    scoped = ir.scope_cast_to_volume(cast, 1)
    # founding block (9) only; late-arc characters cut
    assert len(scoped) == 9
    assert [c["name"] for c in scoped] == ["C%d" % i for i in range(9)]


def test_scope_cast_widens_with_volume():
    cast = _cast(40)
    assert len(ir.scope_cast_to_volume(cast, 1)) == 9
    assert len(ir.scope_cast_to_volume(cast, 2)) == 13
    assert len(ir.scope_cast_to_volume(cast, 3)) == 17
    # big-enough volume keeps everything
    assert len(ir.scope_cast_to_volume(cast, 20)) == 40


def test_scope_cast_noop_without_volume():
    cast = _cast(5)
    assert ir.scope_cast_to_volume(cast, None) == cast
    assert ir.scope_cast_to_volume(cast, 0) == cast


def test_scope_cast_empty():
    assert ir.scope_cast_to_volume([], 1) == []