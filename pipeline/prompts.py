"""Prompt template for one-panel visual analysis.

The template is deliberately narrow: describe ONLY what is visible in the
panel or strongly supported by it. Never write YouTube-style narration,
never invent off-panel events, and use "unknown" for anything uncertain.
"""

ANALYSIS_TEMPLATE = """Analyze the single manga panel in the image.

Rules:
- Describe ONLY what is visible in the image or strongly supported by it.
- Do not invent events, characters, or dialogue that are not visible.
- If something cannot be determined, return the value "unknown".
- Do not write a narration, script, or YouTube description.
- Do not reference the previous/next panel unless it was provided to you.

Return ONLY a JSON object (no extra text, no markdown) shaped exactly like:

{
  "characters": [
    {"name": "unknown", "description": "", "action": "", "emotion": ""}
  ],
  "environment": "",
  "actions": [],
  "objects": [],
  "visual_effects": [],
  "important_event": "",
  "composition": "",
  "story_relevance": "",
  "confidence": 0.0
}

Field guidance:
- characters: only for characters clearly present. name only if recognizable
  from visible traits (otherwise "unknown"). description = physical
  appearance/attire. action = what the character is doing. emotion = facial
  expression / emotional state if readable, else "unknown".
- environment: location/environment shown in the panel.
- actions: list of visible action verbs / events.
- objects: important objects, incl. weapons, shown in the panel.
- visual_effects: speed lines, impact stars, aura, explosion, etc.
- important_event: the single most important visible event.
- composition: shot framing, panel focus, point of view.
- story_relevance: only if a previous/next panel summary was given; otherwise
  base it strictly on what is visible or return "unknown".
- confidence: 0.0 .. 1.0 reflecting how confident you are in the analysis of
  this panel.
"""

OCR_GUIDANCE = """OCR context
------------
The OCR text below was extracted from THIS panel automatically and may be
incorrect. Treat it as potentially imperfect context ONLY: do not assume
the OCR is correct if it conflicts with the image, and do not fabricate
dialogue that is not visible in the artwork."""

OCR_BLOCK = """<ocr_text>
{ocr}
</ocr_text>"""

PREV_BLOCK = """Previous panel summary:
{text}"""

NEXT_BLOCK = """Next panel summary:
{text}"""


def build_analysis_prompt(ocr_context=None, previous_panel=None, next_panel=None):
    parts = [ANALYSIS_TEMPLATE]
    if ocr_context:
        parts.append(OCR_GUIDANCE)
        parts.append(OCR_BLOCK.format(ocr=ocr_context))
    if previous_panel:
        parts.append(PREV_BLOCK.format(text=previous_panel))
    if next_panel:
        parts.append(NEXT_BLOCK.format(text=next_panel))
    return "\n\n".join(parts)


NARRATION_RULES = """You are the narrator of a manga story. Write the narration for the scene described below.

Rules:
- Base the narration ONLY on the facts provided. Never invent characters,
  events, locations, or dialogue that are not listed.
- The dialogue lines are automatic OCR transcriptions and may be imperfect -
  treat them as potentially imperfect context ONLY and never reproduce or
  invent dialogue you are unsure of.
- Write 1-4 short sentences, natural to speak aloud, without stage directions.
- Return ONLY the narration text: no "narration:" prefix, no surrounding
  quotes, no markdown, no JSON, no scene metadata."""

SCENE_HEADER = """Scene: {scene_id}
Characters present: {characters}
Location(s): {locations}
Event(s): {events}
Scene summary: {summary}"""

DIALOGUE_BLOCK = """Dialogue in this scene (OCR, may be imperfect):
{lines}"""


def _present_values(values, fallback="none given"):
    kept = [str(value).strip() for value in (values or []) if value is not None]
    return ", ".join(kept) if kept and any(kept) else fallback


def build_narration_prompt(scene, dialogue_by_panel=None):
    """Prompt for one scene's narration.

    scene: a scene dict from scenes/<page>_scenes.json.
    dialogue_by_panel: optional {panel_id: ocr text} map (potentially imperfect).
    """
    dialogue_by_panel = dialogue_by_panel or {}
    lines = []
    for pid in scene.get("panel_ids") or []:
        text = (dialogue_by_panel.get(pid) or "").strip()
        if text:
            lines.append(f"- [{pid}] {text}")
    parts = [NARRATION_RULES]
    parts.append(
        SCENE_HEADER.format(
            scene_id=scene.get("scene_id", "scene_001"),
            characters=_present_values(scene.get("characters")),
            locations=_present_values(scene.get("locations")),
            events=_present_values(scene.get("events")),
            summary=_present_values([scene.get("summary")]),
        )
    )
    if lines:
        parts.append(DIALOGUE_BLOCK.format(lines="\n".join(lines)))
    return "\n\n".join(parts)


NARRATION_SEGMENT_RULES = """You are writing the narration for ONE scene of a manga explanation video.

Write in plain, natural, engaging spoken English suitable for a YouTube-style
explanation to someone who has never read the manga. Describe what is
happening in the scene, chronologically, following the panel reading order.

Rules:
- Base the narration ONLY on the facts supplied below. NEVER invent events,
  character motivations, locations, dialogue, abilities, relationships, or
  background information.
- If something is uncertain (not clearly supported), mark it with "(uncertain)"
  inside the text or omit it - never assert it as fact.
- The dialogue lines are automatic OCR transcriptions and may be imperfect.
  Treat them as potentially imperfect context ONLY: paraphrase rather than
  copying verbatim, and flag uncertainty instead of inventing a correction.
- Prefer explanation narration over reading dialogue aloud.
- If dialogue is genuinely important to understand the scene, return it as a
  segment with "type": "dialogue" and a "speaker" (use the character name if
  known, otherwise "unknown").
- Do NOT mention panel numbers or scan order to the viewer.
- Do not write narration for panels, shots, or scenes other than this one.
- Split the narration into 1-6 small segments; each segment should normally
  match one or a few visual shots. Keep each segment 1-3 sentences (2-8
  seconds when spoken).

For EVERY segment choose:
- visual_intent from: full_panel, smart_crop, character_closeup, face_closeup,
  object_closeup, action_crop, multi_panel
- camera from: static, slow_zoom_in, slow_zoom_out, pan_left, pan_right,
  pan_up, pan_down
- importance: a float from 0.0 to 1.0

Return ONLY a JSON object (no extra text, no markdown) shaped exactly like:

{
  "segments": [
    {
      "text": "The warrior steps out of the treeline, sword already drawn.",
      "panel_ids": ["p001_002", "p001_001"],
      "estimated_seconds": 4.5,
      "visual_intent": "action_crop",
      "camera": "slow_zoom_in",
      "importance": 0.8
    },
    {
      "text": "He is not alone.",
      "panel_ids": ["p001_001"],
      "estimated_seconds": 3.2,
      "visual_intent": "character_closeup",
      "camera": "static",
      "importance": 0.7,
      "type": "dialogue",
      "speaker": "unknown"
    }
  ]
}"""

SCENE_FACTS = """Scene: {scene_id}
Characters present: {characters}
Location(s): {locations}
Event(s): {events}
Scene summary: {summary}"""

PANEL_FACTS = """Panels in reading order (first to last):
{lines}"""

PANEL_LINE = "- {panel_id}: characters=[{characters}] event={event} dialogue={dialogue}"


def _facts(values, fallback="-"):
    kept = [str(value).strip() for value in (values or []) if value is not None]
    return ", ".join(kept) if kept and any(kept) else fallback


def build_script_prompt(scene, dialogue_by_panel=None, panel_context=None):
    """Prompt for a whole scene's segmented narration script.

    scene: scene dict from scenes/<page>_scenes.json.
    dialogue_by_panel: optional {panel_id: ocr text}.
    panel_context: optional {panel_id: {characters, event}} so the LLM can
        choose visual intents from the available panel information.
    """
    dialogue_by_panel = dialogue_by_panel or {}
    panel_context = panel_context or {}
    lines = []
    for pid in scene.get("panel_ids") or []:
        info = panel_context.get(pid) or {}
        dialogue = (dialogue_by_panel.get(pid) or "").strip()
        lines.append(
            PANEL_LINE.format(
                panel_id=pid,
                characters=_facts(
                    info.get("characters"), fallback="shown"
                ),
                event=_facts([info.get("event")]),
                dialogue=dialogue or "-",
            )
        )
    parts = [NARRATION_SEGMENT_RULES]
    parts.append(
        SCENE_FACTS.format(
            scene_id=scene.get("scene_id", "scene_001"),
            characters=_facts(scene.get("characters")),
            locations=_facts(scene.get("locations")),
            events=_facts(scene.get("events")),
            summary=_facts([scene.get("summary")]),
        )
    )
    parts.append(PANEL_FACTS.format(lines="\n".join(lines)))
    return "\n\n".join(parts)