"""Bounding-box geometry helpers for panel detection.

Boxes are [x, y, width, height] in image pixel coordinates.
"""


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def is_inside(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax >= bx
        and ay >= by
        and (ax + aw) <= (bx + bw)
        and (ay + ah) <= (by + bh)
    )


def filter_boxes(boxes, page_w, page_h, params):
    """Apply size/aspect/edge filters and remove duplicate/nested boxes.

    Each input item is a dict with keys "box" ([x,y,w,h]) and optionally
    "area" and "confidence". Returns the cleaned list, largest duplicates
    kept, sorted top-to-bottom then left-to-right.
    """
    min_area = int(params.get("min_area", 3000))
    min_area_ratio = float(params.get("min_area_ratio", 0.0015))
    max_area_ratio = float(params.get("max_area_ratio", 0.95))
    min_side = int(params.get("min_side", 20))
    max_aspect = float(params.get("max_aspect_ratio", 4.0))
    min_conf = float(params.get("min_confidence", 0.6))
    dup_iou = float(params.get("duplicate_iou", 0.5))
    drop_edge = bool(params.get("drop_edge_touching", True))

    page_area = page_w * page_h
    min_area_abs = max(min_area, min_area_ratio * page_area)
    max_area_abs = max_area_ratio * page_area

    kept = []
    for item in boxes:
        x, y, w, h = item["box"]
        area = int(item.get("area", w * h))
        if w <= 0 or h <= 0:
            continue
        if area < min_area_abs or area > max_area_abs:
            continue
        if w < min_side or h < min_side:
            continue
        if max(w, h) / min(w, h) > max_aspect:
            continue
        if float(item.get("confidence", 1.0)) < min_conf:
            continue
        if drop_edge and (x <= 0 or y <= 0 or (x + w) >= page_w or (y + h) >= page_h):
            continue
        kept.append(item)

    kept.sort(key=lambda item: item["area"], reverse=True)
    dedup = []
    for item in kept:
        box = item["box"]
        redundant = False
        for other in dedup:
            obox = other["box"]
            if iou(box, obox) >= dup_iou:
                redundant = True
                break
            if is_inside(box, obox):
                redundant = True
                break
        if not redundant:
            dedup.append(item)

    dedup.sort(key=lambda item: (item["box"][1], item["box"][0]))
    max_panels = int(params.get("max_panels", 40))
    return dedup[:max_panels]