"""Change Data Capture — diff old vs new entity snapshots (Shot 1 core).

Entity snapshots have the shape ``{label: {entity_id: props}}``.

``detect_changes(old, new)`` returns::

    {
      "added":    [{"label", "id", "props"}],
      "modified": [{"label", "id", "old_props", "new_props"}],
      "deleted":  [{"label", "id", "props"}],
    }

An entity is ``modified`` when the set of props differs (list props compare
order-insensitively, numbers compare numerically).
"""

from __future__ import annotations

from typing import Any


def _props_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, list) or isinstance(vb, list):
            if sorted(va) != sorted(vb):
                return False
        elif va != vb:
            return False
    return True


def detect_changes(old: dict[str, dict[str, dict]] | None,
                   new: dict[str, dict[str, dict]] | None) -> dict[str, list[dict]]:
    """Return added / modified / deleted entities between two snapshots."""
    old = old or {}
    new = new or {}
    added: list[dict] = []
    modified: list[dict] = []
    deleted: list[dict] = []

    for label, entities in new.items():
        old_label = old.get(label, {})
        for eid, props in entities.items():
            if eid not in old_label:
                added.append({"label": label, "id": eid, "props": props})
            elif not _props_equal(old_label[eid], props):
                modified.append(
                    {"label": label, "id": eid, "old_props": old_label[eid], "new_props": props}
                )

    for label, entities in old.items():
        new_label = new.get(label, {})
        for eid, props in entities.items():
            if eid not in new_label:
                deleted.append({"label": label, "id": eid, "props": props})

    return {"added": added, "modified": modified, "deleted": deleted}
