"""Unit tests for change_detector.detect_changes (no database required)."""

from graphrag.change_detector import detect_changes


def test_added_modified_deleted():
    old = {
        "Policy": {"POL-1": {"status": "ACTIVE", "premium": 1000.0}},
        "Claim": {"CLM-1": {"amount": 100.0}},
    }
    new = {
        "Policy": {"POL-1": {"status": "EXPIRED", "premium": 1000.0}, "POL-2": {"status": "ACTIVE"}},
    }
    changes = detect_changes(old, new)
    assert [e["id"] for e in changes["added"]] == ["POL-2"]
    assert [e["id"] for e in changes["modified"]] == ["POL-1"]
    assert changes["modified"][0]["old_props"] == {"status": "ACTIVE", "premium": 1000.0}
    assert changes["modified"][0]["new_props"] == {"status": "EXPIRED", "premium": 1000.0}
    assert [e["id"] for e in changes["deleted"]] == ["CLM-1"]


def test_no_changes():
    snapshot = {"Policy": {"POL-1": {"status": "ACTIVE", "premium": 12500.0}}}
    assert detect_changes(snapshot, snapshot) == {"added": [], "modified": [], "deleted": []}


def test_list_order_insensitive():
    old = {"Coverage": {"COV-1": {"exclusions": ["a", "b"]}}}
    new = {"Coverage": {"COV-1": {"exclusions": ["b", "a"]}}}
    assert detect_changes(old, new)["modified"] == []


def test_empty_old_is_all_added():
    new = {"Policy": {"POL-1": {"status": "ACTIVE"}}}
    changes = detect_changes({}, new)
    assert len(changes["added"]) == 1
    assert changes["deleted"] == []


def test_none_snapshots_are_treated_as_empty():
    assert detect_changes(None, None) == {"added": [], "modified": [], "deleted": []}


def test_new_property_marks_modified():
    old = {"Policy": {"POL-1": {"status": "ACTIVE"}}}
    new = {"Policy": {"POL-1": {"status": "ACTIVE", "premium": 500.0}}}
    assert [e["id"] for e in detect_changes(old, new)["modified"]] == ["POL-1"]
