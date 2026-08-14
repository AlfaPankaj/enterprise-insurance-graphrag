"""AppTest verification: audit_ui page + dashboard page (Phase 4 additions)."""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402


def check_audit_ui():
    at = AppTest.from_file(str(ROOT / "src" / "graphrag" / "audit_ui.py"),
                           default_timeout=120)
    at.run()
    print("audit_ui exceptions:", len(at.exception))
    for e in at.exception:
        print("  EXC:", str(e.value)[:300])
    print("audit_ui error elements:", len(at.error))
    print("audit_ui titles:", [t.value for t in at.title])
    # selectbox with query history + tabs should be present
    print("audit_ui selectboxes:", len(at.selectbox), "| tabs:", len(at.tabs))
    # export buttons for JSON/HTML/PDF (st.download_button element type)
    dl = at.get("download_button")
    print("audit_ui download buttons:", [b.label for b in dl])
    assert not at.exception and not at.error, "audit_ui raised"
    assert len(at.selectbox) >= 1, "query history selectbox missing"
    assert len(dl) == 3, f"expected 3 export buttons, got {len(dl)}"
    print("AUDIT_UI OK")


def check_dashboard():
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=120)
    at.run()
    print("dashboard exceptions at start:", len(at.exception))
    # run a live query -> exercises the new cypher expander + audit caption
    at.text_input[0].set_value("Does claim CLM-0003 have a fraud flag?")
    at.button[0].click().run()
    print("dashboard exceptions after query:", len(at.exception))
    for e in at.exception:
        print("  EXC:", str(e.value)[:300])
    print("dashboard error elements:", len(at.error))
    print("dashboard page_links:", len(at.get("page_link")))
    print("dashboard markdown count:", len(at.markdown))
    assert not at.exception and not at.error, "dashboard raised after query"
    # fraud ground-truth panel: the CLM-0003 query must produce a comparison
    # row with the demo dataset's label (fraud) and a correct verdict
    gt_frames = [d.value for d in at.dataframe
                 if getattr(d.value, "columns", None) is not None
                 and any(c.lower() == "claim" for c in d.value.columns)]
    print("dashboard ground-truth dataframes:", len(gt_frames))
    assert gt_frames, "fraud ground-truth dataframe missing"
    frame = gt_frames[0]
    claim_col = next(c for c in frame.columns if c.lower() == "claim")
    match_col = next(c for c in frame.columns if c.lower() == "match")
    row = frame[frame[claim_col] == "CLM-0003"]
    assert not row.empty, "CLM-0003 missing from ground-truth panel"
    assert row.iloc[0][match_col] == "✅ Correct", \
        f"CLM-0003 verdict not correct: {row.iloc[0].to_dict()}"
    print("ground-truth row:", row.iloc[0].to_dict())
    print("DASHBOARD OK")


check_audit_ui()
check_dashboard()
print("PHASE 4 UI OK")
