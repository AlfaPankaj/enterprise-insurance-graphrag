"""AppTest verification of the dashboard's live query flow (no browser needed)."""
import os
import sys

# ASCII-safe console output on Windows cp1252 terminals
sys.stdout.reconfigure(errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.join(ROOT, "dashboard.py"), default_timeout=120)
at.run()

# Render the full interactive flow: type a query, click Run
at.text_input[0].set_value("Does claim CLM-0003 have a fraud flag?")
at.button[0].click().run()

print("exceptions after query:", len(at.exception))
for e in at.exception:
    print("  EXC:", str(e.value)[:300])
print("error elements:", len(at.error))
print("titles:", [t.value for t in at.title])

# Report what actually rendered so we can confirm the chart + metrics
rendered = {}
for el in at:
    rendered.setdefault(el.type, 0)
    rendered[el.type] += 1
print("element inventory:", rendered)

assert not at.exception and not at.error, "dashboard raised after query"
print("DASHBOARD FLOW OK")
