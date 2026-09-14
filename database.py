import sqlite3, json
from pathlib import Path
DB=Path(__file__).resolve().parents[1]/"database"/"truthforge.db"

def init_db():
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as c:
        c.execute('''CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, timestamp TEXT, question TEXT, response TEXT, overall_risk REAL, verdict TEXT, claims INTEGER, model_version TEXT, result_json TEXT)''')

def save_case(case):
    with sqlite3.connect(DB) as c:
        c.execute("INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,?,?,?,?)", (case['case_id'],case['timestamp'],case['question'],case['response'],case['overall_risk'],case['verdict'],case['claims'],case['model_version'],json.dumps(case)))

def list_cases():
    with sqlite3.connect(DB) as c:
        c.row_factory=sqlite3.Row
        return [dict(r) for r in c.execute("SELECT * FROM cases ORDER BY timestamp DESC").fetchall()]

def get_case(case_id):
    with sqlite3.connect(DB) as c:
        c.row_factory=sqlite3.Row
        r=c.execute("SELECT * FROM cases WHERE case_id=?",(case_id,)).fetchone()
        return dict(r) if r else None
