# halo_backend_fastapi.py
"""
HALO Backend — Hackathon Edition (with Hugging Face, usage stats, messaging)
Demo prototype using in-memory stores so everything is visible instantly.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import datetime
import uuid

# Hugging Face
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------------- Config ----------------
HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# ---------------- Load NLP Model ----------------
print("Loading Hugging Face model... this may take a little while on first run.")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
text_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

app = FastAPI(title="HALO Backend — Hackathon Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- In-memory demo storage ----------------
_children: Dict[str, Dict[str, Any]] = {
    "child1": {"name": "Demo Child 1", "parent_uid": "parent1"},
    "child2": {"name": "Demo Child 2", "parent_uid": "parent1"},
}

_parents: Dict[str, Dict[str, Any]] = {
    "parent1": {"name": "Demo Parent 1"}
}

# Alerts (hardcoded + dynamic)
_alerts: List[Dict[str, Any]] = [
    {
        "id": "a1",
        "child_uid": "child1",
        "type": "message",
        "text": "This looks suspicious",
        "severity": "high",
        "created_at": (datetime.datetime.utcnow() - datetime.timedelta(hours=1)).isoformat(),
        "acknowledged": False,
    },
    {
        "id": "a2",
        "child_uid": "child1",
        "type": "click_link",
        "text": "Clicked on unknown link",
        "severity": "medium",
        "created_at": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat(),
        "acknowledged": False,
    },
    {
        "id": "a3",
        "child_uid": "child2",
        "type": "sos",
        "text": "SOS triggered — battery low and lost GPS",
        "severity": "high",
        "created_at": (datetime.datetime.utcnow() - datetime.timedelta(days=2)).isoformat(),
        "acknowledged": False,
    },
]

# Journals (demo)
_journals: List[Dict[str, Any]] = [
    {"id": "j1", "uid": "child1", "date": "2025-09-17", "good": ["Studied math"], "bad": ["Too much screen time"]},
    {"id": "j2", "uid": "child1", "date": "2025-09-16", "good": ["Helped mom"], "bad": ["Skipped homework"]},
    {"id": "j3", "uid": "child2", "date": "2025-09-15", "good": ["Practiced piano"], "bad": ["Fell asleep in class"]},
]

# Reminders (per-child)
_reminders: Dict[str, List[Dict[str, Any]]] = {
    "child1": [{"uid": "child1", "type": "water_break", "interval_minutes": 60, "created_at": datetime.datetime.utcnow().isoformat()}],
    "child2": [],
}

# Usage data (per child) - includes screen time seconds and (optionally) unlocks
_usage: Dict[str, Dict[str, Any]] = {
    "child1": {
        "total_seconds": 7200,  # 2 hours
        "unlock_count": 35,
        "summary": [
            {"package": "YouTube", "seconds": 3000},
            {"package": "Instagram", "seconds": 1800},
            {"package": "Google Docs", "seconds": 1200},
            {"package": "Duolingo", "seconds": 1200},
        ],
    },
    "child2": {
        "total_seconds": 3600,
        "unlock_count": 18,
        "summary": [
            {"package": "YouTube", "seconds": 1600},
            {"package": "Minecraft", "seconds": 1200},
            {"package": "Zoom", "seconds": 800},
        ],
    },
}

# Messages store (in-memory)
_messages: List[Dict[str, Any]] = [
    # sample existing exchanged messages for demo
    {
        "id": str(uuid.uuid4()),
        "from": "parent1",
        "to": "child1",
        "text": "Hi — did you finish your homework?",
        "ts": (datetime.datetime.utcnow() - datetime.timedelta(hours=2)).isoformat(),
    },
    {
        "id": str(uuid.uuid4()),
        "from": "child1",
        "to": "parent1",
        "text": "Yes! Done. Coming to dinner soon.",
        "ts": (datetime.datetime.utcnow() - datetime.timedelta(hours=1, minutes=30)).isoformat(),
    },
]

# ---------------- Schemas ----------------
class JournalEntry(BaseModel):
    uid: str
    date: Optional[str]
    good: List[str]
    bad: List[str]

class Reminder(BaseModel):
    uid: str
    type: str
    interval_minutes: Optional[int]

class ReportText(BaseModel):
    child_uid: str
    text_content: str

class MessageIn(BaseModel):
    sender: str  # "parent1" or "child1"
    recipient: str  # "child1" or "parent1"
    text: str

# ---------------- Utilities ----------------
def analyze_text(text: str) -> Dict[str, Any]:
    """Use Hugging Face text-classification to assess severity (sentiment proxy)."""
    try:
        res = text_pipe(text[:512])[0]
        label = res.get("label", "")
        score = float(res.get("score", 0.0))
        severity = "low"
        if label.upper() in ("NEGATIVE", "LABEL_1") and score > 0.9:
            severity = "high"
        elif label.upper() in ("NEGATIVE", "LABEL_1") and score > 0.6:
            severity = "medium"
        return {"label": label, "score": score, "severity": severity}
    except Exception as e:
        return {"error": str(e)}

def detect_phishing(text: str) -> bool:
    lower = text.lower()
    keys = ["click here", "login", "verify", "password", "bank", "account", "http", "bit.ly", "tinyurl"]
    return any(k in lower for k in keys)

def make_alert(child_uid: str, typ: str, text: str, severity: str = "low") -> Dict[str, Any]:
    a = {
        "id": str(uuid.uuid4()),
        "child_uid": child_uid,
        "type": typ,
        "text": text,
        "severity": severity,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "acknowledged": False,
    }
    _alerts.insert(0, a)
    return a

# ---------------- Endpoints ----------------

# Demo login endpoints
@app.post("/parent/login")
async def parent_login(payload: Dict[str, str]):
    if payload.get("username") == "parent1" and payload.get("password") == "demo123":
        return {"ok": True, "parent_uid": "parent1"}
    return {"ok": False, "error": "Invalid credentials"}

@app.post("/child/login")
async def child_login(payload: Dict[str, str]):
    uid = payload.get("uid")
    if uid in _children:
        return {"ok": True, "child_uid": uid, "parent_uid": _children[uid]["parent_uid"]}
    return {"ok": False, "error": "Invalid child UID"}

@app.get("/parent/children/{parent_uid}")
async def parent_children(parent_uid: str):
    # return list of children for this parent
    out = [{"uid": uid, "name": meta["name"]} for uid, meta in _children.items() if meta["parent_uid"] == parent_uid]
    return {"ok": True, "children": out}

# Child reports suspicious text -> use HF + phishing heuristic, create alert
@app.post("/child/report_text")
async def report_text(payload: ReportText):
    phishing = detect_phishing(payload.text_content)
    analysis = analyze_text(payload.text_content)
    severity = analysis.get("severity", "low") if isinstance(analysis, dict) else "low"
    alert = make_alert(payload.child_uid, "reported_text", payload.text_content, severity)
    return {"ok": True, "phishing": phishing, "analysis": analysis, "alert_id": alert["id"]}

# SOS
@app.post("/child/sos")
async def child_sos(payload: Dict[str, Any]):
    uid = payload.get("uid")
    note = payload.get("note", "")
    lat = payload.get("lat")
    lng = payload.get("lng")
    text = f"SOS: {note} location={lat},{lng}"
    alert = make_alert(uid, "sos", text, "high")
    return {"ok": True, "sos_id": alert["id"]}

# Child journaling
@app.post("/child/journal")
async def save_journal(entry: JournalEntry):
    date = entry.date or datetime.date.today().isoformat()
    j = {"id": str(uuid.uuid4()), "uid": entry.uid, "date": date, "good": entry.good, "bad": entry.bad}
    _journals.insert(0, j)
    return {"ok": True, "saved": j}

# Reminder endpoints
@app.post("/child/reminder")
async def set_reminder(rem: Reminder):
    r = {"uid": rem.uid, "type": rem.type, "interval_minutes": rem.interval_minutes or 0, "created_at": datetime.datetime.utcnow().isoformat()}
    _reminders.setdefault(rem.uid, []).append(r)
    return {"ok": True, "reminder": r}

@app.get("/child/reminders/{uid}")
async def get_reminders(uid: str):
    return {"ok": True, "reminders": _reminders.get(uid, [])}

# Location (demo)
@app.post("/child/location")
async def update_location(payload: Dict[str, Any]):
    uid = payload.get("uid")
    lat = payload.get("lat")
    lng = payload.get("lng")
    ts = payload.get("ts") or datetime.datetime.utcnow().isoformat()
    return {"ok": True, "location": {"uid": uid, "lat": lat, "lng": lng, "ts": ts}}

# Usage summary (includes numeric totals and breakdown)
@app.get("/child/usage_summary/{uid}")
async def usage_summary(uid: str):
    data = _usage.get(uid)
    if data is None:
        return {"ok": True, "total_seconds": 0, "unlock_count": 0, "summary": []}
    return {
        "ok": True,
        "total_seconds": data.get("total_seconds", 0),
        "unlock_count": data.get("unlock_count", 0),
        "summary": data.get("summary", []),
    }

# Flashcards & blacklist
@app.get("/child/flashcards")
async def flashcards():
    return {"ok": True, "cards": [
        {"q": "Is it safe to click links from strangers?", "a": "No"},
        {"q": "Should you share your password?", "a": "No"},
        {"q": "If someone asks for your location, what should you do?", "a": "Ask a parent"},
    ]}

@app.get("/child/blacklist")
async def blacklist():
    return {"ok": True, "blacklist": ["badsite.com", "phish.com", "malware.net"]}

# Messaging endpoints
@app.post("/message/send")
async def send_message(msg: MessageIn):
    m = {
        "id": str(uuid.uuid4()),
        "from": msg.sender,
        "to": msg.recipient,
        "text": msg.text,
        "ts": datetime.datetime.utcnow().isoformat(),
    }
    _messages.append(m)
    return {"ok": True, "message": m}

@app.get("/parent/messages/{parent_uid}")
async def parent_messages(parent_uid: str):
    # return messages to/from parent and their children
    child_uids = [uid for uid, meta in _children.items() if meta["parent_uid"] == parent_uid]
    out = [m for m in _messages if (m["from"] == parent_uid and (m["to"] in child_uids)) or (m["to"] == parent_uid and (m["from"] in child_uids))]
    out_sorted = sorted(out, key=lambda x: x["ts"], reverse=False)
    return {"ok": True, "messages": out_sorted}

@app.get("/child/messages/{child_uid}")
async def child_messages(child_uid: str):
    out = [m for m in _messages if m["from"] == child_uid or m["to"] == child_uid]
    out_sorted = sorted(out, key=lambda x: x["ts"], reverse=False)
    return {"ok": True, "messages": out_sorted}

# Parent endpoints: alerts, trends, journals, journal stats, find child
@app.get("/parent/alerts/{parent_uid}")
async def parent_alerts(parent_uid: str):
    child_uids = [uid for uid, meta in _children.items() if meta["parent_uid"] == parent_uid]
    out = [a for a in _alerts if a["child_uid"] in child_uids]
    return {"ok": True, "alerts": out}

@app.put("/parent/alert/{alert_id}")
async def acknowledge_alert(alert_id: str, payload: Dict[str, Any]):
    for a in _alerts:
        if a["id"] == alert_id:
            a["acknowledged"] = payload.get("acknowledged", True)
            if "feedback" in payload:
                a["feedback"] = payload["feedback"]
            return {"ok": True, "message": f"Alert {alert_id} updated"}
    raise HTTPException(status_code=404, detail="Alert not found")

@app.get("/parent/alerts/trends/{parent_uid}")
async def alert_trends(parent_uid: str):
    child_uids = [uid for uid, meta in _children.items() if meta["parent_uid"] == parent_uid]
    daily: Dict[str, int] = {}
    for a in _alerts:
        if a["child_uid"] in child_uids:
            d = a["created_at"][:10]
            daily[d] = daily.get(d, 0) + 1
    trends = [{"date": k, "count": v} for k, v in sorted(daily.items())]
    return {"ok": True, "trends": trends}

@app.get("/parent/journal_stats/{parent_uid}")
async def journal_stats(parent_uid: str):
    child_uids = [uid for uid, meta in _children.items() if meta["parent_uid"] == parent_uid]
    good = 0
    bad = 0
    for j in _journals:
        if j["uid"] in child_uids:
            good += len(j.get("good", []))
            bad += len(j.get("bad", []))
    return {"ok": True, "good": good, "bad": bad}

@app.get("/parent/journals/{parent_uid}")
async def parent_journals(parent_uid: str):
    child_uids = [uid for uid, meta in _children.items() if meta["parent_uid"] == parent_uid]
    res = [j for j in _journals if j["uid"] in child_uids]
    return {"ok": True, "journals": res}

@app.get("/parent/find_child/{child_uid}")
async def find_child(child_uid: str):
    # demo static location
    return {"ok": True, "location": {"uid": child_uid, "lat": 12.9716, "lng": 77.5946, "ts": datetime.datetime.utcnow().isoformat()}}

@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.datetime.utcnow().isoformat()}

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("halo_backend_fastapi:app", host="0.0.0.0", port=8000, reload=True)
