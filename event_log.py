import json
import os
import hashlib
from datetime import datetime

LOG_FILE = "events.jsonl"

def compute_hash(timestamp: str, event_type: str, payload: dict, prev_hash: str) -> str:
    """SHA-256 от всех полей события + хеш предыдущего."""
    content = timestamp + event_type + json.dumps(payload, ensure_ascii=False, sort_keys=True) + prev_hash
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def get_last_hash() -> str:
    """Возвращает хеш последнего события (или '0'*64, если лог пуст)."""
    if not os.path.exists(LOG_FILE):
        return "0" * 64
    last_line = None
    with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                last_line = line.strip()
    if not last_line:
        return "0" * 64
    try:
        return json.loads(last_line).get("hash", "0" * 64)
    except json.JSONDecodeError:
        return "0" * 64

def log_event(event_type: str, payload: dict):
    """Записывает событие с хеш-цепочкой."""
    timestamp = datetime.now().isoformat()
    prev_hash = get_last_hash()
    current_hash = compute_hash(timestamp, event_type, payload, prev_hash)
    
    event = {
        "timestamp": timestamp,
        "type": event_type,
        "payload": payload,
        "prev_hash": prev_hash,
        "hash": current_hash
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def read_events(limit: int = 20):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [json.loads(line) for line in lines[-limit:]]

def verify_chain():
    """Проверяет целостность всей цепочки. Возвращает (ok, сообщение)."""
    if not os.path.exists(LOG_FILE):
        return True, "Журнал пуст. Целостность тривиальна."
    
    prev_expected = "0" * 64
    count = 0
    with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return False, f"Строка {i}: повреждённый JSON."
            
            # Проверка 1: prev_hash должен совпадать с хешем предыдущего
            if event.get("prev_hash") != prev_expected:
                return False, f"Строка {i}: разрыв цепочки (prev_hash не совпадает)."
            
            # Проверка 2: хеш должен быть корректным
            recomputed = compute_hash(
                event["timestamp"], event["type"], event["payload"], event["prev_hash"]
            )
            if recomputed != event.get("hash"):
                return False, f"Строка {i}: хеш недействителен (данные подменены)."
            
            prev_expected = event["hash"]
            count += 1
    
    return True, f"✅ Целостность подтверждена: {count} событий, цепочка не разорвана."
