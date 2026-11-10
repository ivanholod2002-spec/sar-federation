import json
import os

LOG_FILE = "events.jsonl"

def compute_metrics():
    """Считает статистику по журналу событий."""
    if not os.path.exists(LOG_FILE):
        return {
            "total_queries": 0,
            "verified_true": 0,
            "verified_false": 0,
            "escalated_to_expert": 0,
            "rules_added": 0,
            "docs_uploaded": 0,
            "verification_rate": 0.0,
            "escalation_rate": 0.0
        }
    
    total_queries = 0
    verified_true = 0
    verified_false = 0
    escalated = 0
    rules_added = 0
    docs_uploaded = 0
    
    with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            etype = event.get("type")
            if etype == "query_received":
                total_queries += 1
            elif etype == "m3_verified":
                valid = event["payload"].get("valid")
                if valid is True:
                    verified_true += 1
                elif valid is False:
                    verified_false += 1
                else:
                    escalated += 1
            elif etype == "rule_added":
                rules_added += 1
            elif etype == "document_uploaded":
                docs_uploaded += 1
    
    # Процент запросов, которые система взяла на себя (true или false)
    verification_rate = 0.0
    escalation_rate = 0.0
    if total_queries > 0:
        verification_rate = ((verified_true + verified_false) / total_queries) * 100
        escalation_rate = (escalated / total_queries) * 100
    
    return {
        "total_queries": total_queries,
        "verified_true": verified_true,
        "verified_false": verified_false,
        "escalated_to_expert": escalated,
        "rules_added": rules_added,
        "docs_uploaded": docs_uploaded,
        "verification_rate": round(verification_rate, 1),
        "escalation_rate": round(escalation_rate, 1)
    }
