from z3 import *
import json

def load_rules(domain: str = "medicine"):
    filename = "rules_law.json" if domain == "law" else "rules_db.json"
    with open(filename, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def normalize_effect(raw_effect, consequences):
    raw_lower = str(raw_effect).lower()
    for key in consequences.keys():
        if key.lower() in raw_lower:
            return key
    return None

def verify_logic(drug, claimed_effect, domain: str = "medicine"):
    try:
        rules = load_rules(domain)
    except Exception:
        return None, f"Ошибка загрузки правил для {domain}", []
        
    DRUG_TO_EFFECT = rules["drugs"]
    EFFECT_TO_CONSEQUENCE = rules["consequences"]
    
    effect_key = normalize_effect(claimed_effect, EFFECT_TO_CONSEQUENCE)
    if not effect_key:
        return None, f"Эффект '{claimed_effect}' не распознан в онтологии '{domain}'", []
    
    s = Solver()
    effect_vars = {e: Bool(f"effect_{i}") for i, e in enumerate(EFFECT_TO_CONSEQUENCE)}
    
    if drug in DRUG_TO_EFFECT:
        for actual_effect in DRUG_TO_EFFECT[drug]:
            if actual_effect in effect_vars:
                s.add(effect_vars[actual_effect] == True)
    
    for eff, cons in EFFECT_TO_CONSEQUENCE.items():
        if eff in effect_vars and cons in effect_vars:
            s.add(Implies(effect_vars[eff], effect_vars[cons]))
    
    if effect_key not in effect_vars:
        return None, f"Эффект '{effect_key}' отсутствует в онтологии '{domain}'", []
    
    s.push()
    s.add(effect_vars[effect_key] == False)
    result = s.check()
    s.pop()
    
    graph_edges = []
    if result == unsat:
        if drug in DRUG_TO_EFFECT:
            for eff in DRUG_TO_EFFECT[drug]:
                graph_edges.append((drug, eff))
                if eff in EFFECT_TO_CONSEQUENCE:
                    graph_edges.append((eff, EFFECT_TO_CONSEQUENCE[eff]))
        return True, f"Z3 доказал: {drug} → {effect_key} [{domain}]", graph_edges
    else:
        return False, f"Z3 НЕ смог доказать: {drug} → {effect_key} [{domain}]", []
