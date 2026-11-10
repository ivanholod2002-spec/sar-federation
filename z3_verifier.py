from z3 import *
import json

def load_rules(domain: str = "medicine"):
    filename = "rules_law.json" if domain == "law" else "rules_db.json"
    try:
        with open(filename, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except:
        return {"drugs": {}, "consequences": {}}

def normalize_effect(raw_effect, consequences):
    raw_lower = str(raw_effect).lower()
    for key in consequences.keys():
        if key.lower() in raw_lower:
            return key
    return None

def verify_logic(drug, claimed_effect, domain: str = "medicine", age: int = 30, disease: str = "none"):
    # ЗАЩИТА: Если LLM не нашла данных, мы не даем Z3 это проверить
    if str(drug).lower() in ["none", "нет_данных", "null", ""] or str(claimed_effect).lower() in ["none", "нет_данных", "null", ""]:
        return None, "В базе знаний нет конкретных фактов для проверки", []

    rules = load_rules(domain)
    DRUG_TO_EFFECT = rules.get("drugs", {})
    EFFECT_TO_CONSEQUENCE = rules.get("consequences", {})
    
    effect_key = normalize_effect(claimed_effect, EFFECT_TO_CONSEQUENCE)
    if not effect_key:
        return None, f"Понятие '{claimed_effect}' не распознано в онтологии '{domain}'", []
    
    s = Solver()
    
    # Переменные
    drug_var = String('drug')
    effect_var = String('effect')
    disease_var = String('disease')
    age_var = Int('age')
    
    s.add(drug_var == drug)
    s.add(disease_var == disease)
    s.add(age_var == age)
    
    # Правила
    for d, effects in DRUG_TO_EFFECT.items():
        for e in effects:
            s.add(Implies(drug_var == d, effect_var == e))
            
    # Проверяем
    s.push()
    s.add(effect_var == effect_key)
    result = s.check()
    s.pop()
    
    if result == sat:
        edges = [(drug, effect_key)]
        return True, f"Z3 доказал: {drug} → {effect_key}", edges
    else:
        return False, f"Z3 НЕ смог доказать: {drug} → {effect_key}", []
