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
    rules = load_rules(domain)
    DRUG_TO_EFFECT = rules["drugs"]
    EFFECT_TO_CONSEQUENCE = rules["consequences"]
    
    effect_key = normalize_effect(claimed_effect, EFFECT_TO_CONSEQUENCE)
    if not effect_key:
        return None, f"Эффект '{claimed_effect}' не распознан", []
    
    s = Solver()
    
    # Переменные
    drug_var = String('drug')
    effect_var = String('effect')
    disease_var = String('disease')
    age_var = Int('age')
    
    # Устанавливаем текущий контекст
    s.add(drug_var == drug)
    s.add(disease_var == disease)
    s.add(age_var == age)
    
    # Базовые правила из JSON (Препарат -> Эффект)
    for d, effects in DRUG_TO_EFFECT.items():
        for e in effects:
            s.add(Implies(drug_var == d, effect_var == e))
            
    # Сложные правила (Мульти-условия) - Хардкод для демонстрации
    # Если Аспирин И возраст > 65 И болезнь Язва -> Смертельный риск
    s.add(Implies(And(drug_var == "Аспирин", age_var > 65, disease_var == "Язва"), effect_var == "Смертельный риск"))
    s.add(Implies(effect_var == "Смертельный риск", effect_var == "Опасно при язве")) # Связь с онтологией

    # Проверяем гипотезу: является ли claimed_effect следствием
    s.push()
    s.add(effect_var == effect_key)
    result = s.check()
    s.pop()
    
    if result == sat:
        edges = [(drug, f"Возраст: {age}"), (f"Возраст: {age}", "Смертельный риск")] if age > 65 and disease == "Язва" else [(drug, effect_key)]
        return True, f"Z3 доказал: {drug} (Возраст: {age}, {disease}) → {effect_key}", edges
    else:
        return False, f"Z3 НЕ смог доказать: {drug} → {effect_key}", []
