import json

def extract_rules(text, domain, client):
    """LLM извлекает логические правила из текста."""
    domain_hint = "медицинские правила (препарат -> эффект -> следствие)" if domain == "medicine" else "юридические правила (договор -> условие -> следствие)"
    
    prompt = f"""Проанализируй текст и извлеки {domain_hint}.
Текст:
{text[:3000]} 

Верни СТРОГО JSON в формате:
{{
  "rules": [
    {{"entity": "название (препарат/договор)", "effect": "свойство или условие", "consequence": "следствие (риск/недействительность)"}}
  ]
}}
Извлекай только четкие правила, где из одного логически вытекает другое."""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        data = json.loads(response.choices[0].message.content)
        return data.get("rules", [])
    except Exception as e:
        print(f"Ошибка парсинга: {e}")
        return []
