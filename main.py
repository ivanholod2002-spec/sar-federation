import chromadb, json
from openai import OpenAI
import os
from dotenv import load_dotenv
from z3_verifier import verify_medical_logic, EFFECT_TO_CONSEQUENCE

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_BASE"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="facts")

def ask_system(question):
    print(f"\n{'='*60}")
    print(f"Вопрос: {question}")
    
    results = collection.query(query_texts=[question], n_results=2)
    facts = results['documents'][0]
    sources = results['metadatas'][0]
    print(f"--- M1 (Найдено {len(facts)} фактов) ---")
    for i in range(len(facts)):
        print(f"  [{sources[i]['source']}] {facts[i]}")
    
    context = "\n".join(facts)
    allowed_effects = list(EFFECT_TO_CONSEQUENCE.keys())
    
    prompt = f"""Проанализируй вопрос, используя ТОЛЬКО эти факты:
{context}

Вопрос: {question}

Верни СТРОГО JSON. В поле "effect" выбери ТОЛЬКО ОДИН эффект из этого списка: {allowed_effects}
Если нужного эффекта нет в списке, напиши "нет в онтологии".

Формат:
{{"drug": "название препарата", "effect": "эффект из списка", "can_take": true или false, "reason": "краткое объяснение"}}"""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    print(f"--- M2 (LLM выдала JSON) ---\n{raw}")
    
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"❌ ОШИБКА ПАРСИНГА JSON: {e}")
        return
    
    drug = data.get("drug", "")
    effect = data.get("effect", "")
    can_take = data.get("can_take", None)
    reason = data.get("reason", "")
    
    print(f"--- M3 (Z3 проверяет: {drug} → {effect}) ---")
    result, msg = verify_medical_logic(drug, effect)
    
    if result is True:
        print(f"✅ {msg}")
        print(f"ВЕРДИКТ СИСТЕМЫ: {'РАЗРЕШЕНО' if can_take else 'ЗАПРЕЩЕНО'} | {reason}")
    elif result is False:
        print(f"❌ {msg}")
        print(f"🚫 ВЕРДИКТ: СИСТЕМА ЗАБЛОКИРОВАЛА ОТВЕТ (логика не подтверждена Z3)")
    else:
        print(f"⚠️  {msg}")
        print(f"👤 ВЕРДИКТ: Требуется человек-эксперт (нет данных в онтологии)")

# Тесты
ask_system("Аспирин опасен для желудка?")
ask_system("Можно ли давать парацетамол при головной боли?")
ask_system("Можно ли давать аспирин при язве?")
