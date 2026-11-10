import streamlit as st
import streamlit.components.v1 as components
import chromadb, json, uuid
from openai import OpenAI
import os
from dotenv import load_dotenv
from z3_verifier import verify_logic, load_rules
from pyvis.network import Network
from event_log import log_event, verify_chain
from metrics import compute_metrics
from extractor import extract_rules
from pypdf import PdfReader
import io

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_BASE"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

st.set_page_config(page_title="САР-ФЕДЕРАЦИЯ", layout="wide")
st.title("🧠 САР-ФЕДЕРАЦИЯ: Медицина + Право")

domain = st.radio(
    "Выберите домен (Federation Layer):",
    options=["medicine", "law", "federation"],
    format_func=lambda x: "🏥 Медицина" if x == "medicine" else ("⚖️ Право" if x == "law" else "🔗 Федерация (Кросс-домен)"),
    horizontal=True
)

collection_med = chroma_client.get_or_create_collection(name="facts")
collection_law = chroma_client.get_or_create_collection(name="facts_law")

with st.sidebar:
    st.header("📊 Метрики доверия")
    m = compute_metrics()
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Запросов", m["total_queries"])
        st.metric("✅ Доказано", m["verified_true"])
    with col_b:
        st.metric("Правил", m["rules_added"])
        st.metric("❌ Заблок.", m["verified_false"])
    
    st.progress(min(m["verification_rate"] / 100, 1.0))
    st.caption(f"Верификация: {m['verification_rate']}%")

    st.divider()
    st.header("📁 Загрузка документов (M1)")
    uploaded_file = st.file_uploader("Загрузить документ (TXT или PDF)", type=["txt", "pdf"])
    
    if uploaded_file:
        st.success(f"Файл готов: {uploaded_file.name}")
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        def read_file_text():
            uploaded_file.seek(0)
            if file_extension == "pdf":
                reader = PdfReader(io.BytesIO(uploaded_file.read()))
                txt = ""
                for page in reader.pages:
                    txt += page.extract_text() or ""
                return txt
            else:
                return uploaded_file.read().decode("utf-8")
                
        if st.button("1. Проглотить документ (RAG)"):
            progress_bar = st.progress(0, text="Читаю файл...")
            text = read_file_text()
            # Делаем куски по 1500 символов для скорости
            chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
            
            target_cols = [collection_med, collection_law] if domain == "federation" else ([collection_law] if domain == "law" else [collection_med])
            
            # Отправляем ПАЧКАМИ по 100 штук
            batch_size = 100
            total_chunks = len(chunks)
            
            for col in target_cols:
                for i in range(0, total_chunks, batch_size):
                    batch = chunks[i:i+batch_size]
                    ids = [str(uuid.uuid4()) for _ in batch]
                    metas = [{"source": uploaded_file.name} for _ in batch]
                    
                    col.add(documents=batch, metadatas=metas, ids=ids)
                    
                    # Обновляем полоску прогресса
                    percent = min((i + batch_size) / total_chunks, 1.0)
                    progress_bar.progress(percent, text=f"Загрузка в базу: {int(percent*100)}%")
            
            log_event("document_uploaded", {"file": uploaded_file.name, "chunks": total_chunks, "domain": domain})
            st.success(f"✅ Загружено {total_chunks} фрагментов!")
                
        st.divider()
        st.markdown("🤖 **Авто-извлечение правил (M5)**")
        if st.button("2. Извлечь правила из текста"):
            with st.spinner("LLM читает документ..."):
                text = read_file_text()
                if domain == "federation":
                    st.session_state['extracted_med'] = extract_rules(text[:3000], "medicine", client)
                    st.session_state['extracted_law'] = extract_rules(text[:3000], "law", client)
                else:
                    st.session_state['extracted_rules'] = extract_rules(text[:3000], domain, client)

        # (Логика одобрения правил остается такой же, пропускаю для краткости, она ниже)
        if domain == "federation":
            if 'extracted_med' in st.session_state and st.session_state['extracted_med']:
                st.info(f"Медицинских правил: {len(st.session_state['extracted_med'])}")
            if 'extracted_law' in st.session_state and st.session_state['extracted_law']:
                st.info(f"Юридических правил: {len(st.session_state['extracted_law'])}")
            if st.button("✅ Одобрить ВСЕ правила"):
                if 'extracted_med' in st.session_state:
                    rules_m = load_rules("medicine")
                    for r in st.session_state['extracted_med']:
                        rules_m["drugs"].setdefault(r.get('entity'), []).append(r.get('effect'))
                        rules_m["consequences"][r.get('effect')] = r.get('consequence')
                    with open("rules_db.json", "w", encoding="utf-8") as f: json.dump(rules_m, f, ensure_ascii=False, indent=2)
                if 'extracted_law' in st.session_state:
                    rules_l = load_rules("law")
                    for r in st.session_state['extracted_law']:
                        rules_l["drugs"].setdefault(r.get('entity'), []).append(r.get('effect'))
                        rules_l["consequences"][r.get('effect')] = r.get('consequence')
                    with open("rules_law.json", "w", encoding="utf-8") as f: json.dump(rules_l, f, ensure_ascii=False, indent=2)
                st.success("✅ Все правила добавлены в онтологии Z3!")
                del st.session_state['extracted_med']; del st.session_state['extracted_law']
        else:
            if 'extracted_rules' in st.session_state and st.session_state['extracted_rules']:
                st.info(f"Найдено правил: {len(st.session_state['extracted_rules'])}")
                for r in st.session_state['extracted_rules']: st.caption(f"• {r.get('entity')} ➔ {r.get('effect')} ➔ {r.get('consequence')}")
                if st.button("✅ Одобрить и добавить в Z3"):
                    rules_update = load_rules(domain)
                    for r in st.session_state['extracted_rules']:
                        rules_update["drugs"].setdefault(r.get('entity'), []).append(r.get('effect'))
                        rules_update["consequences"][r.get('effect')] = r.get('consequence')
                    filename = "rules_law.json" if domain == "law" else "rules_db.json"
                    with open(filename, "w", encoding="utf-8") as f: json.dump(rules_update, f, ensure_ascii=False, indent=2)
                    st.success("✅ Правила добавлены в Z3!"); del st.session_state['extracted_rules']

    st.divider()
    st.header("📜 Журнал (L1)")
    if st.button("🔒 Проверить целостность"):
        ok, msg = verify_chain()
        if ok: st.success(msg)
        else: st.error(msg)

default_question = "Можно ли заставить пациента подписать договор на эксперимент?" if domain == "federation" else ("Можно ли давать ибупрофен при гастрите?" if domain == "medicine" else "Что будет, если договор аренды заключен в устной форме?")
question = st.text_input("Введите вопрос:", default_question)

if st.button("Запустить анализ"):
    if question:
        log_event("query_received", {"question": question, "domain": domain})
        col1, col2 = st.columns(2)
        with col1:
            if domain == "federation":
                res_med = collection_med.query(query_texts=[question], n_results=1)
                res_law = collection_law.query(query_texts=[question], n_results=1)
                facts_med, facts_law = res_med['documents'][0], res_law['documents'][0]
                
                st.subheader("1. M1: Кросс-доменный поиск")
                st.markdown("**Медицина:**"); [st.info(f) for f in facts_med]
                st.markdown("**Право:**"); [st.info(f) for f in facts_law]
                
                rules_med, rules_law = load_rules("medicine"), load_rules("law")
                prompt = f"""Проанализируй вопрос с двух сторон.
Медицинские факты: {facts_med}
Юридические факты: {facts_law}
Вопрос: {question}
Верни СТРОГО JSON:
{{"medical": {{"drug": "...", "effect": "... из {list(rules_med['consequences'].keys())}", "reason": "..."}},
  "legal": {{"drug": "...", "effect": "... из {list(rules_law['consequences'].keys())}", "reason": "..."}}}}"""
                
                response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                raw = response.choices[0].message.content
                data = json.loads(raw)
                st.subheader("2. M2: Кросс-доменный JSON"); st.code(raw, language="json")
                
                st.subheader("3. M3: Вердикты Z3")
                med, law = data.get("medical", {}), data.get("legal", {})
                res_m, msg_m, edges_m = verify_logic(med.get("drug", ""), med.get("effect", ""), "medicine")
                if res_m is True: st.success(f"🏥 {msg_m}")
                else: st.error(f"🏥 {msg_m}")
                res_l, msg_l, edges_l = verify_logic(law.get("drug", ""), law.get("effect", ""), "law")
                if res_l is True: st.success(f"⚖️ {msg_l}")
                else: st.error(f"⚖️ {msg_l}")
                edges = edges_m + edges_l; result = res_m and res_l
                
            else:
                collection = collection_law if domain == "law" else collection_med
                rules = load_rules(domain)
                results = collection.query(query_texts=[question], n_results=2)
                facts, sources = results['documents'][0], results['metadatas'][0]
                st.subheader(f"1. M1: Поиск [{domain}]")
                for i in range(len(facts)): st.info(f"**[{sources[i]['source']}]** {facts[i]}")
                
                context = "\n".join(facts)
                allowed_effects = list(rules["consequences"].keys())
                prompt = f"""Ответь на вопрос: {question}\nФакты: {context}\nВерни СТРОГО JSON: {{"drug": "...", "effect": "... из {allowed_effects}", "can_take": true/false, "reason": "..."}}"""
                response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                raw = response.choices[0].message.content; data = json.loads(raw)
                st.subheader("2. M2: LLM"); st.code(raw, language="json")
                
                st.subheader(f"3. M3: Z3 [{domain}]")
                result, msg, edges = verify_logic(data.get("drug", ""), data.get("effect", ""), domain)
                if result is True: st.success(f"✅ {msg}")
                elif result is False: st.error(f"❌ {msg}")
                else: st.warning(f"⚠️ {msg}")
                
            log_event("m3_verified", {"valid": result, "domain": domain})

        with col2:
            st.subheader("4. Граф Связей")
            if edges:
                net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", directed=True)
                for edge in edges:
                    net.add_node(edge[0], label=edge[0], color="#00FFAA")
                    net.add_node(edge[1], label=edge[1], color="#FF5555" if ("Опасно" in edge[1] or "Риск" in edge[1] or "Недействителен" in edge[1]) else "#5555FF")
                    net.add_edge(edge[0], edge[1])
                net.save_graph("graph.html")
                with open("graph.html", "r", encoding="utf-8") as f: components.html(f.read(), height=520)
            else:
                st.info("Граф не построен.")
