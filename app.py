__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

st.set_page_config(page_title="САР-ФЕДЕРАЦИЯ", page_icon="🧠", layout="wide")

load_dotenv()
api_key = st.secrets.get("API_KEY", os.getenv("API_KEY"))
api_base = st.secrets.get("API_BASE", os.getenv("API_BASE"))
client = OpenAI(api_key=api_key, base_url=api_base)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

st.title("🧠 САР-ФЕДЕРАЦИЯ: Neuro-Symbolic Engine")

with st.sidebar:
    domain = st.radio("Домен:", ["medicine", "law", "federation"], format_func=lambda x: "🏥 Медицина" if x == "medicine" else ("⚖️ Право" if x == "law" else "🔗 Кросс-домен"))
    m = compute_metrics()
    st.metric("Верификация", f"{m['verification_rate']}%")

    with st.expander("📁 Загрузить документ"):
        uploaded_file = st.file_uploader("TXT или PDF", type=["txt", "pdf"])
        if uploaded_file and st.button("Проглотить (RAG)"):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            uploaded_file.seek(0)
            if file_extension == "pdf":
                text = "".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(uploaded_file.read())).pages])
            else:
                text = uploaded_file.read().decode("utf-8")
            
            chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
            target_cols = [chroma_client.get_or_create_collection("facts"), chroma_client.get_or_create_collection("facts_law")] if domain == "federation" else ([chroma_client.get_or_create_collection("facts_law")] if domain == "law" else [chroma_client.get_or_create_collection("facts")])
            with st.spinner("Векторизация..."):
                for col in target_cols:
                    col.add(documents=chunks, metadatas=[{"source": uploaded_file.name}]*len(chunks), ids=[str(uuid.uuid4()) for _ in chunks])
            st.success(f"Загружено {len(chunks)} чанков!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("status") == "success": st.success(msg["content"])
            elif msg.get("status") == "error": st.error(msg["content"])
            else: st.warning(msg["content"])
            
            if "reason" in msg and msg["reason"]:
                st.info(f"💡 **Обоснование:** {msg['reason']}")
            
            if "graph" in msg and msg["graph"]:
                with st.expander("🕸️ Посмотреть паутину связей"):
                    components.html(msg["graph"], height=400)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Око сканирует паутину..."):
            collection_med = chroma_client.get_or_create_collection("facts")
            collection_law = chroma_client.get_or_create_collection("facts_law")
            
            # ВАРИАНТ 1: Гибридный поиск (4 результата вместо 1)
            res_med = collection_med.query(query_texts=[prompt], n_results=4)
            res_law = collection_law.query(query_texts=[prompt], n_results=4) if domain == "federation" else {"documents": [[]], "metadatas": [[]]}
            
            facts_med, meta_med = res_med['documents'][0], res_med['metadatas'][0]
            facts_law, meta_law = res_law['documents'][0], res_law['metadatas'][0]
            
            context = "\n".join(facts_med + facts_law)
            sources = list(set([m['source'] for m in meta_med + meta_law if m])) # Уникальные источники
            
            rules = load_rules(domain)
            allowed = list(rules.get("consequences", {}).keys())
            
            if domain == "law":
                entity_name, property_name = "действие", "последствие"
            else:
                entity_name, property_name = "препарат", "эффект"
            
            # ВАРИАНТ 1: Умный промпт для обобщения
            llm_prompt = f"""Ты — аналитик. Изучи фрагменты текста и ответь на вопрос.
Если вопрос общий (например, "что такое..."), сделай вывод на основе того, что написано в этих фрагментах.
Если информации нет, напиши "НЕТ_ДАННЫХ".
Фрагменты: {context}
Вопрос: {prompt}
Верни СТРОГО JSON: 
{{"{entity_name}": "название/термин или НЕТ_ДАННЫХ", "{property_name}": "из списка {allowed}", "age": 0, "disease": "none", "reason": "объяснение"}}"""
            
            resp = client.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": llm_prompt}], 
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            
            drug = data.get(entity_name, data.get("drug", "none"))
            effect = data.get(property_name, data.get("effect", "none"))
            
            if drug == "НЕТ_ДАННЫХ":
                final_status = "warning"
                final_text = f"⚠️ В извлеченных фрагментах нет ответа."
                st.warning(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text, "status": final_status})
            else:
                result, msg_text, edges = verify_logic(drug, effect, domain, int(data.get("age", 30)), data.get("disease", "none"))
                
                if result is True:
                    final_status = "success"
                    final_text = f"✅ РАЗРЕШЕНО / ДОКАЗАНО: {msg_text}"
                    st.success(final_text)
                elif result is False:
                    final_status = "error"
                    final_text = f"❌ ЗАБЛОКИРОВАНО: {msg_text}"
                    st.error(final_text)
                else:
                    final_status = "warning"
                    final_text = f"⚠️ ТРЕБУЕТСЯ ЭКСПЕРТ: {msg_text}"
                    st.warning(final_text)
                
                if data.get("reason"):
                    st.info(f"💡 **Обоснование:** {data['reason']}")
                
                # ВАРИАНТ 2: Рисуем ПАУТИНУ (Источники + Логика)
                graph_html = ""
                if edges or sources:
                    net = Network(height="400px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)
                    
                    # Добавляем логические связи Z3
                    for edge in edges:
                        net.add_node(edge[0], label=edge[0], color="#00FFAA", size=25)
                        net.add_node(edge[1], label=edge[1], color="#FF5555" if "Риск" in edge[1] else "#5555FF", size=20)
                        net.add_edge(edge[0], edge[1], color="#FFFFFF")
                    
                    # Добавляем связи с документами (Паутина источников)
                    for src in sources:
                        net.add_node(src, label=f"📄 {src}", color="#FFAA00", shape="box")
                        if drug != "none":
                            net.add_edge(drug, src, dashes=True, color="#888888", title="Найдено в")
                    
                    net.save_graph("graph.html")
                    with open("graph.html", "r", encoding="utf-8") as f:
                        graph_html = f.read()
                    with st.expander("🕸️ Посмотреть паутину связей"):
                        components.html(graph_html, height=400)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_text, 
                    "status": final_status,
                    "reason": data.get("reason", ""),
                    "graph": graph_html
                })
