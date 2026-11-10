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
            # Сначала выводим текст рассуждения (как в ChatGPT)
            st.markdown(msg.get("reasoning", ""))
            
            # Затем выводим бейдж верификации Z3 (если есть)
            if msg.get("status") == "success": st.success(msg["content"])
            elif msg.get("status") == "error": st.error(msg["content"])
            elif msg.get("status"): st.warning(msg["content"])
            
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
        with st.spinner("Око размышляет..."):
            collection_med = chroma_client.get_or_create_collection("facts")
            collection_law = chroma_client.get_or_create_collection("facts_law")
            
            res_med = collection_med.query(query_texts=[prompt], n_results=4)
            res_law = collection_law.query(query_texts=[prompt], n_results=4) if domain == "federation" else {"documents": [[]], "metadatas": [[]]}
            
            facts_med, meta_med = res_med['documents'][0], res_med['metadatas'][0]
            facts_law, meta_law = res_law['documents'][0], res_law['metadatas'][0]
            
            context = "\n".join(facts_med + facts_law)
            sources = list(set([m['source'] for m in meta_med + meta_law if m]))
            
            rules = load_rules(domain)
            allowed = list(rules.get("consequences", {}).keys())
            
            # НОВЫЙ ПРОМПТ: Сначала думаем, потом проверяем
            llm_prompt = f"""Ты — экспертный ИИ-ассистент.
Твоя задача: 1) Дать развернутый и умный ответ на вопрос (поле "reasoning"). 2) Выделить конкретный факт для математической проверки Z3 (поле "verification").
Если вопрос общий (например, "что такое..."), просто ответь на него, а в поле "entity" напиши "none".
Если вопрос требует проверки (например, "можно ли давать Х при Y"), выдели сущность и эффект.
Контекст из базы: {context}
Вопрос: {prompt}
Верни СТРОГО JSON:
{{
  "reasoning": "Твой подробный ответ на вопрос (2-3 абзаца). Опирайся на свои знания и контекст.",
  "verification": {{
    "entity": "название или none",
    "effect": "из списка {allowed}",
    "age": 0,
    "disease": "none",
    "reason": "почему Z3 это подтвердит"
  }}
}}"""
            
            resp = client.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": llm_prompt}], 
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            
            # Выводим рассуждение (как обычный ChatGPT)
            reasoning_text = data.get("reasoning", "Нет рассуждений.")
            st.markdown(reasoning_text)
            
            # Проверяем, нужно ли запускать Z3
            verification = data.get("verification", {})
            drug = verification.get("entity", "none")
            effect = verification.get("effect", "none")
            
            final_status = None
            final_text = ""
            graph_html = ""
            
            if drug != "none" and effect != "none":
                result, msg_text, edges = verify_logic(drug, effect, domain, int(verification.get("age", 30)), verification.get("disease", "none"))
                
                if result is True:
                    final_status = "success"
                    final_text = f"✅ **Z3 Верификация:** {msg_text}"
                    st.success(final_text)
                elif result is False:
                    final_status = "error"
                    final_text = f"❌ **Z3 Верификация:** {msg_text}"
                    st.error(final_text)
                else:
                    final_status = "warning"
                    final_text = f"⚠️ **Z3 Верификация:** {msg_text}"
                    st.warning(final_text)
                
                if verification.get("reason"):
                    st.info(f"💡 **Логическое обоснование:** {verification['reason']}")
                
                # Рисуем паутину
                if edges or sources:
                    net = Network(height="400px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)
                    for edge in edges:
                        net.add_node(edge[0], label=edge[0], color="#00FFAA", size=25)
                        net.add_node(edge[1], label=edge[1], color="#FF5555" if "Риск" in edge[1] else "#5555FF", size=20)
                        net.add_edge(edge[0], edge[1], color="#FFFFFF")
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
                "reasoning": reasoning_text,
                "content": final_text,
                "status": final_status,
                "graph": graph_html
            })
