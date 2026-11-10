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

st.set_page_config(page_title="САР-ВСЕЛЕННАЯ", page_icon="🌌", layout="wide")

load_dotenv()
api_key = st.secrets.get("API_KEY", os.getenv("API_KEY"))
api_base = st.secrets.get("API_BASE", os.getenv("API_BASE"))
client = OpenAI(api_key=api_key, base_url=api_base)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

st.title("🌌 САР-ВСЕЛЕННАЯ: Синтез всех наук")
existing_cols = [c.name for c in chroma_client.list_collections()]

with st.sidebar:
    st.header("Управление Вселенной")
    mode = st.radio("Режим:", ["🔬 Одна наука", "🌌 СТЫК ВСЕХ НАУК"])
    
    if mode == "🔬 Одна наука":
        selected_domain = st.selectbox("Выберите науку:", existing_cols) if existing_cols else None
    else:
        selected_domain = "universe"
        st.info(f"В базе {len(existing_cols)} наук.")

    st.divider()
    with st.expander("📁 Загрузить новую науку"):
        new_domain_name = st.text_input("Название науки (напр: Музыка):")
        uploaded_file = st.file_uploader("TXT или PDF", type=["txt", "pdf"])
        
        if uploaded_file and new_domain_name and st.button("Проглотить (RAG)"):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            uploaded_file.seek(0)
            if file_extension == "pdf":
                text = "".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(uploaded_file.read())).pages])
            else:
                text = uploaded_file.read().decode("utf-8")
            
            chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
            safe_name = "".join([c if c.isalnum() else "_" for c in new_domain_name]).lower()
            target_col = chroma_client.get_or_create_collection(name=f"domain_{safe_name}")
            with st.spinner("Векторизация..."):
                target_col.add(documents=chunks, metadatas=[{"source": uploaded_file.name, "domain": new_domain_name}]*len(chunks), ids=[str(uuid.uuid4()) for _ in chunks])
            st.success(f"Наука '{new_domain_name}' загружена!")
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображаем историю
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg.get("reasoning", ""))
            if "graph" in msg and msg["graph"]:
                with st.expander("🕸️ Посмотреть паутину связей"):
                    components.html(msg["graph"], height=400)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Задайте вопрос..."):
    if not selected_domain:
        st.error("Сначала загрузите документы.")
    else:
        # 1. Сохраняем и показываем вопрос пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Око сканирует пространство..."):
                
                # ВАЖНО: Берем последние 3 сообщения для контекста
                history_context = ""
                if len(st.session_state.messages) > 1:
                    recent_msgs = st.session_state.messages[-3:-1] # Берем предпоследние сообщения
                    for m in recent_msgs:
                        content = m.get("content", "") if m["role"] == "user" else m.get("reasoning", "")
                        history_context += f"{m['role'].upper()}: {content}\n"

                # СБОР ДАННЫХ
                if selected_domain == "universe":
                    context_parts = []
                    for col_name in existing_cols:
                        col = chroma_client.get_collection(name=col_name)
                        try:
                            res = col.query(query_texts=[prompt], n_results=1)
                            if res['documents'][0]:
                                facts = res['documents'][0][0]
                                domain_label = res['metadatas'][0][0].get('domain', col_name)
                                context_parts.append(f"[{domain_label}]: {facts}")
                        except: pass
                    context = "\n\n".join(context_parts)
                    
                    # ОБНОВЛЕННЫЙ ПРОМПТ: Добавляем историю диалога
                    llm_prompt = f"""Ты — Универсальный Синтезатор Знаний.
Твоя задача — найти глубокие связи между науками и ответить на вопрос, учитывая ИСТОРИЮ ДИАЛОГА.
ИСТОРИЯ ДИАЛОГА:
{history_context}
Контекст из баз данных:
{context}
Текущий вопрос: {prompt}

Напиши развернутый текст (поле "reasoning"), где объяснишь связи. Если вопрос короткий ("что дальше"), опирайся на историю диалога!
Верни СТРОГО JSON:
{{
  "reasoning": "Твой ответ...",
  "verification": {{ "concept_a": "Название 1", "concept_b": "Название 2", "relation": "связь" }}
}}"""
                else:
                    col = chroma_client.get_collection(name=selected_domain)
                    facts = col.query(query_texts=[prompt], n_results=4)['documents'][0]
                    context = "\n".join(facts)
                    
                    llm_prompt = f"""Ответь на вопрос, опираясь на контекст и историю.
ИСТОРИЯ: {history_context}
Контекст: {context}
Вопрос: {prompt}
Верни СТРОГО JSON:
{{"reasoning": "Твой ответ", "verification": {{"concept_a": "none", "concept_b": "none", "relation": "none"}}}}"""

                # ВЫЗОВ LLM
                resp = client.chat.completions.create(
                    model="deepseek-chat", 
                    messages=[{"role": "user", "content": llm_prompt}], 
                    response_format={"type": "json_object"}
                )
                data = json.loads(resp.choices[0].message.content)
                
                reasoning_text = data.get("reasoning", "Нет рассуждений.")
                st.markdown(reasoning_text)
                
                # ВИЗУАЛИЗАЦИЯ
                graph_html = ""
                ver = data.get("verification", {})
                c1, c2, rel = ver.get("concept_a", ""), ver.get("concept_b", ""), ver.get("relation", "")
                
                if c1 and c2 and c1 != "none":
                    net = Network(height="400px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)
                    net.add_node(c1, label=c1, color="#00FFAA", size=30)
                    net.add_node(c2, label=c2, color="#FF55FF", size=30)
                    net.add_edge(c1, c2, label=rel, color="#FFFFFF", font={"color": "white"})
                    net.save_graph("graph.html")
                    with open("graph.html", "r", encoding="utf-8") as f:
                        graph_html = f.read()
                    with st.expander("🕸️ Посмотреть стык наук"):
                        components.html(graph_html, height=400)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "reasoning": reasoning_text,
                    "graph": graph_html
                })
