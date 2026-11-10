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
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Задайте вопрос..."):
    if not selected_domain:
        st.error("Сначала загрузите документы.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Око сканирует пространство..."):
                
                # Берем последние 3 сообщения для контекста
                history_context = ""
                if len(st.session_state.messages) > 1:
                    recent_msgs = st.session_state.messages[-3:-1]
                    for m in recent_msgs:
                        content = m.get("content", "") if m["role"] == "user" else m.get("reasoning", "")
                        role_name = "Человек" if m["role"] == "user" else "Ты"
                        history_context += f"{role_name}: {content}\n"

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
                    
                    # ЖЕСТКИЙ ПРОМПТ: Запрещаем писать "Пользователь"
                    llm_prompt = f"""Ты — Универсальный Синтезатор Знаний, общающийся напрямую с Человеком.
Твоя задача — дать глубокий, научный ответ, учитывая историю диалога.

ИСТОРИЯ ДИАЛОГА:
{history_context}

КОНТЕКСТ ИЗ БАЗ:
{context}

ТЕКУЩИЙ ВОПРОС: {prompt}

ПРАВИЛА ОТВЕТА:
1. НИКОГДА не пиши в третьем лице (например, "Пользователь спрашивает..."). Обращайся напрямую ("Вы спрашиваете...", "Как вы заметили...").
2. Не выводи свои размышления о том, как ты строишь ответ. Сразу пиши готовый научный текст.
3. Если нужно продолжить предыдущую мысль (например, доказательство Гипотезы Римана), делай это, опираясь на историю.

Верни СТРОГО JSON:
{{
  "reasoning": "Твой прямой и развернутый ответ (научный текст)."
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
{{"reasoning": "Твой прямой ответ."}}"""

                # ВЫЗОВ LLM
                resp = client.chat.completions.create(
                    model="deepseek-chat", 
                    messages=[{"role": "user", "content": llm_prompt}], 
                    response_format={"type": "json_object"}
                )
                data = json.loads(resp.choices[0].message.content)
                
                reasoning_text = data.get("reasoning", "Нет рассуждений.")
                st.markdown(reasoning_text)
                
                # Сохраняем в историю (БЕЗ ГРАФОВ И БЕЗ КНОПОК)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "reasoning": reasoning_text
                })
