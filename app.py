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

st.set_page_config(page_title="DZIMO", page_icon="🧠", layout="wide")

# --- CSS МАГИЯ: ТОЧНАЯ КОПИЯ DEEPSEEK ---
st.markdown("""
<style>
    /* Скрываем стандартный хедер и футер */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Делаем фон белым */
    .stApp { background-color: #ffffff; }
    
    /* Стилизация контейнера с кнопками (прилипает к полю ввода) */
    div[data-testid="stHorizontalBlock"] {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 15px 15px 0 0;
        border: 1px solid #e0e0e0;
        border-bottom: none;
        margin-bottom: -15px; /* Сдвигаем вниз, чтобы слилось с полем */
    }
    
    /* Стилизация самих кнопок-таблеток */
    button[data-testid="baseButton-secondary"] {
        border-radius: 20px !important;
        border: 1px solid #d0d0d0 !important;
        background-color: #ffffff !important;
        color: #333 !important;
        padding: 6px 16px !important;
        font-size: 14px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    button[data-testid="baseButton-primary"] {
        border-radius: 20px !important;
        border: 1px solid #4A90E2 !important;
        background-color: #f0f4ff !important;
        color: #4A90E2 !important;
        padding: 6px 16px !important;
        font-size: 14px !important;
    }

    /* Стилизация поля ввода, чтобы оно прилипло к кнопкам */
    div[data-testid="stChatInput"] {
        border-radius: 0 0 15px 15px !important;
        border: 1px solid #e0e0e0;
        border-top: 1px solid #f0f0f0;
        background-color: #f9f9f9;
        padding-bottom: 10px;
    }
    
    /* Красивые иконки под сообщениями */
    .action-icons {
        display: flex;
        gap: 15px;
        margin-top: 10px;
        opacity: 0.6;
        transition: opacity 0.2s;
    }
    .action-icons:hover { opacity: 1; }
    .action-icons svg {
        width: 18px;
        height: 18px;
        cursor: pointer;
        fill: none;
        stroke: #555;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .action-icons svg:hover { stroke: #000; }
</style>
""", unsafe_allow_html=True)

load_dotenv()
api_key = st.secrets.get("API_KEY", os.getenv("API_KEY"))
api_base = st.secrets.get("API_BASE", os.getenv("API_BASE"))
client = OpenAI(api_key=api_key, base_url=api_base)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Получаем список наук
existing_cols = [c.name.replace("domain_", "").capitalize() for c in chroma_client.list_collections()]

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("📁 База знаний")
    with st.expander("Загрузить новую науку"):
        new_domain_name = st.text_input("Название науки:")
        uploaded_file = st.file_uploader("TXT или PDF", type=["txt", "pdf"])
        if uploaded_file and new_domain_name and st.button("Проглотить"):
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

st.title("🧠 DZIMO")
st.caption("Neuro-Symbolic Engine")

# --- ЧАТ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg.get("reasoning", ""))
            # ИКОНКИ ПОД СООБЩЕНИЕМ (Как на скрине)
            st.markdown("""
            <div class="action-icons">
                <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M23 4v6h-6"></path><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-2"></path></svg>
                <svg viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# --- ПАНЕЛЬ УПРАВЛЕНИЯ (Точная копия) ---
# Используем колонки, чтобы кнопки были слева
col1, col2, col3 = st.columns([2, 2, 6])

with col1:
    deep_mode = st.button("🌌 Глубокое погружение", use_container_width=True)
with col2:
    science_mode = st.button("🌐 Умный поиск (Наука)", use_container_width=True)

# Логика переключения режимов
if deep_mode:
    st.session_state.mode = "deep"
if science_mode:
    st.session_state.mode = "science"

if "mode" not in st.session_state:
    st.session_state.mode = "science"

selected_domain = None
if st.session_state.mode == "science" and existing_cols:
    selected_domain = st.selectbox("Выберите науку:", existing_cols, label_visibility="collapsed")

# --- ВВОД ---
if prompt := st.chat_input("Сообщение для DZIMO..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую..."):
            history_context = ""
            if len(st.session_state.messages) > 1:
                recent_msgs = st.session_state.messages[-3:-1]
                for m in recent_msgs:
                    content = m.get("content", "") if m["role"] == "user" else m.get("reasoning", "")
                    history_context += f"{m['role'].upper()}: {content}\n"

            if st.session_state.mode == "deep":
                context_parts = []
                for col_name in chroma_client.list_collections():
                    col = chroma_client.get_collection(name=col_name.name)
                    try:
                        res = col.query(query_texts=[prompt], n_results=1)
                        if res['documents'][0]:
                            context_parts.append(f"[{res['metadatas'][0][0].get('domain', col_name.name)}]: {res['documents'][0][0]}")
                    except: pass
                context = "\n\n".join(context_parts)
                
                llm_prompt = f"""Ты — DZIMO. Режим: Глубокое погружение (Стык наук).
ИСТОРИЯ: {history_context}
КОНТЕКСТ: {context}
ВОПРОС: {prompt}
Найди изоморфизмы (связи) между разными науками. Пиши научно.
Верни СТРОГО JSON: {{"reasoning": "Твой ответ."}}"""

            else:
                if not selected_domain:
                    st.error("Выберите науку!")
                    st.stop()
                col_name = [c.name for c in chroma_client.list_collections() if c.name.replace("domain_", "").capitalize() == selected_domain][0]
                col = chroma_client.get_collection(name=col_name)
                facts = col.query(query_texts=[prompt], n_results=3)['documents'][0]
                context = "\n".join(facts)
                
                llm_prompt = f"""Ты — DZIMO. Режим: Умный поиск (Наука: {selected_domain}).
ИСТОРИЯ: {history_context}
КОНТЕКСТ: {context}
ВОПРОС: {prompt}
Отвечай строго по теме ({selected_domain}). Если пользователь сменил тему, начни с фразы: "Вижу, мы перешли на другую тему..." и отвечай на новый вопрос.
Верни СТРОГО JSON: {{"reasoning": "Твой ответ."}}"""

            resp = client.chat.completions.create(
                model="deepseek-chat", messages=[{"role": "user", "content": llm_prompt}], response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            reasoning_text = data.get("reasoning", "Нет ответа.")

            st.markdown(reasoning_text)
            # Добавляем иконки и к новому сообщению
            st.markdown("""
            <div class="action-icons">
                <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M23 4v6h-6"></path><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-2"></path></svg>
                <svg viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "reasoning": reasoning_text})
