__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb, json, uuid
from openai import OpenAI
import os
from dotenv import load_dotenv
from z3_verifier import verify_logic, load_rules
from pypdf import PdfReader
import io

st.set_page_config(page_title="DZIMO", page_icon="🧠", layout="wide")

# --- CSS: ДЕЛАЕМ КНОПКИ КРАСИВЫМИ (Как пилюли) ---
st.markdown("""
<style>
    /* Скрываем лишнее */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #ffffff; }
    
    /* Стилизация радио-кнопок под таблетки DeepSeek */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 10px;
        margin-bottom: -10px; /* Приближаем к полю ввода */
    }
    div[role="radiogroup"] label {
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 6px 16px;
        background-color: #ffffff;
        color: #4b5563;
        cursor: pointer;
        transition: 0.2s;
        font-size: 14px;
        display: flex;
        align-items: center;
    }
    div[role="radiogroup"] label:hover {
        background-color: #f3f4f6;
    }
    /* Убираем стандартный кружок радио */
    div[role="radiogroup"] label > div:first-child {
        display: none;
    }
    /* Активная кнопка */
    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #eff6ff;
        border-color: #3b82f6;
        color: #3b82f6;
    }

    /* Иконки под сообщениями */
    .msg-icons {
        display: flex;
        gap: 14px;
        margin-top: 8px;
        opacity: 0.6;
    }
    .msg-icons svg { width: 16px; height: 16px; cursor: pointer; stroke: #555; fill: none; stroke-width: 2; }
    .msg-icons svg:hover { stroke: #000; opacity: 1; }
</style>
""", unsafe_allow_html=True)

load_dotenv()
api_key = st.secrets.get("API_KEY", os.getenv("API_KEY"))
api_base = st.secrets.get("API_BASE", os.getenv("API_BASE"))
client = OpenAI(api_key=api_key, base_url=api_base)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Список наук (убираем префиксы domain_)
existing_cols = [c.name.replace("domain_", "").capitalize() for c in chroma_client.list_collections()]

# --- БОКОВАЯ ПАНЕЛЬ (Загрузка) ---
with st.sidebar:
    st.header("📁 База знаний")
    with st.expander("Загрузить новую науку"):
        new_domain_name = st.text_input("Название науки:")
        uploaded_file = st.file_uploader("TXT или PDF", type=["txt", "pdf"])
        if uploaded_file and new_domain_name and st.button("Проглотить"):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            uploaded_file.seek(0)
            text = "".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(uploaded_file.read())).pages]) if file_extension == "pdf" else uploaded_file.read().decode("utf-8")
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
            # Ряд иконок под ответом
            st.markdown("""
            <div class="msg-icons">
                <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M23 4v6h-6"></path><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-2"></path></svg>
                <svg viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# --- ПАНЕЛЬ УПРАВЛЕНИЯ (Переключатели) ---
# Используем st.radio, замаскированный под кнопки
mode = st.radio(
    "Режим работы", 
    ["🌌 Глубокое погружение", "🌐 Умный поиск (Наука)"], 
    horizontal=True, 
    label_visibility="collapsed"
)

# Если выбран режим Науки, показываем выпадающий список
selected_domain = None
if mode == "🌐 Умный поиск (Наука)":
    if existing_cols:
        selected_domain = st.selectbox("Выберите науку:", existing_cols)
    else:
        st.warning("⚠️ Загрузите документы в боковой панели, чтобы выбрать науку.")

# --- ПОЛЕ ВВОДА (Всегда внизу) ---
if prompt := st.chat_input("Сообщение для DZIMO..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую..."):
            history_context = ""
            if len(st.session_state.messages) > 1:
                for m in st.session_state.messages[-3:-1]:
                    history_context += f"{m['role'].upper()}: {m.get('content') or m.get('reasoning')}\n"

            if mode == "🌌 Глубокое погружение":
                context_parts = []
                for col_name in chroma_client.list_collections():
                    col = chroma_client.get_collection(name=col_name.name)
                    try:
                        res = col.query(query_texts=[prompt], n_results=1)
                        if res['documents'][0]: 
                            context_parts.append(f"[{res['metadatas'][0][0].get('domain')}]: {res['documents'][0][0]}")
                    except: pass
                context = "\n\n".join(context_parts)
                
                llm_prompt = f"Ты DZIMO. Режим: Стык наук.\nИСТОРИЯ: {history_context}\nКОНТЕКСТ: {context}\nВОПРОС: {prompt}\nНайди изоморфизмы (связи) между разными науками. Верни JSON: {{\"reasoning\": \"текст\"}}"
            
            else:
                if not selected_domain:
                    st.error("Выберите науку!")
                    st.stop()
                
                # Находим точное имя коллекции
                col_name = [c.name for c in chroma_client.list_collections() if c.name.replace("domain_", "").capitalize() == selected_domain][0]
                col = chroma_client.get_collection(name=col_name)
                
                facts = col.query(query_texts=[prompt], n_results=3)['documents'][0]
                context = "\n".join(facts)
                
                llm_prompt = f"Ты DZIMO. Режим: Наука ({selected_domain}).\nИСТОРИЯ: {history_context}\nКОНТЕКСТ: {context}\nВОПРОС: {prompt}\nЕсли пользователь резко сменил тему, начни ответ с 'Вижу, мы перешли на другую тему. Ок...'. Верни JSON: {{\"reasoning\": \"текст\"}}"

            resp = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": llm_prompt}], response_format={"type": "json_object"})
            reasoning_text = json.loads(resp.choices[0].message.content).get("reasoning", "Нет ответа.")

            st.markdown(reasoning_text)
            
            # Иконки под новым ответом
            st.markdown("""
            <div class="msg-icons">
                <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M23 4v6h-6"></path><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                <svg viewBox="0 0 24 24"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-2"></path></svg>
                <svg viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "reasoning": reasoning_text})
