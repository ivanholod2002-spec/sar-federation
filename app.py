__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb, json, uuid
from openai import OpenAI
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import io

st.set_page_config(page_title="DZIMO", page_icon="🧠", layout="centered")

# --- CSS МАГИЯ: ТОЧНАЯ КОПИЯ ИНТЕРФЕЙСА DEEPSEEK ---
st.markdown("""
<style>
    /* Убираем стандартные отступы Streamlit */
    .block-container { padding-top: 2rem; padding-bottom: 8rem; }
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Стилизация контейнера формы как единого поля ввода DeepSeek */
    div[data-testid="stForm"] {
        border: 1px solid #e5e7eb !important;
        border-radius: 16px !important;
        padding: 10px 15px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        position: fixed;
        bottom: 20px;
        left: 20%;
        right: 20%;
        z-index: 1000;
    }
    
    /* Убираем рамки у текстового поля внутри формы */
    div[data-testid="stForm"] textarea {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
        font-size: 16px !important;
        padding: 0 !important;
        resize: none !important;
    }
    
    /* Стилизация кнопок-таблеток */
    div[data-testid="stForm"] button {
        border-radius: 20px !important;
        font-size: 13px !important;
        padding: 4px 12px !important;
        background-color: #ffffff !important;
        color: #4b5563 !important;
        border: 1px solid #e5e7eb !important;
        transition: 0.2s;
    }
    div[data-testid="stForm"] button:hover {
        background-color: #f3f4f6 !important;
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

# Получаем список наук (без префикса domain_)
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
if "mode" not in st.session_state:
    st.session_state.mode = "deep" # По умолчанию - Глубокое погружение

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg.get("reasoning", ""))
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

# --- ФОРМА (ЕДИНОЕ ПОЛЕ ВВОДА КАК У DEEPSEEK) ---
# Мы используем форму, чтобы поле и кнопки были одним целым
with st.form(key="dzimo_input_form", clear_on_submit=True):
    # Верхняя часть: поле ввода текста
    prompt = st.text_area("Сообщение для DZIMO...", height=68, label_visibility="collapsed")
    
    # Нижняя часть: кнопки режимов и отправка
    col1, col2, col3, col4, col5 = st.columns([2.5, 2.5, 2, 0.5, 0.5])
    
    with col1:
        deep_btn = st.form_submit_button("🌌 Глубокое погружение")
    with col2:
        science_btn = st.form_submit_button("🌐 Умный поиск (Наука)")
    with col3:
        # Выбор науки появляется прямо в этой же панели (если нажата кнопка Науки)
        selected_domain = None
        if st.session_state.mode == "science" and existing_cols:
            selected_domain = st.selectbox("Наука", existing_cols, label_visibility="collapsed")
    with col4:
        st.markdown("<div style='text-align: right; padding-top: 5px;'><svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><path d='M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48'></path></svg></div>", unsafe_allow_html=True)
    with col5:
        submit_btn = st.form_submit_button("↑")

# --- ЛОГИКА ОБРАБОТКИ ---
if deep_btn:
    st.session_state.mode = "deep"
    st.rerun()
if science_btn:
    st.session_state.mode = "science"
    st.rerun()

# Если нажали Enter в поле или кнопку Отправить
if submit_btn and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую..."):
            history_context = ""
            if len(st.session_state.messages) > 1:
                for m in st.session_state.messages[-3:-1]:
                    history_context += f"{m['role'].upper()}: {m.get('content') or m.get('reasoning')}\n"

            if st.session_state.mode == "deep":
                context_parts = []
                for col_name in chroma_client.list_collections():
                    col = chroma_client.get_collection(name=col_name.name)
                    try:
                        res = col.query(query_texts=[prompt], n_results=1)
                        if res['documents'][0]: context_parts.append(f"[{res['metadatas'][0][0].get('domain')}]: {res['documents'][0][0]}")
                    except: pass
                context = "\n\n".join(context_parts)
                llm_prompt = f"Ты DZIMO. Режим: Стык наук.\nИСТОРИЯ: {history_context}\nКОНТЕКСТ: {context}\nВОПРОС: {prompt}\nНайди изоморфизмы. Верни JSON: {{\"reasoning\": \"текст\"}}"
            else:
                if not selected_domain:
                    st.error("Выберите науку в выпадающем списке!")
                    st.stop()
                col_name = [c.name for c in chroma_client.list_collections() if c.name.replace("domain_", "").capitalize() == selected_domain][0]
                facts = chroma_client.get_collection(name=col_name).query(query_texts=[prompt], n_results=3)['documents'][0]
                context = "\n".join(facts)
                llm_prompt = f"Ты DZIMO. Режим: Наука ({selected_domain}).\nИСТОРИЯ: {history_context}\nКОНТЕКСТ: {context}\nВОПРОС: {prompt}\nЕсли сменили тему, начни: 'Вижу, мы перешли на другую тему...'. Верни JSON: {{\"reasoning\": \"текст\"}}"

            resp = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": llm_prompt}], response_format={"type": "json_object"})
            reasoning_text = json.loads(resp.choices[0].message.content).get("reasoning", "Нет ответа.")

            st.markdown(reasoning_text)
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
