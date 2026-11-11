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

st.set_page_config(page_title="DZIMO | Neuro-Symbolic AI", page_icon="🧠", layout="wide")

# --- КАСТОМНЫЙ CSS ДЛЯ КРАСОТЫ (Стиль DeepSeek) ---
st.markdown("""
<style>
    /* Скрываем стандартный хедер Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Стилизация кнопок-переключателей */
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        padding: 5px 15px;
        font-size: 14px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        border-color: #4A90E2;
        color: #4A90E2;
    }
    
    /* Красивые блоки ответов */
    .stChatMessage {
        background-color: transparent;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
api_key = st.secrets.get("API_KEY", os.getenv("API_KEY"))
api_base = st.secrets.get("API_BASE", os.getenv("API_BASE"))
client = OpenAI(api_key=api_key, base_url=api_base)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

st.title("🧠 DZIMO: Neuro-Symbolic Engine")

# Получаем список всех наук из базы
existing_cols = [c.name for c in chroma_client.list_collections()]

# --- БОКОВАЯ ПАНЕЛЬ (Только для админа/загрузки) ---
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

# --- ИНТЕРФЕЙС ЧАТА ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "science" # По умолчанию - конкретная наука

# Отображаем историю
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg.get("reasoning", ""))
        else:
            st.markdown(msg["content"])

# --- ПАНЕЛЬ УПРАВЛЕНИЯ НАД ВВОДОМ (Как у DeepSeek) ---
st.markdown("---")
col1, col2, col3 = st.columns([2, 2, 6])

with col1:
    if st.button("🌌 Глубокое погружение", use_container_width=True):
        st.session_state.mode = "deep"
with col2:
    if st.button("🔬 Конкретная наука", use_container_width=True):
        st.session_state.mode = "science"

# Если выбран режим науки, показываем выпадающий список
selected_domain = None
if st.session_state.mode == "science":
    if existing_cols:
        selected_domain = st.selectbox("Выберите область (профессию/науку):", existing_cols, label_visibility="collapsed")
    else:
        st.warning("Загрузите документы в боковой панели, чтобы выбрать науку.")

# --- ВВОД СООБЩЕНИЯ ---
if prompt := st.chat_input("Сообщение для DZIMO..."):
    
    # 1. Сохраняем и показываем вопрос пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Логика DZIMO
    with st.chat_message("assistant"):
        with st.spinner("Анализирую..."):
            
            # Берем последние 3 сообщения для контекста
            history_context = ""
            if len(st.session_state.messages) > 1:
                recent_msgs = st.session_state.messages[-3:-1]
                for m in recent_msgs:
                    content = m.get("content", "") if m["role"] == "user" else m.get("reasoning", "")
                    history_context += f"{m['role'].upper()}: {content}\n"

            # --- РЕЖИМ 1: ГЛУБОКОЕ ПОГРУЖЕНИЕ (Стык всех наук) ---
            if st.session_state.mode == "deep":
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
                
                llm_prompt = f"""Ты — DZIMO, Универсальный Синтезатор Знаний.
Твоя задача — найти глубокие связи между разными науками и ответить.
ИСТОРИЯ: {history_context}
КОНТЕКСТ ИЗ БАЗ: {context}
ВОПРОС: {prompt}

ПРАВИЛА:
1. Найди изоморфизмы (связи) между концепциями из разных областей.
2. Пиши научно, но доступно.
3. Верни СТРОГО JSON: {{"reasoning": "Твой развернутый ответ."}}"""

            # --- РЕЖИМ 2: КОНКРЕТНАЯ НАУКА (Строгий эксперт) ---
            else:
                if not selected_domain:
                    st.error("Выберите науку!")
                    st.stop()
                    
                col = chroma_client.get_collection(name=selected_domain)
                facts = col.query(query_texts=[prompt], n_results=3)['documents'][0]
                context = "\n".join(facts)
                
                llm_prompt = f"""Ты — DZIMO, узкоспециализированный эксперт в области {selected_domain}.
Твоя задача — решать конкретную задачу строго по теме.
ИСТОРИЯ ДИАЛОГА: {history_context}
КОНТЕКСТ ИЗ БАЗЫ: {context}
ТЕКУЩИЙ ВОПРОС: {prompt}

ПРАВИЛА:
1. Отвечай ТОЛЬКО по теме ({selected_domain}).
2. Если пользователь резко сменил тему (спросил про что-то не связанное с предыдущим вопросом), начни ответ с фразы: "Вижу, мы перешли на другую тему. Ок, теперь разбираем: ..." и отвечай на новый вопрос.
3. НЕ поддакивай пользователю. Если он ошибается — исправляй на основе фактов.
4. Верни СТРОГО JSON: {{"reasoning": "Твой ответ."}}"""

            # ВЫЗОВ LLM
            resp = client.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": llm_prompt}], 
                response_format={"type": "json_object"},
                max_tokens=2048
            )
            
            raw_content = resp.choices[0].message.content
            try:
                data = json.loads(raw_content)
                reasoning_text = data.get("reasoning", "Нет ответа.")
            except:
                reasoning_text = raw_content

            st.markdown(reasoning_text)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "reasoning": reasoning_text
            })
