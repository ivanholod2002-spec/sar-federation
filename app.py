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

# --- КАСТОМНЫЙ CSS (Стиль DeepSeek) ---
st.markdown("""
<style>
    /* Делаем фон светлым и чистым */
    .stApp { background-color: #ffffff; }
    
    /* Прячем стандартный хедер */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Стилизуем контейнер с кнопками, чтобы он прилипал к низу */
    .bottom-controls {
        position: fixed;
        bottom: 85px; /* Высота строки ввода */
        left: 20%;
        right: 20%;
        background: white;
        z-index: 999;
    }
    
    /* Стилизация кнопок-таблеток (Pills) */
    div[data-testid="stPills"] > div {
        display: flex;
        gap: 10px;
    }
    div[data-testid="stPills"] button {
        border-radius: 20px !important;
        border: 1px solid #e0e0e0 !important;
        background-color: transparent !important;
        color: #333 !important;
        padding: 5px 15px !important;
        font-size: 14px !important;
    }
    div[data-testid="stPills"] button[aria-checked="true"] {
        background-color: #f0f4ff !important;
        border-color: #4A90E2 !important;
        color: #4A90E2 !important;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
api_key = st.secrets.get("API_KEY", os.getenv("API_KEY"))
api_base = st.secrets.get("API_BASE", os.getenv("API_BASE"))
client = OpenAI(api_key=api_key, base_url=api_base)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

st.title("🧠 DZIMO")
st.caption("Neuro-Symbolic Engine для решения сложных задач")

# Получаем список всех наук из базы
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

# --- ЧАТ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg.get("reasoning", ""))
        else:
            st.markdown(msg["content"])

# --- НИЖНЯЯ ПАНЕЛЬ УПРАВЛЕНИЯ (КАК У DEEPSEEK) ---
# Создаем контейнер внизу экрана
st.markdown('<div class="bottom-controls">', unsafe_allow_html=True)

# Кнопки-переключатели режимов (Pills)
mode = st.pills(
    "Режим работы", 
    ["🌌 Глубокое погружение (Стык наук)", "🔬 Конкретная наука"], 
    default="🔬 Конкретная наука",
    label_visibility="collapsed"
)

# Если выбрана конкретная наука, показываем выпадающий список прямо под кнопками
selected_domain = None
if mode == "🔬 Конкретная наука" and existing_cols:
    selected_domain = st.selectbox(
        "Выберите область знаний:", 
        existing_cols, 
        label_visibility="collapsed"
    )

st.markdown('</div>', unsafe_allow_html=True)

# --- ВВОД СООБЩЕНИЯ ---
if prompt := st.chat_input("Сообщение для DZIMO..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("DZIMO анализирует..."):
            
            history_context = ""
            if len(st.session_state.messages) > 1:
                recent_msgs = st.session_state.messages[-3:-1]
                for m in recent_msgs:
                    content = m.get("content", "") if m["role"] == "user" else m.get("reasoning", "")
                    history_context += f"{m['role'].upper()}: {content}\n"

            # --- ЛОГИКА РЕЖИМОВ ---
            if mode == "🌌 Глубокое погружение (Стык наук)":
                context_parts = []
                for col_name in chroma_client.list_collections():
                    col = chroma_client.get_collection(name=col_name.name)
                    try:
                        res = col.query(query_texts=[prompt], n_results=1)
                        if res['documents'][0]:
                            facts = res['documents'][0][0]
                            domain_label = res['metadatas'][0][0].get('domain', col_name.name)
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

            elif mode == "🔬 Конкретная наука":
                if not selected_domain:
                    st.error("Выберите науку в выпадающем списке!")
                    st.stop()
                    
                # Находим правильное имя коллекции по красивому названию
                col_name = [c.name for c in chroma_client.list_collections() if c.name.replace("domain_", "").capitalize() == selected_domain][0]
                col = chroma_client.get_collection(name=col_name)
                
                facts = col.query(query_texts=[prompt], n_results=3)['documents'][0]
                context = "\n".join(facts)
                
                llm_prompt = f"""Ты — DZIMO, узкоспециализированный эксперт в области {selected_domain}.
Твоя задача — решать конкретную задачу строго по теме.
ИСТОРИЯ ДИАЛОГА: {history_context}
КОНТЕКСТ ИЗ БАЗЫ: {context}
ТЕКУЩИЙ ВОПРОС: {prompt}

ПРАВИЛА:
1. Отвечай ТОЛЬКО по теме ({selected_domain}).
2. Если пользователь резко сменил тему, начни ответ с фразы: "Вижу, мы перешли на другую тему. Ок, теперь разбираем: ..." и отвечай на новый вопрос.
3. НЕ поддакивай пользователю. Если он ошибается — исправляй на основе фактов.
4. Верни СТРОГО JSON: {{"reasoning": "Твой ответ."}}"""
            else:
                st.warning("Выберите режим работы.")
                st.stop()

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
