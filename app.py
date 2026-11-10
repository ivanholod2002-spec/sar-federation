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

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="САР-ФЕДЕРАЦИЯ | Neuro-Symbolic AI", page_icon="🧠", layout="wide")

# Кастомный CSS для красоты
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #1E2129; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .stChatMessage { border-radius: 15px; padding: 10px; }
    .verdict-yes { color: #00FFAA; font-weight: bold; font-size: 20px; }
    .verdict-no { color: #FF5555; font-weight: bold; font-size: 20px; }
</style>
""", unsafe_allow_html=True)

load_dotenv()
api_key = st.secrets.get("API_KEY", os.getenv("API_KEY"))
api_base = st.secrets.get("API_BASE", os.getenv("API_BASE"))
client = OpenAI(api_key=api_key, base_url=api_base)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

st.title("🧠 САР-ФЕДЕРАЦИЯ: Neuro-Symbolic Engine")
st.caption("Аудируемый ИИ с математической верификацией Z3")

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.header("Панель управления")
    
    domain = st.radio(
        "Домен (Federation Layer):",
        options=["medicine", "law", "federation"],
        format_func=lambda x: "🏥 Медицина" if x == "medicine" else ("⚖️ Право" if x == "law" else "🔗 Кросс-домен"),
        horizontal=False
    )
    
    st.divider()
    st.subheader("📊 Метрики доверия")
    m = compute_metrics()
    col_a, col_b = st.columns(2)
    col_a.metric("Запросов", m["total_queries"])
    col_b.metric("Правил", m["rules_added"])
    col_a.metric("✅ Доказано", m["verified_true"])
    col_b.metric("❌ Заблок.", m["verified_false"])
    st.progress(min(m["verification_rate"] / 100, 1.0), text=f"Верификация: {m['verification_rate']}%")

    st.divider()
    with st.expander("📁 Загрузить документ (M1/M5)"):
        uploaded_file = st.file_uploader("TXT или PDF", type=["txt", "pdf"])
        if uploaded_file:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            def read_file_text():
                uploaded_file.seek(0)
                if file_extension == "pdf":
                    reader = PdfReader(io.BytesIO(uploaded_file.read()))
                    return "".join([page.extract_text() or "" for page in reader.pages])
                return uploaded_file.read().decode("utf-8")
            
            if st.button("Проглотить (RAG)"):
                with st.spinner("Векторизация..."):
                    text = read_file_text()
                    chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
                    target_cols = [chroma_client.get_or_create_collection("facts"), chroma_client.get_or_create_collection("facts_law")] if domain == "federation" else ([chroma_client.get_or_create_collection("facts_law")] if domain == "law" else [chroma_client.get_or_create_collection("facts")])
                    for col in target_cols:
                        col.add(documents=chunks, metadatas=[{"source": uploaded_file.name}]*len(chunks), ids=[str(uuid.uuid4()) for _ in chunks])
                    st.success(f"Загружено {len(chunks)} чанков!")
                    
            if st.button("Извлечь правила (LLM)"):
                with st.spinner("Анализ текста..."):
                    text = read_file_text()
                    if domain == "federation":
                        st.session_state['extracted_med'] = extract_rules(text[:3000], "medicine", client)
                        st.session_state['extracted_law'] = extract_rules(text[:3000], "law", client)
                        st.success("Правила извлечены (Медицина + Право)!")
                    else:
                        st.session_state['extracted_rules'] = extract_rules(text[:3000], domain, client)
                        st.success("Правила извлечены!")

    with st.expander("🔐 Журнал аудита (L1)"):
        if st.button("Проверить целостность"):
            ok, msg = verify_chain()
            if ok: st.success(msg)
            else: st.error(msg)

# --- ОСНОВНОЙ ЧАТ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображаем историю
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "graph" in message:
            with st.expander("Посмотреть граф связей"):
                components.html(message["graph"], height=400)

# Поле ввода (как в ChatGPT)
if prompt := st.chat_input("Задайте медицинский или юридический вопрос..."):
    
    # 1. Добавляем вопрос пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Логика САР
    with st.chat_message("assistant"):
        with st.spinner("Око анализирует..."):
            log_event("query_received", {"question": prompt, "domain": domain})
            
            # M1: Поиск
            collection_med = chroma_client.get_or_create_collection("facts")
            collection_law = chroma_client.get_or_create_collection("facts_law")
            
            facts_med = collection_med.query(query_texts=[prompt], n_results=2)['documents'][0]
            facts_law = collection_law.query(query_texts=[prompt], n_results=2)['documents'][0] if domain == "federation" else []
            
            context = "\n".join(facts_med + facts_law)
            rules = load_rules(domain)
            allowed_effects = list(rules.get("consequences", {}).keys())
            
            # M2: LLM (с учетом истории)
            history_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
            llm_prompt = f"""Ты — нейро-символьный ИИ. 
Контекст из базы: {context}
История диалога: {history_prompt}
Вопрос: {prompt}
Верни СТРОГО JSON: 
{{"drug": "название", "effect": "из списка {allowed_effects}", "age": число_или_0, "disease": "болезнь_или_none", "reason": "объяснение"}}"""
            
            response = client.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": llm_prompt}], 
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            
            # M3: Z3 (с возрастом и болезнью)
            result, msg, edges = verify_logic(
                drug=data.get("drug", ""), 
                claimed_effect=data.get("effect", ""), 
                domain=domain,
                age=int(data.get("age", 30)),
                disease=data.get("disease", "none")
            )
            
            # Формируем красивый ответ
            if result is True:
                verdict = f"<span class='verdict-yes'>✅ РАЗРЕШЕНО / ДОКАЗАНО</span><br>{msg}<br><br><i>{data.get('reason', '')}</i>"
            elif result is False:
                verdict = f"<span class='verdict-no'>❌ ЗАБЛОКИРОВАНО (Противоречие)</span><br>{msg}"
            else:
                verdict = f"<span style='color: orange;'>⚠️ ТРЕБУЕТСЯ ЭКСПЕРТ</span><br>{msg}"

            st.markdown(verdict, unsafe_allow_html=True)
            
            # Строим граф
            graph_html = ""
            if edges:
                net = Network(height="400px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)
                for edge in edges:
                    net.add_node(edge[0], label=edge[0], color="#00FFAA")
                    net.add_node(edge[1], label=edge[1], color="#FF5555" if "Риск" in edge[1] or "Опасн" in edge[1] else "#5555FF")
                    net.add_edge(edge[0], edge[1])
                net.save_graph("graph.html")
                with open("graph.html", "r", encoding="utf-8") as f:
                    graph_html = f.read()
                components.html(graph_html, height=400)
            
            # Сохраняем ответ в историю
            st.session_state.messages.append({
                "role": "assistant", 
                "content": verdict,
                "graph": graph_html
            })
            log_event("m3_verified", {"valid": result, "domain": domain})
