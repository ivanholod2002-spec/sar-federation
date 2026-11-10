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

st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
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

with st.sidebar:
    st.header("Панель управления")
    domain = st.radio("Домен:", ["medicine", "law", "federation"], format_func=lambda x: "🏥 Медицина" if x == "medicine" else ("⚖️ Право" if x == "law" else "🔗 Кросс-домен"))
    
    st.divider()
    m = compute_metrics()
    st.metric("Верификация", f"{m['verification_rate']}%")
    st.metric("Правил в базе", m["rules_added"])

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
            for col in target_cols:
                col.add(documents=chunks, metadatas=[{"source": uploaded_file.name}]*len(chunks), ids=[str(uuid.uuid4()) for _ in chunks])
            st.success(f"Загружено {len(chunks)} чанков!")

# --- ЧАТ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if "graph" in msg and msg["graph"]:
            components.html(msg["graph"], height=400)
        
        # Если система не знает ответа, показываем форму обучения
        if msg.get("needs_training"):
            with st.form(key=f"train_{msg['id']}"):
                st.write("👤 **Эксперт, обучите систему:**")
                new_drug = st.text_input("Сущность (Препарат/Договор):")
                new_effect = st.text_input("Свойство:")
                new_consequence = st.text_input("Следствие:")
                if st.form_submit_button("✅ Добавить правило"):
                    if new_drug and new_effect and new_consequence:
                        rules_update = load_rules(domain)
                        rules_update["drugs"].setdefault(new_drug, []).append(new_effect)
                        rules_update["consequences"][new_effect] = new_consequence
                        filename = "rules_law.json" if domain == "law" else "rules_db.json"
                        with open(filename, "w", encoding="utf-8") as f:
                            json.dump(rules_update, f, ensure_ascii=False, indent=2)
                        log_event("rule_added", {"drug": new_drug, "effect": new_effect, "consequence": new_consequence})
                        st.success("Правило добавлено! Задайте вопрос снова.")
                        # Удаляем старые сообщения для чистоты
                        st.session_state.messages = []
                        st.rerun()

if prompt := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Око анализирует..."):
            collection_med = chroma_client.get_or_create_collection("facts")
            collection_law = chroma_client.get_or_create_collection("facts_law")
            facts_med = collection_med.query(query_texts=[prompt], n_results=1)['documents'][0]
            facts_law = collection_law.query(query_texts=[prompt], n_results=1)['documents'][0] if domain == "federation" else []
            
            context = "\n".join(facts_med + facts_law)
            rules = load_rules(domain)
            allowed = list(rules.get("consequences", {}).keys())
            
            resp = client.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": f"Контекст: {context}\nВопрос: {prompt}\nJSON: {{\"drug\": \"...\", \"effect\": \"из {allowed}\", \"age\": 0, \"disease\": \"none\", \"reason\": \"...\"}}"}], 
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            
            result, msg_text, edges = verify_logic(data.get("drug", ""), data.get("effect", ""), domain, int(data.get("age", 30)), data.get("disease", "none"))
            
            if result is True:
                verdict = f"<span class='verdict-yes'>✅ РАЗРЕШЕНО / ДОКАЗАНО</span><br>{msg_text}"
                needs_training = False
            elif result is False:
                verdict = f"<span class='verdict-no'>❌ ЗАБЛОКИРОВАНО</span><br>{msg_text}"
                needs_training = False
            else:
                verdict = f"<span style='color: orange;'>⚠️ ТРЕБУЕТСЯ ЭКСПЕРТ</span><br>{msg_text}"
                needs_training = True # Включаем форму обучения!

            st.markdown(verdict, unsafe_allow_html=True)
            
            graph_html = ""
            if edges:
                net = Network(height="400px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)
                for edge in edges:
                    net.add_node(edge[0], label=edge[0], color="#00FFAA")
                    net.add_node(edge[1], label=edge[1], color="#FF5555" if "Риск" in edge[1] else "#5555FF")
                    net.add_edge(edge[0], edge[1])
                net.save_graph("graph.html")
                with open("graph.html", "r", encoding="utf-8") as f:
                    graph_html = f.read()
                components.html(graph_html, height=400)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": verdict,
                "graph": graph_html,
                "needs_training": needs_training,
                "id": str(uuid.uuid4())
            })
