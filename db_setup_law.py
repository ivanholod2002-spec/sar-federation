import chromadb
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="facts_law")

def add_fact(id, text, source):
    collection.add(documents=[text], metadatas=[{"source": source}], ids=[id])
    print(f"Добавлен факт: {text}")

add_fact("law_1", "Договор аренды недвижимости должен быть заключен в письменной форме. Устная форма недопустима.", "ГК РФ ст. 651")
add_fact("law_2", "Договор займа на сумму более 10 000 рублей требует нотариального заверения.", "ГК РФ ст. 808")
add_fact("law_3", "Трудовой договор должен быть подписан обеими сторонами: работником и работодателем.", "ТК РФ ст. 67")
