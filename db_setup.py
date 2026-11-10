import chromadb
from dotenv import load_dotenv

load_dotenv()

# Chroma сама сделает эмбеддинги локально!
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="facts")

def add_fact(id, text, source):
    collection.add(
        documents=[text],
        metadatas=[{"source": source}],
        ids=[id]
    )
    print(f"Добавлен факт: {text}")

add_fact("1", "Аспирин (Aspirin) является антиагрегантом и разжижает кровь.", "Медицинский справочник")
add_fact("2", "Аспирин может вызывать кровотечение желудка.", "PubMed: Article 123")
add_fact("3", "Парацетамол (Paracetamol) является жаропонижающим и обезболивающим.", "Медицинский справочник")
