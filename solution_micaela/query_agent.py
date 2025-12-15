import os
import re
import json
import faiss
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
INDEX_DIR = os.path.join(os.path.dirname(__file__), 'index')
SALDOS_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'saldos.csv'))

def load_index():
    idx_path = os.path.join(INDEX_DIR, 'faiss_index.bin')
    meta_path = os.path.join(INDEX_DIR, 'metadata.json')
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        raise FileNotFoundError('Index not found. Run build_index.py first.')
    index = faiss.read_index(idx_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    texts = meta['texts']
    metadatas = meta['metadatas']
    return index, texts, metadatas

def lookup_balance(id_cedula):
    if not os.path.exists(SALDOS_CSV):
        return None
    df = pd.read_csv(SALDOS_CSV)
    row = df[df['ID_Cedula'].astype(str).str.strip() == id_cedula.strip()]
    if row.empty:
        return None
    return {'ID_Cedula': row.iloc[0]['ID_Cedula'], 'Nombre': row.iloc[0]['Nombre'], 'Balance': float(row.iloc[0]['Balance'])}

def retrieve_docs(query, top_k=3):
    # load TF-IDF vectorizer produced during indexing
    vec_path = os.path.join(INDEX_DIR, 'vectorizer.joblib')
    if not os.path.exists(vec_path):
        raise FileNotFoundError('Vectorizer not found. Run build_index.py first.')
    vectorizer = joblib.load(vec_path)
    q_emb = vectorizer.transform([query]).toarray().astype('float32')
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    index, texts, metas = load_index()
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(texts):
            continue
        results.append({'text': texts[idx], 'meta': metas[idx]})
    return results

def route_and_respond(question):
    # detect ID pattern like V-12345678
    id_match = re.search(r"\b[Vv]-?\d{6,8}\b", question)
    if id_match:
        id_val = id_match.group(0).replace(' ', '')
        bal = lookup_balance(id_val)
        if bal:
            return f"Balance para {bal['Nombre']} ({bal['ID_Cedula']}): {bal['Balance']}"
        else:
            return f"No se encontró balance para el ID {id_val}."

    # detect keywords for KB
    kb_keywords = ['abrir cuenta', 'transferencia', 'tarjeta', 'tarjetas', 'cuenta', 'transferir']
    lowered = question.lower()
    if any(k in lowered for k in kb_keywords):
        docs = retrieve_docs(question, top_k=4)
        # If OpenAI key present, you could call an LLM to synthesize; fallback to returning retrieved docs
        answer = 'He encontrado estos fragmentos relevantes de la base de conocimientos:\n\n'
        for d in docs:
            answer += f"Fuente: {d['meta']['source']} (chunk {d['meta']['chunk']})\n{d['text']}\n\n"
        return answer

    # General response: fall back to simple reply (LLM integration optional)
    return "Consulta general detectada. Si desea una respuesta generada por un LLM configure `OPENAI_API_KEY` y actualice este script para usar LangChain/OpenAI." 

def main():
    print('Agente de consulta. Escriba su pregunta y pulse Enter (Ctrl+C para salir).')
    while True:
        q = input('\nPregunta: ').strip()
        if not q:
            continue
        try:
            resp = route_and_respond(q)
            print('\nRespuesta:\n')
            print(resp)
        except Exception as e:
            print('Error:', e)

if __name__ == '__main__':
    main()
