import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_groq_app.groq_llm import GroqLLM


KB_INDEX_DIR = "kb_faiss"
DATA_CSV = "data/saldos.csv"


def load_vectorstore(index_dir: str = KB_INDEX_DIR):
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    idx = Path(index_dir)
    if not idx.exists():
        raise FileNotFoundError(f"Índice FAISS no encontrado en {idx}. Ejecuta index_kb primero.")
    return FAISS.load_local(str(idx), emb)


def load_balances(csv_path: str = DATA_CSV) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV de saldos no encontrado en {p}")
    df = pd.read_csv(p)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def find_balance(df: pd.DataFrame, id_value: str) -> Optional[str]:
    # columns that might identify the ID
    id_cols = [c for c in df.columns if any(k in c for k in ("id", "cedula", "dni", "document"))]
    if not id_cols:
        return None
    for col in id_cols:
        # compare as string
        match = df[df[col].astype(str).str.strip() == id_value.strip()]
        if not match.empty:
            # assume column 'balance' or 'saldo' or similar
            bal_cols = [c for c in df.columns if any(k in c for k in ("balance", "saldo", "amount", "monto"))]
            bal = match.iloc[0][bal_cols[0]] if bal_cols else "(saldo no disponible)"
            return f"ID encontrado en columna '{col}'. Saldo: {bal}"
    return None


def is_balance_query(text: str) -> Optional[str]:
    # simple heuristics: contains palabra 'saldo' or 'balance' and un numero/ID
    if re.search(r"\bsaldo\b|\bbalance\b|saldo de la|saldo para|consultar saldo", text, re.I):
        # try to extract ID-like token (numbers with 6-12 digits or alphanum)
        m = re.search(r"(\d{6,12})", text)
        if m:
            return m.group(1)
        # or extract token after 'cedula' or 'id'
        m2 = re.search(r"\b(?:cedula|cédula|dni|id)[:\s]*([A-Za-z0-9\-]+)", text, re.I)
        if m2:
            return m2.group(1)
    return None


def is_kb_query(text: str) -> bool:
    kb_terms = [
        "abrir cuenta",
        "apertura de cuenta",
        "transferencia",
        "hacer una transferencia",
        "deposito",
        "depósito",
        "tarjeta",
        "cheque",
        "requisitos",
        "documentos",
    ]
    for t in kb_terms:
        if t in text.lower():
            return True
    return False


def main_loop():
    print("Iniciando aplicación LangChain + Groq (CLI). Escribe 'salir' para terminar.")

    # prepare resources
    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        print("Advertencia: no se pudo cargar el índice FAISS:", e)
        retriever = None

    try:
        df = load_balances()
    except Exception as e:
        print("Advertencia: no se pudo cargar CSV de saldos:", e)
        df = None

    llm_model = os.environ.get("GROQ_MODEL", "groq-1")
    llm = GroqLLM(model=llm_model)

    qa_chain = None
    if retriever:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    while True:
        user = input("\nPregunta> ")
        if not user:
            continue
        if user.strip().lower() in {"salir", "exit", "quit"}:
            print("Adiós")
            break

        # 1) Balance query
        bal_id = is_balance_query(user)
        if bal_id and df is not None:
            res = find_balance(df, bal_id)
            if res:
                print("[Respuesta - Balance]")
                print(res)
                continue
            else:
                print("[Balance] No se encontró información para el ID proporcionado.")
                continue

        # 2) KB query
        if is_kb_query(user) and qa_chain is not None:
            print("[Respuesta - Base de Conocimientos] Recuperando documentos relevantes...")
            answer = qa_chain.run(user)
            print(answer)
            continue

        # 3) General query -> LLM
        print("[Respuesta - LLM] Generando respuesta con el modelo...")
        try:
            prompt = f"Responde brevemente en español. Pregunta: {user}\nRespuesta:"
            out = llm(prompt)
            print(out)
        except Exception as e:
            print("Error al usar el LLM:", e)


if __name__ == "__main__":
    main_loop()
