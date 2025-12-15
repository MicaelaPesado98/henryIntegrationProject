from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

from pathlib import Path
import re
import pandas as pd

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq_app.groq_llm import GroqLLM
from langchain_groq_app.index_kb import build_and_save_faiss


KB_INDEX_DIR = "kb_faiss"
DATA_CSV = "data/saldos.csv"


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    source: str
    answer: str


app = FastAPI(title="LangChain Groq Router")


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
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def find_balance(df: pd.DataFrame, id_value: str) -> Optional[str]:
    id_cols = [c for c in df.columns if any(k in c for k in ("id", "cedula", "dni", "document"))]
    if not id_cols:
        return None
    for col in id_cols:
        match = df[df[col].astype(str).str.strip() == id_value.strip()]
        if not match.empty:
            bal_cols = [c for c in df.columns if any(k in c for k in ("balance", "saldo", "amount", "monto"))]
            bal = match.iloc[0][bal_cols[0]] if bal_cols else "(saldo no disponible)"
            return f"ID encontrado en columna '{col}'. Saldo: {bal}"
    return None


def is_balance_query(text: str) -> Optional[str]:
    if re.search(r"\bsaldo\b|\bbalance\b|saldo de la|saldo para|consultar saldo", text, re.I):
        m = re.search(r"(\d{6,12})", text)
        if m:
            return m.group(1)
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


@app.on_event("startup")
def startup_event():
    global VECTORSTORE, RETRIEVER, BALANCES_DF, LLM, QA_CHAIN
    VECTORSTORE = None
    RETRIEVER = None
    BALANCES_DF = None
    QA_CHAIN = None

    try:
        VECTORSTORE = load_vectorstore()
        RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 4})
        print("Vectorstore cargado")
    except Exception as e:
        print("No se cargó vectorstore:", e)
        # Intentar indexar desde la KB adjunta (HW - LangChain II/knowledge_base)
        try:
            kb_dir = os.path.join(os.getcwd(), "HW - LangChain II", "knowledge_base")
            print(f"Intentando indexar KB desde: {kb_dir}")
            build_and_save_faiss(kb_dir=kb_dir, output_dir=KB_INDEX_DIR)
            VECTORSTORE = load_vectorstore()
            RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 4})
            print("Vectorstore creado y cargado desde KB adjunta")
        except Exception as e2:
            print("Fallo al indexar la KB adjunta:", e2)

    try:
        BALANCES_DF = load_balances()
        print("CSV de saldos cargado")
    except Exception as e:
        print("No se cargó CSV de saldos:", e)

    LLM = GroqLLM(model=os.environ.get("GROQ_MODEL", "groq-1"))

    if RETRIEVER is not None:
        from langchain.chains import RetrievalQA

        QA_CHAIN = RetrievalQA.from_chain_type(llm=LLM, chain_type="stuff", retriever=RETRIEVER)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/reindex")
def reindex(kb_dir: Optional[str] = None):
    try:
        build_and_save_faiss(kb_dir or "knowledge_base", output_dir=KB_INDEX_DIR)
        return {"ok": True, "detail": "Reindexado completado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    text = req.query

    # 1) Balance
    bal_id = is_balance_query(text)
    if bal_id and BALANCES_DF is not None:
        res = find_balance(BALANCES_DF, bal_id)
        if res:
            return QueryResponse(source="balance", answer=res)
        return QueryResponse(source="balance", answer="ID no encontrado")

    # Fallback: try to detect any ID from the balances dataframe present in the text
    if BALANCES_DF is not None:
        id_cols = [c for c in BALANCES_DF.columns if any(k in c for k in ("id", "cedula", "dni", "document"))]
        if id_cols:
            for col in id_cols:
                for val in BALANCES_DF[col].astype(str).tolist():
                    if val and val in text:
                        res = find_balance(BALANCES_DF, val)
                        if res:
                            return QueryResponse(source="balance", answer=res)

    # 2) KB
    if is_kb_query(text) and QA_CHAIN is not None:
        answer = QA_CHAIN.run(text)
        return QueryResponse(source="kb", answer=answer)

    # 3) LLM
    try:
        prompt = f"Responde brevemente en español. Pregunta: {text}\nRespuesta:"
        out = LLM(prompt)
        return QueryResponse(source="llm", answer=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    """Devuelve el estado de los recursos cargados en el servidor (para depuración)."""
    return {
        "vectorstore_loaded": bool(VECTORSTORE),
        "retriever_loaded": bool(RETRIEVER),
        "qa_chain_loaded": bool(QA_CHAIN),
        "balances_loaded": bool(BALANCES_DF),
        "balance_columns": list(BALANCES_DF.columns) if BALANCES_DF is not None else [],
    }
