import os
import glob
import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'knowledge_base'))
OUT_DIR = os.path.join(os.path.dirname(__file__), 'index')
os.makedirs(OUT_DIR, exist_ok=True)

def load_kb_files(kb_dir):
    files = glob.glob(os.path.join(kb_dir, '*.txt'))
    docs = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                docs.append({'source': os.path.basename(fp), 'text': text})
    return docs

def chunk_text(text, max_len=500, overlap=100):
    """Chunk text by paragraphs, splitting long paragraphs into overlapping windows.

    - max_len: maximum characters per chunk
    - overlap: number of overlapping characters between consecutive chunks
    """
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    cur = ''
    for p in paras:
        # if adding this paragraph keeps us within max_len, append to current
        if cur and len(cur) + 2 + len(p) <= max_len:
            cur = cur + '\n\n' + p
            continue

        # otherwise, flush current
        if cur:
            chunks.append(cur)
            cur = ''

        # if paragraph small enough, start new current
        if len(p) <= max_len:
            cur = p
            continue

        # paragraph too long -> create overlapping windows
        start = 0
        step = max_len - overlap if (max_len - overlap) > 0 else max_len
        while start < len(p):
            end = start + max_len
            chunk = p[start:end]
            chunks.append(chunk)
            if end >= len(p):
                break
            start += step

    if cur:
        chunks.append(cur)
    return chunks

def build_index():
    print('Loading knowledge base from', KB_DIR)
    docs = load_kb_files(KB_DIR)
    all_texts = []
    metadata = []
    for d in docs:
        chunks = chunk_text(d['text'], max_len=800)
        for i, c in enumerate(chunks):
            all_texts.append(c)
            metadata.append({'source': d['source'], 'chunk': i})
    if not all_texts:
        print('No documents found in knowledge base. Aborting index build.')
        return

    print(f'Found {len(all_texts)} text chunks. Computing embeddings...')
    # Use a TF-IDF vectorizer with n-grams as a lightweight embedding fallback
    # Increase max_features and use unigrams+bigrams for better recall
    vectorizer = TfidfVectorizer(max_features=2048, ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(all_texts).toarray().astype('float32')
    # save vectorizer for query time
    joblib.dump(vectorizer, os.path.join(OUT_DIR, 'vectorizer.joblib'))

    # normalize for cosine similarity with inner product index
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(OUT_DIR, 'faiss_index.bin'))
    np.save(os.path.join(OUT_DIR, 'embeddings.npy'), embeddings)
    with open(os.path.join(OUT_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'texts': all_texts, 'metadatas': metadata}, f, ensure_ascii=False, indent=2)

    print('Index saved to', OUT_DIR)

if __name__ == '__main__':
    build_index()
