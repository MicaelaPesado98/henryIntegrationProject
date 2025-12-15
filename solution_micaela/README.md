# Solución de ejemplo: Indexado y agente de consulta

Este ejemplo muestra cómo indexar una base de conocimientos con `sentence-transformers/all-MiniLM-L6-v2` y FAISS, además de un agente CLI simple que enruta consultas a: 1) búsqueda de balances por ID en `data/saldos.csv`, 2) recuperación de documentos de la `knowledge_base/`, y 3) respuestas generales (requieren LLM configurado).

Archivos añadidos:
- `requirements.txt` — dependencias necesarias.
- `build_index.py` — genera embeddings y guarda un índice FAISS en `solution/index/`.
- `query_agent.py` — CLI que enruta preguntas y devuelve respuestas.

Uso básico:

1. Crear un entorno virtual e instalar dependencias (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r "HW - LangChain II\solution_micaela\requirements.txt"
```

2. Construir el índice (desde la raíz del repo):

```powershell
python "HW - LangChain II\solution_micaela\build_index.py"
```

3. Ejecutar el agente de consulta:

```powershell
python "HW - LangChain II\solution_micaela\query_agent.py"
```

Ejemplos de preguntas que detecta automáticamente:
- "¿Cuál es el balance de V-12345678?"  → búsqueda en `data/saldos.csv`.
- "¿Cómo abro una cuenta?" → recuperación desde `knowledge_base`.
- Preguntas generales → mensaje indicando configurar LLM (OpenAI) para respuestas generadas.

Notas y siguientes pasos:
- Para respuestas generadas por LLM: configurar `OPENAI_API_KEY` y modificar `query_agent.py` para usar `langchain` + `OpenAI` u otro modelo.
- El indexado aquí es sencillo; en producción se recomienda manejo de chunks, control de metadatos y persistencia más robusta.

**Especificaciones Técnicas**
- **Lenguaje**: Python 3.9+ (se probó en el venv del proyecto).
- **Dependencias clave**: listado en `solution_micaela/requirements.txt` (ej.: `faiss-cpu`, `scikit-learn`, `joblib`, `pandas`, `sentence-transformers` — algunas se mantienen por compatibilidad).

**Arquitectura y componentes**
- **Indexación (script)**: `solution_micaela/build_index.py`
	- Lee archivos de `HW - LangChain II/knowledge_base/*.txt`.
	- Fragmenta cada documento en chunks con overlap (parámetros: `max_len` por defecto 800 chars, `overlap` 100 chars) para mantener contexto entre fragmentos.
	- Genera vectores usando TF-IDF como solución ligera y robusta en entornos con problemas de dependencias de HF. Configuración actual: `TfidfVectorizer(ngram_range=(1,2), max_features=2048)`.
	- Normaliza vectores y crea un índice FAISS (`IndexFlatIP`) para búsquedas por similitud (cosine vía inner-product con vectores normalizados).
	- Persiste en `solution_micaela/index/`: `faiss_index.bin`, `embeddings.npy`, `metadata.json`, `vectorizer.joblib`.

- **Agente / Router (CLI)**: `solution_micaela/query_agent.py`
	- Ruteo por tipo de consulta:
		- **Consulta de balance**: detecta patrones de ID (`V-12345678`) y busca en `data/saldos.csv` (sin usar LLM). Devuelve nombre y balance.
		- **Consulta KB**: detecta palabras clave (p.ej. "abrir cuenta", "transferencia", "tarjeta") y ejecuta recuperación con FAISS + TF-IDF (se retorna fragmentos relevantes).
		- **Respuesta general**: fallback que indica cómo activar LLM (OpenAI) para generar respuestas.
	- `solution_micaela/run_tests.py` contiene pruebas de ejemplo ejecutadas automáticamente.

**Decisiones de diseño y razones**
- **TF-IDF + FAISS en vez de embeddings semánticos por defecto**: durante la implementación se encontraron conflictos de versiones entre `huggingface-hub`, `transformers` y `sentence-transformers` en el entorno del usuario. Para garantizar una solución reproducible y ligera, usé TF-IDF (local, rápido) y FAISS para búsqueda eficiente. Esto ofrece buena relevancia para KB basadas en texto corto/mediano y permite evitar sobrecargar el entorno con modelos grandes.
- **Chunking con solapamiento**: mejora la recuperación de respuestas que atraviesan límites de párrafos; reduce pérdida de contexto.
- **Persistencia del vectorizer**: `vectorizer.joblib` permite que la etapa de consulta reproduzca exactamente la transformación usada en indexación.

**Cómo migrar a embeddings semánticos (opcional)**
- Si quieres mayor calidad semántica, puedo cambiar la generación de embeddings a `sentence-transformers/all-MiniLM-L6-v2`:
	1. Resolver dependencias: instalar versiones compatibles de `huggingface-hub` y `transformers` (puede requerir actualizar o forzar versiones en el venv).
	2. Reemplazar TF-IDF por `SentenceTransformer(...).encode(...)` en `build_index.py` y `retrieve_docs`.
	3. Regenerar índice.
- Alternativa segura: descargar el modelo localmente y cargar desde disco para minimizar llamadas al Hub.

**Comandos reproducibles (PowerShell)**
- Crear venv e instalar dependencias:
```
python -m venv ".venv"
.\.venv\Scripts\Activate.ps1
pip install -r "HW - LangChain II\solution_micaela\requirements.txt"
```
- Construir índice:
```
C:/Users/lpesa/Documents/micaela/HW-iacopilot/.venv/Scripts/python.exe "HW - LangChain II\solution_micaela\build_index.py"
```
- Ejecutar pruebas automáticas:
```
C:/Users/lpesa/Documents/micaela/HW-iacopilot/.venv/Scripts/python.exe "HW - LangChain II\solution_micaela\run_tests.py"
```

**Limitaciones y notas**
- La calidad del retrieval depende de la calidad y diversidad de la KB; actualmente `transferencia.txt` y `nueva_cuenta.txt` contienen textos muy similares — esto afecta ranking.
- TF-IDF no captura relaciones semánticas profundas; para consultas abiertas o lenguaje más variado, se recomienda embeddings semánticos + re-ranking.
- Si deseas, puedo automatizar la integración con `OpenAI` (LangChain) para sintetizar las respuestas recuperadas en lenguaje natural. Necesitaré `OPENAI_API_KEY` para pruebas.
