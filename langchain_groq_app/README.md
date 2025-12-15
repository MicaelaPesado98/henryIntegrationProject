Resumen

Este pequeño proyecto demuestra un flujo simple con LangChain que decide entre: consultar un CSV de saldos, recuperar de una base de conocimientos indexada con FAISS, o responder usando un LLM (Groq).

Estructura

- `index_kb.py`: indexa archivos de `knowledge_base/` (archivos `.txt`/`.md`) usando `all-MiniLM-L6-v2` y guarda un índice FAISS en `kb_faiss/`.
- `groq_llm.py`: wrapper minimalista para llamar a la API de Groq desde LangChain.
- `app.py`: CLI interactiva que enruta consultas entre los 3 flujos.
- `requirements.txt`: dependencias.

Requisitos previos

- Python 3.8+
- Tener la carpeta `knowledge_base/` con archivos de texto (para el flujo KB).
- El CSV `data/saldos.csv` debe existir (ej.: columnas `id` y `balance` o `saldo`).

Variables de entorno

- `GROQ_API_KEY`: clave de la API de Groq.
- `GROQ_API_URL` (opcional): URL base de la API de Groq (p. ej. `https://api.groq.com/v1`).
- `GROQ_MODEL` (opcional): modelo a usar, por defecto `groq-1`.

Instalación

En PowerShell (desde la raíz del repo):

```powershell
python -m pip install -r .\langchain_groq_app\requirements.txt
```

Indexar la base de conocimientos (ejecuta esto si tienes `knowledge_base/` con archivos):

```powershell
python .\langchain_groq_app\index_kb.py --kb_dir .\knowledge_base --out .\langchain_groq_app\kb_faiss
```

Ejecutar la aplicación CLI:

```powershell
$env:GROQ_API_KEY = "tu_api_key_aqui"
$env:GROQ_API_URL = "https://api.groq.com/v1"  # si necesitas ajustarlo
python .\langchain_groq_app\app.py
```

Notas y personalización

- El wrapper `GroqLLM` hace una petición POST genérica. La forma exacta del request/response depende de la versión de la API de Groq; si la respuesta tiene otra estructura, modifica `groq_llm.py` para extraer el texto adecuado.
- El enrutador usa reglas heurísticas simples (palabras clave) para detectar consultas de saldo y consultas de KB. Se puede reemplazar por un clasificador LLM más robusto si lo deseas.
- Ajusta las columnas de `data/saldos.csv` al formato de tu CSV real (nombres de columna y tipos).

Si quieres, puedo:
- Adaptar `groq_llm.py` a la forma exacta de respuesta de la API Groq si me indicas un ejemplo de la respuesta.
- Añadir una pequeña prueba automatizada o un servidor HTTP en lugar del CLI.
\nServidor HTTP

Se ha añadido un servidor FastAPI en `langchain_groq_app/server.py` que expone los endpoints:
- `GET /health` - healthcheck
- `POST /reindex` - reconstruye el índice FAISS desde `knowledge_base/` (opcional `kb_dir` query param)
- `POST /query` - cuerpo JSON `{ "query": "tu pregunta" }`, devuelve `{ "source": "balance|kb|llm", "answer": "..." }`

Ejecutar el servidor (desde la raíz del repo):

```powershell
$env:GROQ_API_KEY = "tu_api_key"
uvicorn langchain_groq_app.server:app --host 0.0.0.0 --port 8000 --reload
```

El enrutamiento de la consulta funciona igual que en el CLI: primero intenta detectar consultas de saldo, luego consultas a la base de conocimientos (si el índice FAISS está cargado), y finalmente delega al LLM.
