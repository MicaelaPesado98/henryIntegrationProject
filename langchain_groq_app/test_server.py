import os
from fastapi.testclient import TestClient
from langchain_groq_app import server

os.environ['GROQ_API_KEY'] = 'dummy'

client = TestClient(server.app)

print('STATUS:', client.get('/status').json())
print('BAL:', client.post('/query', json={'query':'Consultar saldo cedula V-12345678'}).json())
print('KB:', client.post('/query', json={'query':'¿Qué documentos se necesitan para abrir cuenta?'}).json())
