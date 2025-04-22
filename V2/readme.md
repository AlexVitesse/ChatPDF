# Sistema de QA para Documentos con LangChain y Groq

Este proyecto implementa un sistema de preguntas y respuestas sobre documentos utilizando LangChain, Groq y técnicas avanzadas de recuperación de información. El sistema puede procesar documentos PDF (como "Los Miserables" en el ejemplo) y responder preguntas sobre su contenido.

## 📋 Requerimientos

### 🔧 Dependencias principales

Instala las siguientes bibliotecas con pip:

```bash
pip install langchain langchain-community chromadb pymupdf fastembed rank_bm25 groq
📚 Bibliotecas requeridas
langchain: Framework principal para construir cadenas de procesamiento

langchain-community: Integraciones con modelos y herramientas de la comunidad

chromadb: Base de datos vectorial para almacenar embeddings

pymupdf: Para procesamiento de archivos PDF

fastembed: Embeddings rápidos y eficientes

rank_bm25: Implementación del algoritmo BM25 para recuperación de información

groq: Para acceder a los modelos de Groq

🔑 Requisitos adicionales
API Key de Groq: Necesitas una clave API de Groq para usar sus modelos LLM. Puedes obtenerla en Groq Cloud.

Documento PDF: Por defecto el sistema busca un archivo llamado src/Los-miserables.pdf. Puedes:

Colocar tu propio PDF en esa ruta

Modificar el código para apuntar a tu archivo

🛠️ Configuración
Crea un archivo .env en el directorio raíz con tu API key de Groq:

GROQ_API_KEY=tu_api_key_aquí
Opcionalmente puedes configurar:

Modelo LLM (por defecto usa meta-llama/llama-4-maverick-17b-128e-instruct)

Parámetros de temperatura

Configuración de los chunks (tamaño y solapamiento)

🚀 Uso
Ejecuta el script principal y sigue las instrucciones:

La primera vez que se ejecute, procesará el PDF y creará una base de datos vectorial (esto puede tomar varios minutos dependiendo del tamaño del documento).

En ejecuciones posteriores, cargará la base vectorial existente para mayor velocidad.

Ingresa tus preguntas sobre el documento cuando se te solicite.

Escribe "salir" para terminar la sesión.

🧠 Funcionamiento interno
El sistema utiliza:

Ensemble Retriever: Combina:

Vector Similarity (ChromaDB con embeddings)

BM25: Algoritmo tradicional de recuperación de información

Modelo LLM de Groq: Para generar respuestas precisas basadas en el contexto recuperado

Procesamiento de documentos:

División en chunks con solapamiento

Embeddings con el modelo all-MiniLM-L6-v2

Persistencia en ChromaDB para reutilización

📄 Estructura de archivos
/proyecto
│   README.md
│   main.py                 # Script principal
│   .env                    # Configuración de entorno
│
└───src/
    │   Los-miserables.pdf  # Documento de ejemplo (puede ser reemplazado)
    │
└───chroma_db_dir/          # Base de datos vectorial (se crea automáticamente)
💡 Mejoras posibles
Añadir soporte para múltiples formatos de documento (Word, HTML, etc.)

Implementar una interfaz web o de chat

Añadir caché para respuestas frecuentes

Permitir múltiples documentos/conocimientos

pero dame todo en formato md
markdown
# Sistema de QA para Documentos con LangChain y Groq

Este proyecto implementa un sistema de preguntas y respuestas sobre documentos utilizando LangChain, Groq y técnicas avanzadas de recuperación de información.

## 📋 Requisitos Técnicos

### 🔧 Dependencias Principales

```bash
pip install langchain langchain-community chromadb pymupdf fastembed rank_bm25 groq python-dotenv
📚 Bibliotecas Requeridas
Biblioteca	Versión Recomendada	Descripción
langchain	>=0.1.0	Framework principal
langchain-community	>=0.0.1	Integraciones comunitarias
chromadb	>=0.4.0	Base de datos vectorial
pymupdf	>=1.22.0	Procesamiento de PDFs
fastembed	>=0.2.0	Embeddings eficientes
rank_bm25	>=0.2.1	Algoritmo BM25
groq	>=0.3.0	Cliente para Groq API
python-dotenv	>=1.0.0	Manejo de variables de entorno
⚙️ Configuración Inicial
Archivo .env:

GROQ_API_KEY=tu_api_key_aqui
MODEL_NAME=meta-llama/llama-4-maverick-17b-128e-instruct
Estructura de Directorios:

/proyecto
├── src/
│   └── Los-miserables.pdf  # Documento de ejemplo
├── chroma_db_dir/         # Base vectorial (auto-generado)
├── .env                   # Configuración
└── main.py                # Script principal
🛠️ Funcionalidades Clave
Procesamiento de Documentos
Soporte para archivos PDF

División inteligente de texto con:

Chunk size: 1000 caracteres

Overlap: 250 caracteres

Embeddings con all-MiniLM-L6-v2

Sistema de Recuperación
Ensemble Retriever que combina:

40% Vector Similarity (ChromaDB)

60% BM25

Búsqueda en top 10 chunks relevantes

Modelo de Lenguaje
Modelo: llama-4-maverick-17b-128e-instruct

Proveedor: Groq Cloud

Temperatura: 0.6 (balance creatividad/precisión)

🚀 Cómo Usar
Primera Ejecución:

bash
python main.py
Procesará el PDF y creará la base vectorial

Ejecuciones Posteriores:

Cargará la base vectorial existente

Modo interactivo para hacer preguntas

Comandos:

Ingresa tu pregunta

Escribe salir para terminar

🔍 Ejemplo de Uso
python
# Importaciones principales
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
from dotenv import load_dotenv

load_dotenv()  # Carga variables de .env

# Configuración LLM
llm = ChatOpenAI(
    model_name=os.getenv("MODEL_NAME"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.6,
    openai_api_key=os.getenv("GROQ_API_KEY")
)
📌 Notas Importantes
Requisitos de Hardware:

4GB RAM mínimo

2GB espacio en disco (para bases vectoriales grandes)

Limitaciones:

Tamaño máximo de PDF: ~50MB

Soporte experimental para otros formatos

Optimización:

Ajustar chunk_size según tipo de documento

Modificar weights del EnsembleRetriever para mejores resultados

📄 Licencia
MIT License - Libre uso y modificación