# Importaciones principales
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
#llama-3.2-90b-vision-preview
#meta-llama/llama-4-scout-17b-16e-instruct
# Configuración del LLM con Groq
llm = ChatOpenAI(
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.6,
    openai_api_key="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"
)

# Configuración de embeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
embed_model = FastEmbedEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_length=384
)

# Carga y procesamiento del documento
chroma_dir = "chroma_db_dir"
if Path(chroma_dir).exists():
    print("✅ Cargando base vectorial existente...")
    vectorstore = Chroma(
        embedding_function=embed_model,
        persist_directory=chroma_dir,
        collection_name="el-caballero"
    )
    # Cargamos los documentos existentes para BM25
    docs = list(vectorstore.get()['documents'])  # Obtenemos los textos
else:
    print("🔄 Procesando PDF y creando base vectorial...")
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    loader = PyMuPDFLoader("src/Los-miserables.pdf")
    data_pdf = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True
    )
    docs = text_splitter.split_documents(data_pdf)
    
    # Creamos ChromaDB
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory=chroma_dir,
        collection_name="el-caballero"
    )
    docs = [doc.page_content for doc in docs]  # Preparamos para BM25

# Configuración del Ensemble Retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_texts(docs)
bm25_retriever.k = 10

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.4, 0.6]
)

# Configuración del prompt
from langchain.prompts import PromptTemplate
custom_prompt_template = """Eres un experto analizando documentos. Responde en español usando solo el contexto proporcionado.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
- Si la pregunta no puede responderse con el contexto, di "No tengo información sobre esto en el documento"
- Sé preciso y conciso
- Cita las páginas relevantes cuando sea posible

Respuesta:"""
prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=['context', 'question']
)

# Sistema QA mejorado
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,  # Usamos el ensemble retriever
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=True
)

# Función para mostrar resultados
def ask_question(question):
    print(f"\n🔍 Pregunta: {question}")
    
    # Depuración: Mostrar chunks recuperados
    print("\n=== CHUNKS RECUPERADOS ===")
    retrieved_docs = ensemble_retriever.get_relevant_documents(question)
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get('page', 'N/A') if hasattr(doc, 'metadata') else 'N/A'
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        print(f"\n📄 Chunk {i+1} (Página {source}):")
        print(content[:200] + "...")
    
    # Obtener respuesta
    response = qa.invoke({"query": question})
    
    print("\n💡 Respuesta:")
    print(response['result'])
    """
    print("\n📚 Fuentes utilizadas:")
    for doc in response['source_documents']:
        page = doc.metadata.get('page', 'N/A')
        print(f"- Página {page}: {doc.page_content[:100]}...")
    """

# Ejemplo de uso interactivo
while True:
    print("\n" + "="*50)
    user_question = input("\nIngresa tu pregunta (o 'salir' para terminar): ")
    if user_question.lower() == 'salir':
        break
    ask_question(user_question)