import os
import json
import uuid
import time
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- Ollama & Chroma Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# Note: Ollama runs locally, so an API key isn't needed, but we keep .env for consistency if needed.
load_dotenv()
PDF_PATH = r"D:\Gen AI\Ollama\Olama_work\CVGF_IOM.pdf"  # CHANGE THIS to your PDF file name
CHROMA_DIR = "chroma_db_rag_data_ollama"
LOG_FILE = "conversation_log_ollama.json"
SESSION_ID = str(uuid.uuid4())

# --- Ollama Setup ---
OLLAMA_MODEL = "llama3.2:latest" # Ensure this model is pulled and running in Ollama
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama server address

# 1. LLM Setup (for response generation)
# Note: Ollama class can handle chat and text generation
LLM = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

# 2. Embedding Setup (for vectorizing documents and queries)
# Note: OllamaEmbeddings uses the specified model for embeddings (llama3:8b is often suitable)
EMBEDDING_MODEL = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

# --- 1. Document Indexing (RAG Setup) ---
def index_document(pdf_path: str) -> Chroma:
    """Loads a PDF, splits it, embeds it using Ollama, and saves it to ChromaDB."""
    
    if os.path.exists(CHROMA_DIR) and os.path.isdir(CHROMA_DIR):
        print(f" Loading existing ChromaDB from {CHROMA_DIR}...")
        # IMPORTANT: When reloading, you MUST pass the SAME embedding function
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDING_MODEL)

    if not os.path.exists(pdf_path):
         raise FileNotFoundError(f"FATAL: Document '{pdf_path}' not found.")

    print(f" Indexing document '{pdf_path}' using {OLLAMA_MODEL}. This may take a moment...")
    
    # 1. Load the document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    docs = text_splitter.split_documents(documents)
    print(f"   -> Split into {len(docs)} chunks.")

    # 3. Create Vector Store (Embeddings + Indexing)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=EMBEDDING_MODEL, # Using OllamaEmbeddings
        persist_directory=CHROMA_DIR,
    )
    vectorstore.persist()
    print(f"   -> ChromaDB created and saved to {CHROMA_DIR}")
    return vectorstore

# --- 2. JSON Logging Function (Unchanged) ---
def log_conversation(session_id: str, turn: int, user_query: str, ai_response: str, sources: list):
    """
    Appends the current turn to a JSON conversation log file.
    """
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            # Handle empty file case
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = {"session_id": session_id, "messages": []}
    else:
        log_data = {"session_id": session_id, "messages": []}

    # Extract source paths/page numbers from LangChain Document objects
    source_details = [
        f"{doc.metadata.get('source', 'N/A')}: Page {doc.metadata.get('page', 'N/A')}" 
        for doc in sources
    ]

    log_data["messages"].append({
        "turn": turn,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": user_query,
        "ai_response": ai_response,
        "retrieved_sources": source_details
    })

    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"\n[LOG] Conversation turn {turn} logged to {LOG_FILE}")
    print("-" * 50)


# --- 3. Chat Logic with Memory and RAG Chain (Prompt updated for better Ollama handling) ---
def run_chat(vectorstore: Chroma):
    """
    The main conversation loop using a history-aware RAG chain.
    """
    # 1. Conversation History (Memory)
    memory = ConversationBufferWindowMemory(
        k=5, 
        return_messages=True, 
        memory_key="chat_history", 
        input_key="input"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2. History-Aware Retriever Chain
    # We use ChatPromptTemplate for better structure and compatibility with Ollama
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone, search-engine-ready question. If the input is already standalone, return it as is. Do NOT answer the question, just rephrase."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        LLM, retriever, contextualize_q_prompt
    )

    # 3. Final Answer Generation Chain
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert AI assistant that answers questions based ONLY on the provided context.\
              If the answer is not in the context, state that you cannot find the information. \nContext: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 4. Final RAG Chain combining retrieval and generation
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_prompt | LLM | StrOutputParser())

    print(f"\n Starting Ollama Session (Model: {OLLAMA_MODEL}, Session ID: {SESSION_ID}). Type 'exit' to quit.")
    
    turn = 0
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        turn += 1
        
        # 5. Invoke the RAG Chain
        try:
            # Pass the input and current memory contents to the chain
            chain_input = {"input": user_input, "chat_history": memory.chat_memory.messages}
            
            # Use .invoke() to run the chain
            result = rag_chain.invoke(chain_input)
            ai_response = result['answer']
            sources = result['context'] # The documents retrieved

            print(f"AI: {ai_response}")
            
            # Print sources for user (optional, but good for RAG)
            unique_sources = set(
                f"Page {doc.metadata.get('page', 'N/A')} from {doc.metadata.get('source', 'N/A')}"
                for doc in sources
            )
            if unique_sources:
                 print(f"\n[Sources Found]: {', '.join(unique_sources)}")
            
            # 6. Update Memory
            memory.chat_memory.add_message(HumanMessage(content=user_input))
            memory.chat_memory.add_message(AIMessage(content=ai_response))
            
            # 7. Log Conversation
            log_conversation(SESSION_ID, turn, user_input, ai_response, sources)

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your Ollama server is running and the model is available.")
            break

# --- Main Execution ---
if __name__ == "__main__":
    
    # Pre-flight check for Ollama server
    try:
        from langchain_community.llms import Ollama
        # A simple connection test
        Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL).invoke("test") 
        print(f"🚀 Ollama server connected successfully with model {OLLAMA_MODEL}.")
    except Exception:
        print(f"FATAL: Could not connect to Ollama server at {OLLAMA_BASE_URL} or model {OLLAMA_MODEL} is not available.")
        print("Please run 'ollama serve' and 'ollama pull llama3:8b' in your terminal.")
        exit()
        
    # Check for PDF
    if not os.path.exists(PDF_PATH):
        print(f"FATAL: Document '{PDF_PATH}' not found. Please place your PDF in the project directory.")
        exit()
    else:
        # Step 1: Index Document
        chroma_store = index_document(PDF_PATH)

        # Step 2: Run Chat
        run_chat(chroma_store)