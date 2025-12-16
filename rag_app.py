# Disable telemetry and warnings FIRST
import os
import warnings
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['POSTHOG_DISABLED'] = 'True'
warnings.filterwarnings('ignore')

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.node_parser import SimpleNodeParser
import chromadb
import time
import json

# Configure models
Settings.llm = Ollama(model="qwen2.5:latest", request_timeout=180.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.node_parser = SimpleNodeParser.from_defaults(
    chunk_size=1024,
    chunk_overlap=200
)

print("=" * 80)
print("HR Document RAG System - Smart Indexing")
print("=" * 80)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("hr_documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Track indexed files
INDEXED_FILES_PATH = "./chroma_db/indexed_files.json"

def load_indexed_files():
    if os.path.exists(INDEXED_FILES_PATH):
        with open(INDEXED_FILES_PATH, 'r') as f:
            return json.load(f)
    return []

def save_indexed_files(files):
    with open(INDEXED_FILES_PATH, 'w') as f:
        json.dump(files, f)

# Get current files
if not os.path.exists("./data"):
    os.makedirs("./data")
    print("\n⚠️  Created ./data folder. Please add your HR PDF files there and run again.\n")
    exit()

current_files = [f for f in os.listdir("./data") if f.lower().endswith('.pdf')]
indexed_files = load_indexed_files()

new_files = [f for f in current_files if f not in indexed_files]
removed_files = [f for f in indexed_files if f not in current_files]

if not current_files:
    print("\n⚠️  No PDF files found in ./data folder!\n")
    exit()

# Show status
print(f"\n📊 Status:")
print(f"   Total PDFs in folder: {len(current_files)}")
print(f"   Already indexed: {len(indexed_files)}")
print(f"   New PDFs to add: {len(new_files)}")
if removed_files:
    print(f"   Removed PDFs: {len(removed_files)}")

# Ask what to do
if new_files:
    print(f"\n📄 New files detected:")
    for f in new_files:
        print(f"   + {f}")
    
    choice = input("\n🔄 [1] Add new files only  [2] Rebuild all  [3] Cancel: ").strip()
    
    if choice == "3":
        print("Cancelled.")
        exit()
    elif choice == "2":
        print("\n🗑️  Rebuilding entire index...")
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db", ignore_errors=True)
            time.sleep(1)
        
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("hr_documents")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        print("📄 Loading all PDF documents...")
        documents = SimpleDirectoryReader("./data", required_exts=[".pdf"]).load_data()
        
        print(f"✅ Loaded {len(documents)} document sections")
        print("🔨 Building vector index...")
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print(f"✅ Index created with {chroma_collection.count()} chunks!")
        save_indexed_files(current_files)
        
    else:  # Add new files only
        print("\n➕ Adding new files to existing index...")
        
        # Load existing index
        if chroma_collection.count() > 0:
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            
            # Load only new files
            new_documents = []
            for new_file in new_files:
                file_path = os.path.join("./data", new_file)
                print(f"   Processing: {new_file}")
                docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                new_documents.extend(docs)
            
            print(f"✅ Loaded {len(new_documents)} new document sections")
            print("🔨 Adding to index...")
            
            # Add new documents to existing index
            for doc in new_documents:
                index.insert(doc)
            
            print(f"✅ Index updated! Total chunks: {chroma_collection.count()}")
            
            # Update tracked files
            indexed_files.extend(new_files)
            save_indexed_files(indexed_files)
        else:
            # No existing index, create new
            print("No existing index found. Creating new index...")
            documents = SimpleDirectoryReader("./data", required_exts=[".pdf"]).load_data()
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            print(f"✅ Index created with {chroma_collection.count()} chunks!")
            save_indexed_files(current_files)

elif removed_files:
    print(f"\n⚠️  Some previously indexed files are missing:")
    for f in removed_files:
        print(f"   - {f}")
    
    choice = input("\n🔄 Rebuild index to remove them? (y/n): ").strip().lower()
    if choice == 'y':
        print("\n🗑️  Rebuilding index...")
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db", ignore_errors=True)
            time.sleep(1)
        
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("hr_documents")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        documents = SimpleDirectoryReader("./data", required_exts=[".pdf"]).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print(f"✅ Index rebuilt with {chroma_collection.count()} chunks!")
        save_indexed_files(current_files)
    else:
        print("Index unchanged.")
        exit()

else:
    print("\n✅ All files already indexed!")
    print(f"   Total: {len(current_files)} PDFs, {chroma_collection.count()} chunks")
    
    choice = input("\n🔄 Rebuild anyway? (y/n): ").strip().lower()
    if choice != 'y':
        print("Exiting.")
        exit()
    
    print("\n🗑️  Rebuilding entire index...")
    if os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db", ignore_errors=True)
        time.sleep(1)
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("hr_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    documents = SimpleDirectoryReader("./data", required_exts=[".pdf"]).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print(f"✅ Index created with {chroma_collection.count()} chunks!")
    save_indexed_files(current_files)

print("\n" + "=" * 80)
print("✅ Done! Run 'streamlit run streamlit_app.py' to use the updated index.")
print("=" * 80)