import os
import uuid
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from docx import Document as DocxDocument
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

# ---------- Step 1: Load documents ----------
def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_file(path):
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_docx_file(path):
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)


def read_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_text_file(path)
    if ext == ".pdf":
        return read_pdf_file(path)
    if ext == ".docx":
        return read_docx_file(path)
    raise ValueError(f"Unsupported file format: {ext}")


# ---------- Step 2: Chunking ----------
def split_text(text, chunk_size=500):
    sentences = text.replace("\n", " ").split(". ")
    chunks, current, current_size = [], [], 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not sentence.endswith("."):
            sentence += "."

        size = len(sentence)
        if current_size + size > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_size = [sentence], size
        else:
            current.append(sentence)
            current_size += size

    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------- Step 3: Vector database ----------
chroma_client = chromadb.PersistentClient(path="chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="documents_collection",
    embedding_function=embedding_fn,
)


# ---------- Step 4: Index documents ----------
def process_document(path):
    try:
        content = read_document(path)
        chunks = split_text(content)
        name = os.path.basename(path)
        metadatas = [{"source": name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{name}_chunk_{i}" for i in range(len(chunks))]
        return ids, chunks, metadatas
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return [], [], []


def add_to_collection(collection, ids, texts, metadatas, batch_size=100):
    for i in range(0, len(texts), batch_size):
        j = i + batch_size
        collection.upsert(
            ids=ids[i:j], documents=texts[i:j], metadatas=metadatas[i:j]
        )


def process_and_add_documents(collection, folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isfile(path) or file.startswith("."):
            continue
        print(f"Processing {file}...")
        ids, texts, metas = process_document(path)
        add_to_collection(collection, ids, texts, metas)
        print(f"  Added {len(texts)} chunks")


# ---------- Step 5: Semantic search ----------
def semantic_search(collection, query, n_results=3):
    return collection.query(query_texts=[query], n_results=n_results)


def get_context_with_sources(results):
    context = "\n\n".join(results["documents"][0])
    sources = [
        f"{m['source']} (chunk {m['chunk']})" for m in results["metadatas"][0]
    ]
    return context, sources


# ---------- Step 6: Generate answer with Groq ----------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"


def get_prompt(context, conversation_history, query):
    return f"""Based on the following context and conversation history,
please provide a relevant and contextual response.
If the answer cannot be derived from the context, only use the conversation
history or say "I cannot answer this based on the provided information."

Context from documents:
{context}

Previous conversation:
{conversation_history}

Human: {query}

Assistant:"""


def generate_response(query, context, conversation_history=""):
    prompt = get_prompt(context, conversation_history, query)
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"


# ---------- Step 7: Conversation memory ----------
conversations = {}


def create_session():
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    return session_id


def add_message(session_id, role, content):
    conversations.setdefault(session_id, []).append(
        {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
    )


def format_history_for_prompt(session_id, max_messages=5):
    history = conversations.get(session_id, [])[-max_messages:]
    lines = []
    for m in history:
        role = "Human" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def contextualize_query(query, conversation_history):
    if not conversation_history:
        return query
    system = (
        "Given a chat history and the latest user question which might "
        "reference context in the chat history, formulate a standalone "
        "question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and "
        "otherwise return it as is."
    )
    try:
        r = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Chat history:\n{conversation_history}\n\nQuestion:\n{query}",
                },
            ],
            temperature=0,
        )
        return r.choices[0].message.content
    except Exception as e:
        print(f"Error contextualizing query: {e}")
        return query


# ---------- Step 8: Full RAG pipeline ----------
def conversational_rag_query(collection, query, session_id, n_chunks=3):
    history = format_history_for_prompt(session_id)
    standalone = contextualize_query(query, history)
    print(f"[Searching for: {standalone}]")

    results = semantic_search(collection, standalone, n_chunks)
    context, sources = get_context_with_sources(results)

    answer = generate_response(standalone, context, history)

    add_message(session_id, "user", query)
    add_message(session_id, "assistant", answer)
    return answer, sources


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not found. Check your .env file.")

    process_and_add_documents(collection, "docs")

    session = create_session()
    print("\nRAG chatbot ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        answer, sources = conversational_rag_query(collection, q, session)
        print(f"\nAssistant: {answer}")
        print("Sources:", ", ".join(sources), "\n")
