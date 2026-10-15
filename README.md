# RAG chatbot without LangChain

A small question-answering bot that reads my own documents (PDF, DOCX, TXT) and answers questions about them. It is written in plain Python with ChromaDB for storage and Groq for the language model. No LangChain, no LlamaIndex.

I built it by following the FutureSmart AI tutorial "Building RAG Applications Without LangChain or LlamaIndex", then changed a few things: Groq instead of OpenAI, `pypdf` instead of the abandoned PyPDF2, and `upsert` so re-running the script doesn't choke on duplicate chunk IDs.

## Why skip the frameworks

I wanted to see what RAG actually does before hiding it behind abstractions. When every step is a function of ten lines you can read, a bad answer is easy to trace: either the wrong chunks came back, or the right chunks came back and the model ignored them. With a framework in the middle I would have been guessing.

## The idea in a paragraph

A language model knows nothing about my files. So I cut each file into small pieces, turn every piece into a vector that represents its meaning, and store those vectors in a database. When a question arrives, I turn it into a vector too, pull the few closest pieces out of the database, paste them into a prompt, and tell the model to answer from that text only. That is the whole trick.

## How it flows

```
docs/ -> read text -> split into chunks -> embed -> ChromaDB      (once, at startup)

question -> rewrite as standalone -> search ChromaDB -> prompt Groq -> answer + sources
```

## What each part does

**Reading files.** One small function per format, and a dispatcher that picks one from the file extension. Anything else raises an error instead of being silently skipped.

**Chunking.** The text is split on sentence boundaries and packed into chunks of about 500 characters. Small chunks make search more precise and keep the prompt short. Splitting on sentences means a chunk never ends mid-thought.

**Embeddings.** Groq doesn't offer an embedding model, so I use `all-MiniLM-L6-v2` from sentence-transformers, which runs locally on CPU. It is small, free, and good enough for this size of project. ChromaDB calls it automatically, both when indexing and when searching.

**Storage.** ChromaDB with a persistent client, so the index lives in `chroma_db/` and survives restarts. Each chunk is stored with an ID like `report.pdf_chunk_3` and metadata with the file name and chunk number. That metadata is how the bot can say where an answer came from.

**Answering.** The top three chunks go into a prompt along with the recent chat history and the question. Temperature is 0 so answers stay close to the text. The prompt tells the model to say it cannot answer if the context doesn't contain the answer, which is what stops it from making things up.

**Follow-up questions.** This was the part I found most interesting. A vector search for "Where is it located?" returns nothing useful, because the database has no idea what "it" is. So before searching, I ask the model to rewrite the question using the chat history, which turns it into "Where is GreenGrow Innovations located?". The search then works. The last five messages are kept as history.

## Running it

You need Python 3.10 or newer and a free Groq API key from https://console.groq.com.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then put your real key in .env
python rag.py
```

The first run downloads the embedding model (around 90 MB) and indexes whatever is in `docs/`. After that you get a prompt in the terminal. Type `exit` to quit. Drop your own files into `docs/` and run it again, and they get indexed automatically.

A sample document, `docs/greengrow.txt`, is included so you can try it straight away:

```
You: When was GreenGrow Innovations founded?
Assistant: GreenGrow Innovations was founded in 2010.
Sources: greengrow.txt (chunk 0), greengrow.txt (chunk 1)

You: Where is it located?
[Searching for: Where is GreenGrow Innovations located?]
Assistant: It started in a small garage in Portland, Oregon, and now has offices in California and Iowa.
```

## Things worth tuning

- `MODEL` in `rag.py`: the Groq model. I use `openai/gpt-oss-20b`. Groq changes its model list from time to time, so check what your key can access.
- `chunk_size` in `split_text`: bigger chunks carry more context, smaller ones match more precisely. If you change it, delete `chroma_db/` and re-run so everything is re-indexed.
- `n_chunks` in `conversational_rag_query`: how many chunks go to the model. Raise it if answers miss details.
- `max_tokens` in `generate_response`: set to 2000 because the gpt-oss models spend part of the limit on reasoning before they answer.

## What it can't do yet

Chunks don't overlap, so an answer that straddles two chunks can get lost. Nothing filters out weak matches, so an unrelated question still retrieves the closest chunks, although the prompt usually makes the model decline. Scanned PDFs return no text because there is no OCR. And it's terminal only.

## Next

I want to add a distance cutoff for retrieval, overlapping chunks, and a small Streamlit front end.

## Credit

Based on the FutureSmart AI tutorial by Sakalya Mitra and Pradip Nichite.

## Chat memory in PostgreSQL

Conversation history is stored in a local PostgreSQL database instead of a Python dictionary, so it survives restarts.

Every question and answer is saved as a row in a `chat_messages` table (session id, role, content, the sources used, timestamp). On each question the bot reads the last five messages of the current session from the database, uses them to rewrite follow-ups into standalone questions, and includes them in the prompt. At startup it offers to continue the previous conversation, and typing `history` prints the saved messages.

Setup on a Mac:

```bash
brew install postgresql@18
brew services start postgresql@18
createdb ragdb
pip install -r requirements.txt
```

The connection string lives in `.env` as `DATABASE_URL` (see `.env.example`). The table is created automatically on the first run.
