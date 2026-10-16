"""Redis cache for answers, so repeated questions skip retrieval and the LLM call."""
import hashlib
import json
import os
import re

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = 24 * 60 * 60  # keep answers for one day


def _connect():
    try:
        client = redis.Redis.from_url(
            REDIS_URL, decode_responses=True, socket_connect_timeout=1
        )
        client.ping()
        print("Redis cache connected.")
        return client
    except Exception:
        print("Redis not available, running without cache.")
        return None


client = _connect()


def docs_fingerprint(folder="docs"):
    """Changes when a document is added, removed or edited, so stale answers are never reused."""
    parts = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and not name.startswith("."):
            st = os.stat(path)
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def make_key(question, n_chunks, model):
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    raw = f"{model}|{n_chunks}|{docs_fingerprint()}|{normalized}"
    return "rag:answer:" + hashlib.sha256(raw.encode()).hexdigest()


def cache_get(question, n_chunks, model):
    if client is None:
        return None
    try:
        value = client.get(make_key(question, n_chunks, model))
        return json.loads(value) if value else None
    except Exception:
        return None


def cache_set(question, n_chunks, model, answer, sources):
    if client is None or answer.startswith("Error generating response"):
        return  # never cache failures
    try:
        client.setex(
            make_key(question, n_chunks, model),
            CACHE_TTL_SECONDS,
            json.dumps({"answer": answer, "sources": sources}),
        )
    except Exception:
        pass
