# Database: chat memory

The chatbot stores its conversation memory in a local PostgreSQL database called `ragdb`, in one table, `chat_messages`.

| File | What it is |
|---|---|
| `schema.sql` | The table definition. `rag.py` also creates it automatically on first run. |
| `queries.sql` | Queries to inspect the saved memory. |
| `sample_output.txt` | Real output from running `queries.sql` after a short chat with the bot. |

## How it is used

- Every question and answer is inserted as a row, together with the session id and the sources used.
- On each new question, `rag.py` reads the last five messages of the current session, rewrites follow-up questions into standalone ones, and adds the history to the prompt.
- Because the data is in Postgres and not in Python memory, it survives restarts. At startup the bot offers to continue the last conversation.

## Look at it yourself

```bash
createdb ragdb
psql ragdb -f db/schema.sql
python rag.py                      # ask a few questions, then type exit
psql ragdb -f db/queries.sql
```
