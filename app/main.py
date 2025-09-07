import psycopg2
import uvicorn
from fastapi import FastAPI

from config import settings
from database import init_index

app = FastAPI()

index = init_index(provider=settings.model_provider)


@app.get("/health")
def health():
    """Checks the status of the API."""
    return {"status": "all good!"}


@app.get("/health-vectordb")
def health_vectordb():
    """Checks the status of the vector store connection."""
    conn = psycopg2.connect(settings.postgres_conn_str)
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"SELECT * FROM pg_catalog.pg_tables WHERE schemaname = 'public'")
        rows = c.fetchall()

    return rows


@app.post("/query")
def query(query: str):
    """Runs retrieval augmented generation (RAG) over a vector store."""
    global index

    query_engine = index.as_query_engine()
    results = query_engine.query(query)
    return results


@app.post("/retrieve")
def retrieve(query: str) -> list[str]:
    """Runs retrieval over a vector store."""
    global index

    retriever = index.as_retriever()
    results = retriever.retrieve(query)
    return [
        node.text for node in results
    ]  # api connection error when running in docker but not in host session


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
