"""MCP server exposing the student's scientific initiation paper (TP2 ChromaDB index)."""
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scientific-initiation-paper")
_store = None


def get_store():
    global _store
    if _store is None:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        _store = Chroma(collection_name="rapport_recursive",
                        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
                        persist_directory="./chroma_db")
    return _store


@mcp.tool()
def search_paper(query: str, k: int = 4) -> str:
    """Search the student's scientific initiation paper and return the k most relevant passages with page numbers."""
    if not Path("./chroma_db").exists():
        return "paper index not found (chroma_db/ is missing)."
    docs = get_store().similarity_search(query, k=k)
    return "\n\n".join(f"[page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs) or "No passage found."


@mcp.tool()
def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())


@mcp.tool()
def list_pages(keyword: str) -> list[int]:
    """Return the page numbers of the scientific initiation paper whose text contains the keyword (case-insensitive).
    Use it for questions such as 'on which pages is X mentioned?'."""
    if not Path("./chroma_db").exists():
        return []
    data = get_store()._collection.get(include=["documents", "metadatas"])
    kw = keyword.lower()
    pages = {meta.get("page") for doc, meta in zip(data["documents"], data["metadatas"])
             if kw in doc.lower() and meta.get("page") is not None}
    return sorted(int(p) for p in pages)


# âœ… Exercise 5.6: list_pages added above


@mcp.resource("paper://info")
def paper_info() -> str:
    """Basic information about the indexed paper."""
    if not Path("./chroma_db").exists():
        return "No index."
    return f"{get_store()._collection.count()} chunks indexed from the scientific initiation paper."


if __name__ == "__main__":
    # âš ï¸ Windows: load Chroma + embeddings BEFORE the stdio loop starts. Lazy-loading them inside a tool call
    # makes the tool hang forever under stdio on Windows (agent cell 5.4 never returns).
    if Path("./chroma_db").exists():
        get_store().similarity_search("warmup", k=1)
    mcp.run(transport="stdio")    # never print() to stdout in a stdio server: stdout carries the protocol
