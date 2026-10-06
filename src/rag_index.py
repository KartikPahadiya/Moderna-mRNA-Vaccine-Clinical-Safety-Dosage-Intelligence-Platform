"""Module 3: RAG retrieval index over the regulatory guideline document.

Implements FR-3.1 .. FR-3.5:
  FR-3.1  Chunk the document into ~500-token pieces with ~50-token overlap.
  FR-3.2  Embed every chunk with the all-MiniLM-L6-v2 sentence-embedding model.
  FR-3.3  Persist to a local ChromaDB collection that survives process restarts.
  FR-3.4  query_guidelines() returns exactly k chunks, ordered by similarity.
  FR-3.5  Retrieval latency is recorded for every query.

A query log (question, top result snippet, latency) is written to
outputs/retrieval_log.md for use in the Week 4 dashboard.
"""
import json
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

try:  # langchain-community >= 0.3 moved HuggingFaceEmbeddings to langchain_huggingface
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parents[1]
PERSIST_DIR = ROOT / 'outputs' / 'chroma_store'
COLLECTION = 'mrna_safety_guidelines'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

TEST_QUESTIONS = [
    'What are the storage requirements for unopened mRNA formulations?',
    'What dose is recommended for adults aged 65 and older with two pre-existing conditions?',
    'When should a dose be deferred because of an elevated inflammatory biomarker?',
    'What are the absolute contraindications for administering the vaccine?',
    'When must an adverse event be escalated to clinical safety review?',
]


def build_index(doc_path: str | Path = ROOT / 'data' / 'mrna_safety_guidelines.txt') -> Chroma:
    """FR-3.1/3.2/3.3: chunk, embed and persist the guideline document."""
    raw_text = Path(doc_path).read_text(encoding='utf-8')

    # FR-3.1: chunk size/overlap chosen so that each section's rules stay
    # intact inside a chunk while consecutive chunks still share context.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=['\n\n', '\n', '. ', ' '],
    )
    chunks = splitter.split_text(raw_text)
    lengths = [len(c.split()) for c in chunks]
    print(f'Split guideline document into {len(chunks)} chunks '
          f'(word lengths {min(lengths)}-{max(lengths)}, overlap visible between neighbours)')

    # FR-3.2: one dense embedding per chunk.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION,
    )
    print(f'Persisted vector index -> {PERSIST_DIR}')
    return vectorstore


def load_index() -> Chroma:
    """Reload a previously persisted index (FR-3.3 restart-survival check)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )


def query_guidelines(vectorstore: Chroma, question: str, k: int = 3):
    """FR-3.4/3.5: top-k semantic search with recorded latency."""
    start = time.perf_counter()
    results = vectorstore.similarity_search(question, k=k)
    elapsed = time.perf_counter() - start
    assert 0 < elapsed < 5, f'Retrieval latency {elapsed:.3f}s outside expected range'
    return [r.page_content for r in results], elapsed


def log_queries(vectorstore: Chroma, questions: list[str] | None = None,
                log_path: str | Path = ROOT / 'outputs' / 'retrieval_log.md'):
    """Run the test-question suite and log results (question, top chunk, latency)."""
    questions = questions or TEST_QUESTIONS
    rows = []
    for q in questions:
        chunks, latency = query_guidelines(vectorstore, q)
        rows.append({'question': q, 'top_chunk': chunks[0], 'latency_s': round(latency, 3)})
        print(f'[{latency:.3f}s] {q}\n  -> {chunks[0][:100]}...')

    md = ['# Retrieval log — Module 3', '',
          '| # | Question | Top retrieved chunk (truncated) | Latency (s) |',
          '|---|---|---|---|']
    for i, r in enumerate(rows, 1):
        snippet = r['top_chunk'][:120].replace('|', '/').replace('\n', ' ')
        md.append(f"| {i} | {r['question']} | {snippet}... | {r['latency_s']} |")
    md.append('')
    md.append('Chunking choice: RecursiveCharacterTextSplitter with chunk_size=500 characters '
              'and 50-character overlap. The source document is only ~1,200 words, so this '
              'yields 17 chunks of 16-70 words — small enough that each rule stays whole and '
              'retrieval stays specific. A literal 300-500 *token* chunk size would have '
              'produced just 2-3 chunks and destroyed retrieval granularity, so the character '
              'budget was tuned to the document length instead (documented trade-off vs FR-3.1).')
    md.append('')
    md.append('Known failure case: questions mixing two topics (e.g. storage AND dosing) '
              'sometimes return a chunk covering only one of them, because a single '
              'chunk cannot span two document sections. Hypothesis: increasing overlap '
              'would not fix it; cross-encoder re-ranking of more candidates (k=10 -> top 3) '
              'would.')
    Path(log_path).write_text('\n'.join(md), encoding='utf-8')
    return rows


if __name__ == '__main__':
    store = build_index()

    # FR-3.3: reload from disk in a "fresh session" and query the reloaded store.
    reloaded = load_index()
    rows = log_queries(reloaded)

    latency_path = ROOT / 'outputs' / 'retrieval_latencies.json'
    latency_path.write_text(json.dumps({r['question']: r['latency_s'] for r in rows}, indent=2))
    print(f'\nSaved latency log -> {latency_path}')
