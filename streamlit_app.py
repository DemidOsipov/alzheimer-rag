from __future__ import annotations

import hashlib
import os
import pathlib
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import tiktoken
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from gigachat import GigaChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer


APP_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_DATA_DIR = APP_ROOT / "articles"
DEFAULT_CITATIONS = APP_ROOT / "alzheimer_citations.xlsx"

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-hybrid-agentic-local-384")
PC_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PC_REGION = os.getenv("PINECONE_REGION", "us-east-1")

LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a friendly chatbot assistant for biomedical scientists. "
    "Provided context may help answering the question. "
    "Cite sources using the numeric labels shown in the context, like [1], [2]. "
    "Reuse the same number if multiple chunks are from the same source. "
    "Each factual sentence must include at least one citation. "
    "End with a 'Sources:' list where each line starts with the label, e.g., [1] <citation> (no extra numbering)."
)

QUERY_REWRITE_PROMPT = (
    "Given the conversation history and the current question, rewrite the question "
    "to be a standalone search query that incorporates relevant context from the conversation. "
    "If the question references previous answers or questions, expand it to include those details. "
    "Return only the rewritten query, nothing else."
)

HISTORY_SUMMARY_PROMPT = (
    "Summarize the conversation into a compact memory for retrieval. "
    "Keep only key entities, constraints, and decisions. "
    "Return a short bullet list without citations."
)


def _load_env() -> None:
    load_dotenv(str(APP_ROOT / ".env"))


def _require_env() -> None:
    for k in ("PINECONE_API_KEY", "GIGACHAT_API_KEY"):
        if not os.getenv(k):
            raise RuntimeError(f"Missing {k} in .env")


@st.cache_resource(show_spinner=False)
def _get_docling_converter() -> DocumentConverter:
    return DocumentConverter()


@st.cache_resource(show_spinner=False)
def _get_gigachat() -> GigaChat:
    return GigaChat(
        credentials=os.environ["GIGACHAT_API_KEY"],
        verify_ssl_certs=False,
        timeout=60,
    )


def _convert_missing_pdfs(data_dir: pathlib.Path, md_dir: pathlib.Path) -> None:
    pdfs = sorted(data_dir.glob("**/*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {data_dir}")
    md_dir.mkdir(parents=True, exist_ok=True)

    converter = _get_docling_converter()
    for pdf in pdfs:
        md_path = md_dir / f"{pdf.stem}.md"
        if md_path.exists():
            continue
        result = converter.convert(str(pdf))
        md = result.document.export_to_markdown()
        md_path.write_text(md, encoding="utf-8")


@st.cache_data(show_spinner=False)
def _load_citation_map(citations_xlsx: str) -> Dict[str, str]:
    path = pathlib.Path(citations_xlsx)
    if not path.exists():
        raise RuntimeError(f"Citations file not found: {citations_xlsx}")
    df = pd.read_excel(path)
    df = df.dropna(subset=["pdf file name", "citation"])
    return {
        str(row["pdf file name"]).strip().lower(): str(row["citation"]).strip()
        for _, row in df.iterrows()
    }


def _get_citation(pdf_filename: str, citation_map: Dict[str, str]) -> str:
    key = str(pdf_filename).strip().lower()
    citation = citation_map.get(key)
    if not citation:
        raise RuntimeError(f"Missing citation for PDF: {pdf_filename}")
    return citation


@st.cache_data(show_spinner=False)
def _load_docs(
    md_dir: str, data_dir: str, citations_xlsx: str, citations_mtime: float
) -> List[Document]:
    citation_map = _load_citation_map(citations_xlsx)
    docs: List[Document] = []
    for md_path in sorted(pathlib.Path(md_dir).glob("**/*.md")):
        text = md_path.read_text(encoding="utf-8")
        pdf_path = pathlib.Path(data_dir) / f"{md_path.stem}.pdf"
        source = str(pdf_path if pdf_path.exists() else md_path)
        pdf_name = pdf_path.name if pdf_path.exists() else md_path.name
        citation = _get_citation(pdf_name, citation_map)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "md": str(md_path),
                    "citation": citation,
                },
            )
        )
    if not docs:
        raise RuntimeError(f"No markdown files found in {md_dir}")
    return docs


@st.cache_data(show_spinner=False)
def _build_chunks(docs: List[Document]) -> List[Document]:
    encoding = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding.name,
        chunk_size=400,
        chunk_overlap=50,
        separators=[
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n- ",
            "\n* ",
            "\n1. ",
            "\n|",
            "\n",
            ". ",
            " ",
            "",
        ],
    )
    return splitter.split_documents(docs)


@st.cache_resource(show_spinner=False)
def _get_bm25(chunks: List[Document]) -> BM25Encoder:
    texts = [d.page_content for d in chunks]
    bm25 = BM25Encoder()
    bm25.fit(texts)
    return bm25


@st.cache_resource(show_spinner=False)
def _get_dense_encoder() -> SentenceTransformer:
    return SentenceTransformer(LOCAL_EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def _get_query_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def _get_index() -> Pinecone:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = {ix["name"]: ix for ix in pc.list_indexes()}
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=PC_CLOUD, region=PC_REGION),
        )
    return pc.Index(INDEX_NAME)


def _namespace_for_docs(docs: List[Document]) -> str:
    pdfs = sorted(
        {
            pathlib.Path(d.metadata.get("source", "")).name.lower()
            for d in docs
            if d.metadata.get("source")
        }
    )
    ns_hash = hashlib.sha1("|".join(pdfs).encode("utf-8")).hexdigest()[:8]
    return f"docs_{ns_hash}"


def _make_id(doc: Document) -> str:
    base = f"{doc.metadata.get('source','')}|{doc.metadata.get('md','')}|{doc.page_content}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def _upsert_missing(
    index, chunks: List[Document], bm25: BM25Encoder, namespace: str
) -> int:
    items: List[Tuple[str, Document]] = []
    for doc in chunks:
        citation = doc.metadata.get("citation")
        if not citation or str(citation).strip().lower() == "unknown":
            raise RuntimeError(f"Missing citation for source: {doc.metadata.get('source')}")
        doc_id = doc.metadata.get("id") or _make_id(doc)
        doc.metadata["id"] = doc_id
        items.append((doc_id, doc))

    ids = [i for i, _ in items]
    existing_ids = set()
    needs_refresh = set()
    for i in range(0, len(ids), 100):
        fetched = index.fetch(ids=ids[i : i + 100], namespace=namespace)
        for _id, vec in fetched.get("vectors", {}).items():
            existing_ids.add(_id)
            cit = (vec.get("metadata") or {}).get("citation")
            if not cit or str(cit).strip().lower() == "unknown":
                needs_refresh.add(_id)

    ids_to_upsert = set(ids) - (existing_ids - needs_refresh)
    missing_docs = [doc for _id, doc in items if _id in ids_to_upsert]
    if not missing_docs:
        return 0

    texts = [d.page_content for d in missing_docs]
    encoder = _get_dense_encoder()
    dense_vecs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        dense_vecs.extend(encoder.encode(batch, normalize_embeddings=True).tolist())

    sparse_vecs = []
    for i in range(0, len(texts), batch_size):
        sparse_vecs.extend(bm25.encode_documents(texts[i : i + batch_size]))

    to_upsert = []
    for doc, dense, sparse in zip(missing_docs, dense_vecs, sparse_vecs):
        metadata = {
            **doc.metadata,
            "citation": str(doc.metadata["citation"]).strip(),
            "context": doc.page_content,
        }
        to_upsert.append(
            {
                "id": doc.metadata["id"],
                "values": dense,
                "sparse_values": sparse,
                "metadata": metadata,
            }
        )

    for i in range(0, len(to_upsert), 100):
        index.upsert(vectors=to_upsert[i : i + 100], namespace=namespace)
    return len(to_upsert)


def _get_retriever(index, bm25: BM25Encoder, namespace: str) -> PineconeHybridSearchRetriever:
    return PineconeHybridSearchRetriever(
        embeddings=_get_query_embeddings(),
        sparse_encoder=bm25,
        index=index,
        namespace=namespace,
        alpha=0.5,
        top_k=TOP_K,
    )


def _format_context(docs: List[Document], max_chars: int = 3500) -> str:
    ctx = []
    source_to_label: Dict[str, str] = {}
    next_id = 1
    for d in docs:
        src = d.metadata.get("citation")
        if not src or str(src).strip().lower() == "unknown":
            raise RuntimeError(f"Missing citation in metadata for source: {d.metadata.get('source')}")
        if src not in source_to_label:
            source_to_label[src] = str(next_id)
            next_id += 1
        label = source_to_label[src]
        ctx.append(f"[{label}] {src}\n{d.page_content[:1000]}")
    s = "\n\n".join(ctx)
    return s[:max_chars]


def _rewrite_query(history: List[Tuple[str, str]], current_q: str, summary: str) -> str:
    if not history and not summary:
        return current_q

    gigachat = _get_gigachat()
    hist_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in history[-3:]])
    summary_block = f"Summary:\n{summary}\n\n" if summary else ""
    prompt = (
        f"{QUERY_REWRITE_PROMPT}\n\n"
        f"{summary_block}"
        f"Recent conversation:\n{hist_text}\n\n"
        f"Current question: {current_q}"
    )

    messages = [{"role": "user", "content": prompt}]
    resp = gigachat.chat({"model": "GigaChat", "messages": messages, "temperature": 0})
    rewritten = resp.choices[0].message.content.strip()
    return rewritten if rewritten else current_q


def _summarize_history(history: List[Tuple[str, str]], prev_summary: str) -> str:
    if not history:
        return ""
    gigachat = _get_gigachat()
    hist_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    prompt = f"{HISTORY_SUMMARY_PROMPT}\n\nPrevious summary:\n{prev_summary}\n\nConversation:\n{hist_text}"
    messages = [{"role": "user", "content": prompt}]
    resp = gigachat.chat({"model": "GigaChat", "messages": messages, "temperature": 0})
    return resp.choices[0].message.content.strip()


def _ask_gigachat(question: str, context: str) -> str:
    gigachat = _get_gigachat()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
    ]
    resp = gigachat.chat({"model": "GigaChat", "messages": messages, "temperature": 0})
    return resp.choices[0].message.content


def _init_session_state() -> None:
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "namespace" not in st.session_state:
        st.session_state.namespace = None
    if "history" not in st.session_state:
        st.session_state.history: List[Tuple[str, str]] = []
    if "history_summary" not in st.session_state:
        st.session_state.history_summary = ""


def main() -> None:
    st.set_page_config(page_title="RAG Assistant", layout="wide")
    st.title("RAG assistant")

    _load_env()
    try:
        _require_env()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    _init_session_state()

    st.sidebar.header("Data")
    data_dir = str(DEFAULT_DATA_DIR)
    citations_xlsx = str(DEFAULT_CITATIONS)

    if st.sidebar.button("Initialize/Reindex"):
        citations_path = pathlib.Path(citations_xlsx)
        if not citations_path.exists():
            st.sidebar.error(f"Citations file not found: {citations_xlsx}")
            return

        data_path = pathlib.Path(data_dir)
        md_dir = data_path.parent / "articles_md"
        citations_mtime = citations_path.stat().st_mtime

        with st.spinner("Preparing documents..."):
            _convert_missing_pdfs(data_path, md_dir)
            docs = _load_docs(str(md_dir), str(data_path), citations_xlsx, citations_mtime)
            chunks = _build_chunks(docs)
            bm25 = _get_bm25(chunks)

        with st.spinner("Indexing..."):
            index = _get_index()
            namespace = _namespace_for_docs(docs)
            _upsert_missing(index, chunks, bm25, namespace)
            retriever = _get_retriever(index, bm25, namespace)
            st.session_state.retriever = retriever
            st.session_state.namespace = namespace
            st.session_state.history = []
            st.session_state.history_summary = ""
            st.sidebar.success("Indexing complete!")

    if st.session_state.retriever is None:
        st.info("Click 'Initialize/Reindex' in the sidebar to start.")
        return

    question = st.text_area("Research question", height=120, key="question_input")
    col1, col2 = st.columns([1, 10])
    with col1:
        submit = st.button("Ask")
    with col2:
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.history_summary = ""
            st.rerun()

    if not submit:
        if st.session_state.history:
            st.subheader("Conversation history")
            for i, (q, a) in enumerate(st.session_state.history, 1):
                with st.expander(f"Q{i}: {q[:80]}..."):
                    st.write(f"**Q:** {q}")
                    st.write(f"**A:** {a}")
        return

    q = " ".join(question.strip().split())
    if not q:
        st.warning("Please enter a question.")
        return

    with st.spinner("Rewriting query..."):
        rewritten_q = _rewrite_query(
            st.session_state.history, q, st.session_state.history_summary
        )

    with st.spinner("Retrieving..."):
        retrieved = st.session_state.retriever.invoke(rewritten_q)
        context = _format_context(retrieved)

    with st.spinner("Generating answer..."):
        answer = _ask_gigachat(q, context)

    st.session_state.history.append((q, answer))
    if len(st.session_state.history) >= 4:
        st.session_state.history_summary = _summarize_history(
            st.session_state.history, st.session_state.history_summary
        )

    st.subheader("Answer")
    st.write(answer)

    if st.session_state.history:
        st.subheader("Conversation history")
        for i, (hist_q, hist_a) in enumerate(st.session_state.history, 1):
            with st.expander(f"Q{i}: {hist_q[:80]}..."):
                st.write(f"**Q:** {hist_q}")
                st.write(f"**A:** {hist_a}")


if __name__ == "__main__":
    main()
