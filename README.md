# RAG Streamlit App

RAG app with dialogue support and citations.  
Uses Pinecone (hybrid retrieval) and GigaChat (generation).

## Repo structure

- `rag.ipynb` — main RAG notebook (setup, articles conversion, indexing, LLM QA chain, metrics)
- `helpers.py` — notebook helper functions (chunk analysis/preview)
- `streamlit_app.py` — Streamlit UI
- `README.md` — short usage instructions and ideas on extending RAG to other modalities
- `articles/` — input PDFs (example corpus)
- `articles_md/` — converted Markdown (auto-generated)
- `questions/` — evaluation question sets
- `alzheimer_citations.xlsx` — mapping [PDF filename → full citation]
- `.env_template` — API keys template
- `pyproject.toml` / `poetry.lock` — dependency definitions

## Setup

1) Install dependencies:

```
poetry install
```

2) Create `.env` with your keys:

```
cp .env_template .env
```

Get api keys for free at https://developers.sber.ru/ and https://app.pinecone.io and set:

- `PINECONE_API_KEY`
- `GIGACHAT_API_KEY`


3) The repo already includes example PDFs in `articles/` and the xlsx file that maps paper names to citations.
   You can use them as-is, or replace them with your own.

4) **Required**: Have a citations file (`alzheimer_citations.xlsx`) with columns:
   - `pdf file name` (must match PDF filenames exactly)
   - `citation` - full citation text
   
   The app and notebook require this file to work.

## Run rag.ipynb

Once setup, you can run (or just read through) all cells in notebook `rag.ipynb` to explore RAG implementation and metrics.
Conversion of pdf articles to markdown is already done - files are present in articles_md.
If you add new articles, they will be converted automatically in dedicated cell or at the start of the streamlit app, but it will take several minutes for each paper.


## Run streamlit app

```
poetry run streamlit run streamlit_app.py
```
App will open in your browser in a few seconds.

In the app:
- Click **Initialize/Reindex** - it will take several minutes to index when running app for the first time. Next time it will take seconds.
- Write a question and click **Ask**
- Use **Clear history** to reset the dialogue if needed

Long sessions are supported: the app keeps a compact summary of the dialogue
and uses it (plus new question) to rewrite retrieval queries, so dialogue context
persists.

## Notes

- Missing PDFs are converted to Markdown in a sibling `articles_md/` folder.
- A Pinecone index is created if it doesn’t exist.

## Other data modalities

### Which data modalities can this pipeline be extended to?

- Images (figures, charts from papers)
- Tables
- Graph data (knowledge graph: genes–targets–pathways–phenotypes)

### How can this be done?

From general principles to implementation ideas:

1) **Multimodal extraction**
   - **Images**: Docling for PDF parsing + PIL/OpenCV (reliable extraction + preprocessing).
   - **Tables**: Camelot or Tabula + pandas (robust table parsing and cleanup).
   - **Graphs**: spaCy/LLM for entity‑relation extraction + Neo4j for graph storage and queries.

2) **Unified representations**
   - **Images**: we can use CLIP embeddings.
   - **Tables**: serialize to text and embed with the same text model for consistency.
   - **Graphs**: Cypher queries in Neo4j.

3) **Indexing by modality**
   - Use separate Pinecone namespaces for each modality and store there extensive metadata.

4) **Multimodal retrieval**
   - Query Pinecone (text/tables), CLIP (images), and Neo4j (graphs), then fuse results.

5) **Answer synthesis**
   - Assemble context blocks by modality.
   - Indicate which modality supports each claim.

Extending to other modalities leads to some difficulties: we need to have a similarity metric between text query and other-modality chunks. So it requires a use of multimodal embeding model such as CLIP. Mapping images and text into a shared embedding space is not as accurate as embedding only one modality and thus can lead to a worse retrieval quality.


## Which models were used and why?

For retrieval, I used hybrid search with both dense and sparse scores.
The dense embeddings are generated using sentence-transformers/all-MiniLM-L6-v2 — a small and fast model that provides acceptable embedding quality.
The sparse score is computed using BM25, which is fast and does not require training. The score is calculated based on term frequency (TF), inverse document frequency (IDF), and chunk length, and it is considered a strong baseline model.

As the main LLM, I use the GigaChat Lite API to avoid downloading large weights while still achieving acceptable reasoning quality with low response latency. Having a more powerful LLM would definetely help with better answers. Another thing that can improve RAG pipeline is a use of knowledge graphs for better understanding of links between different entities such as drugs and their targets.