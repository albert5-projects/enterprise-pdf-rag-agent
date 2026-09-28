import streamlit as st
import os
import re
import hashlib
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever   # pip install rank_bm25
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Enterprise RAG System", page_icon="📄")
st.title("📄 Enterprise PDF RAG Agent")
st.write("Upload a PDF, and the AI will answer questions based strictly on its text.")


# ---------- Cached resources (loaded ONCE, not on every rerun) ----------
@st.cache_resource
def get_embeddings():
    # Stronger than all-MiniLM-L6-v2. Alternatives: "BAAI/bge-small-en-v1.5" (faster)
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_llm():
    return ChatGroq(api_key=groq_api_key, model_name="openai/gpt-oss-120b", temperature=0)


# Footer/header boilerplate that repeats on every slide. Adjust to your deck.
BOILERPLATE = re.compile(r"Group\s*\d+\s*First Review\s*\d+\s*/\s*\d+", re.IGNORECASE)


def clean(text: str) -> str:
    text = BOILERPLATE.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


@st.cache_resource(show_spinner="Indexing document...")
def build_index(file_bytes: bytes, file_hash: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    docs = PyPDFLoader(path).load()
    for d in docs:
        d.page_content = clean(d.page_content)
    docs = [d for d in docs if d.page_content]  # drop empty pages

    # Slides have little text per page -> smaller chunks + page label in the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    for s in splits:
        s.page_content = f"[Slide/Page {s.metadata.get('page', 0) + 1}] {s.page_content}"

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        collection_name=f"doc_{file_hash}",  # unique per file -> no duplicates
    )
    vector_ret = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30}
    )
    bm25 = BM25Retriever.from_documents(splits)
    bm25.k = 8

    # Hybrid: semantic + keyword search
    hybrid = EnsembleRetriever(retrievers=[vector_ret, bm25], weights=[0.6, 0.4])
    return hybrid, docs


def expand_query(question: str) -> list[str]:
    """Let the LLM rephrase the question so wording differences don't matter."""
    prompt = (
        "Generate 3 alternative search queries for finding the answer to the question "
        "below in a project presentation/report. Use different vocabulary and synonyms "
        "(e.g. 'team members' -> 'submitted by, guided by, students, authors, developers'). "
        "Return one query per line, no numbering.\n\nQuestion: " + question
    )
    out = get_llm().invoke(prompt).content
    variants = [l.strip("-• ").strip() for l in out.split("\n") if l.strip()]
    return [question] + variants[:3]


def retrieve(hybrid, question: str, limit: int = 10):
    seen, results = set(), []
    for q in expand_query(question):
        for doc in hybrid.invoke(q):
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                results.append(doc)
    return results[:limit]


system_prompt = (
    "You are an intelligent enterprise assistant. Answer the user's question using the "
    "retrieved context below. The question may be worded differently from the document, "
    "so match by meaning (e.g. 'team members' = names listed as authors, submitted by, "
    "group members, etc.). Combine information across chunks if needed. "
    "Only if the context truly contains nothing relevant, say "
    "'I cannot answer this based on the provided document.' Do not hallucinate."
    "\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# ---------- UI ----------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    hybrid, docs = build_index(file_bytes, file_hash)
    st.success(f"Document ready ({len(docs)} pages with text).")

    with st.expander("🛠 Debug: text extracted from page 1"):
        st.write(docs[0].page_content if docs else "No text extracted (may need OCR)")

    st.markdown("---")
    st.subheader("💬 Ask the Document")
    user_question = st.text_input("Enter your question:")

    if user_question:
        with st.spinner("Analyzing document..."):
            context_docs = retrieve(hybrid, user_question)
            chain = create_stuff_documents_chain(get_llm(), prompt)
            answer = chain.invoke({"input": user_question, "context": context_docs})

            st.write("### 🤖 Answer:")
            st.info(answer)

            with st.expander("🔍 View Source Chunks"):
                for i, doc in enumerate(context_docs):
                    st.write(f"**Chunk {i+1} (Page {doc.metadata.get('page', 0) + 1}):**")
                    st.write(doc.page_content)
                    st.markdown("---")