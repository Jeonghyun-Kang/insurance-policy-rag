import os
import tempfile
from typing import List, Tuple

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DB_DIR = "./chroma_db"


def load_pdf(uploaded_file):
    """
    Streamlit uploaded_file을 임시 PDF 파일로 저장한 뒤 PyMuPDFLoader로 로드
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()

    os.remove(tmp_path)
    return docs


def split_documents(docs):
    """
    보험 약관 문서를 chunk 단위로 분할
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


@staticmethod
def _unused():
    pass


def get_embeddings():
    """
    무료 로컬 HuggingFace embedding 사용
    한국어 보험 문서를 고려해 multilingual 모델 사용
    """
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks, persist_directory=DB_DIR):
    """
    ChromaDB 생성
    """
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="insurance_policy",
    )

    return vectorstore


def load_vectorstore(persist_directory=DB_DIR):
    """
    이미 생성된 ChromaDB 로드
    """
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="insurance_policy",
    )

    return vectorstore


def retrieve_documents(vectorstore, query: str, k: int = 5):
    """
    질문과 관련된 약관 chunk 검색
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(k * 2, 10),
            "lambda_mult": 0.5,
        },
    )
    docs = retriever.invoke(query)
    return docs


def format_context(docs):
    """
    검색된 문서를 LLM context로 변환
    """
    return "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ]
    )


def generate_rule_based_answer(query: str, docs):
    """
    OpenAI quota 없이도 데모 가능하도록 만든 기본 답변 생성 함수.
    LLM이 없을 때는 검색된 문서 기반으로 요약형 답변을 제공.
    """
    if not docs:
        return "관련 약관 조항을 찾지 못했습니다."

    top_doc = docs[0]
    source = top_doc.metadata.get("source", "unknown")
    page = top_doc.metadata.get("page", "unknown")

    answer = f"""
질문과 가장 관련성이 높은 약관 조항을 찾았습니다.

검색된 약관 내용에 따르면, 다음 조항이 질문과 관련됩니다.

{top_doc.page_content[:800]}

근거:
- Source: {source}
- Page: {page}
"""
    return answer.strip()