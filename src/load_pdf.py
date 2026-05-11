from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import fitz  # PyMuPDF를 불러오는지 확인
except ImportError:
    raise ImportError("pymupdf 라이브러리가 설치되지 않았습니다. 'pip install pymupdf'를 실행하세요.")

def process_insurance_pdf(file_path):
    # 1. PDF 로드 (표/이미지 위치 보존에 유리한 PyMuPDF 사용)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # 2. 텍스트 분할 (보험 조항이 잘리지 않게 적절한 크기 설정)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    return splits
