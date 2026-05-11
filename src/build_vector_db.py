from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_db(splits):
    # model_kwargs를 추가하여 CPU 사용을 강제합니다.
    model_kwargs = {'device': 'cpu'}
    
    # 한국어 처리에 최적화된 로컬 임베딩 모델 사용 (OpenAI 비용 절감 및 보안)
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )
    
    # Chroma DB 생성 및 저장
    vector_store = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    return vector_store