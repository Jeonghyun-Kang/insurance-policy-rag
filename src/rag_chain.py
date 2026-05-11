from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def get_rag_chain(vector_store, search_type="similarity"):
    # 환경 변수에 GOOGLE_API_KEY가 있으면 Gemini를, 없으면 Ollama를 쓰도록 설정
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if google_key:
        # 배포용 (Streamlit Cloud)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_key)
    else:
        # 로컬용 (Ollama)
        llm = ChatOllama(model="llama3.1", temperature=0)
    
    # 검색 방식 설정 (포트폴리오 개선 포인트: similarity vs mmr)
    retriever = vector_store.as_retriever(
        search_type=search_type, 
        search_kwargs={"k": 3}
    )

    # 보험 도메인 특화 프롬프트
    template = """당신은 보험 전문 상담 AI입니다. 
    반드시 제공된 약관 내용(Context)을 기반으로 답변하세요. 
    만약 내용이 없다면 "약관상 확인할 수 없습니다"라고 답하세요.
    
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True, # 근거 문서 반환 설정
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain