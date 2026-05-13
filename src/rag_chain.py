import os
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_rag_chain(vector_store, search_type="similarity"):
    # 1. 모델 설정 (Gemini OR Ollama)
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if google_key:
        # 배포용 (Streamlit Cloud)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_key)
    else:
        # 로컬용 (Ollama)
        llm = ChatOllama(model="llama3.1", temperature=0)
    
    # 2. 검색기 설정 (포트폴리오 개선 포인트: similarity vs mmr)
    retriever = vector_store.as_retriever(
        search_type=search_type, 
        search_kwargs={"k": 3}
    )

    # 3. 보험 도메인 특화 프롬프트
    system_prompt = (
        "당신은 보험 전문 상담 AI입니다. "
        "반드시 제공된 약관 내용(Context)을 기반으로 답변하세요. "
        "만약 내용이 없다면 '약관상 확인할 수 없습니다'라고 답하세요."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. 체인 생성(구 ㄱRetrievalQA대체)
    # 문서를 합치는 체인
    qa_chain = create_stuff_documents_chain(llm, prompt)
    # 최종 리트리버 체인
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True, # 근거 문서 반환 설정
    #     chain_type_kwargs={"prompt": prompt}
 