import streamlit as st
from dotenv import load_dotenv
from src.load_pdf import process_insurance_pdf
from src.build_vector_db import create_vector_db
# from src.rag_chain import get_rag_chain
import os

load_dotenv()

st.set_page_config(page_title="삼성화재 RAG 프로토타입", layout="wide")
st.title("🛡️ 보험 약관 근거 기반 Q&A 시스템")

# 1. 사이드바 - 파일 업로드 및 설정
with st.sidebar:
    st.header("1. 문서 업로드")
    uploaded_file = st.file_uploader("보험 약관 PDF를 업로드하세요", type="pdf")
    
    st.header("2. 검색 옵션")
    search_mode = st.selectbox("Retrieval 방식을 선택하세요", ["similarity", "mmr"])
    st.info("MMR은 중복된 정보가 많은 약관에서 더 다양한 조항을 검색합니다.")

# 2. RAG 엔진 초기화
if uploaded_file and "vector_store" not in st.session_state:
    with st.spinner("약관을 분석 중입니다..."):
        # 임시 파일 저장
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        splits = process_insurance_pdf("temp.pdf")
        st.session_state.vector_store = create_vector_db(splits)
        st.success("분석 완료!")

# 3. 채팅 인터페이스
if "vector_store" in st.session_state:
    
    if prompt := st.chat_input("질문을 입력해 주세요."):
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            # 1. Retrieval 방식 설정
            if search_mode == "mmr":
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 5,
                        "fetch_k": 10,
                        "lambda_mult": 0.5,
                    },
                )
            else:
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )

            # 2. 관련 약관 조항 검색
            source_docs = retriever.invoke(prompt)

            # 3. 검색 결과 기반 답변 생성
            if source_docs:
                top_doc = source_docs[0]
                page = top_doc.metadata.get("page", 0) + 1

                answer = f"""
**관련 조항 요약**

{top_doc.page_content[:10000]}
"""
            else:
                answer = "관련 약관 조항을 찾지 못했습니다."

            # 4. 답변 출력
            st.markdown(answer)
            
            # 5. 근거(Source) 표시
            with st.expander("📍 근거 확인하기"):
                if source_docs:
                    for i, doc in enumerate(source_docs, start=1):
                        page = doc.metadata.get("page", 0) + 1
                        source = doc.metadata.get("source", "unknown")
                        content = doc.page_content[:1000]

                        st.write(f"### 근거 {i}")
                        st.write(f"**Source:** {source}")
                        st.write(f"**Page:** {page}")
                        st.write(content)
                        st.divider()
                else:
                    st.write("검색된 근거 문서가 없습니다.")
else:
    st.warning("먼저 보험 약관 PDF를 업로드해 주세요.")

# # 3. 채팅 인터페이스
# if "vector_store" in st.session_state:
#     # chain = get_rag_chain(st.session_state.vector_store, search_mode)
    
#     if prompt := st.chat_input("질문을 입력해 주세요."):
#         st.chat_message("user").markdown(prompt)
        
#         with st.chat_message("assistant"):
#             # 1. 체인 실행 (결과를 result 변수에 담습니다)
#             result = chain.invoke({"input": prompt})

#             # 2. 결과에서 답변(answer)과 근거(context)를 꺼냅니다
#             answer = result["answer"]
#             source_docs = result["context"]
            
#             st.markdown(answer)
            
#             # 근거(Source) 표시
#             with st.expander("📍 근거 확인하기"):
#                 for doc in source_docs:
#                     page = doc.metadata.get("page", 0) + 1
#                     content = doc.page_content[:200]
#                     st.write(f"**Page {page}**: ...{content}...")
# else:
#     st.warning("먼저 보험 약관 PDF를 업로드해 주세요.")
# import os
# import shutil
# import streamlit as st

# from rag_pipeline import (
#     load_pdf,
#     split_documents,
#     build_vectorstore,
#     load_vectorstore,
#     retrieve_documents,
#     generate_rule_based_answer,
#     DB_DIR,
# )


# st.set_page_config(
#     page_title="Insurance Policy RAG Assistant",
#     page_icon="📄",
#     layout="wide",
# )


# st.title("📄 Insurance Policy RAG Assistant")
# st.caption("보험 약관 문서를 기반으로 관련 조항을 검색하고 근거 기반 답변을 제공합니다.")


# with st.sidebar:
#     st.header("Settings")

#     uploaded_file = st.file_uploader(
#         "보험 약관 PDF 업로드",
#         type=["pdf"],
#     )

#     k = st.slider(
#         "검색할 문서 chunk 개수",
#         min_value=1,
#         max_value=10,
#         value=5,
#     )

#     reset_db = st.button("🗑️ Chroma DB 초기화")

#     if reset_db:
#         if os.path.exists(DB_DIR):
#             shutil.rmtree(DB_DIR)
#             st.success("Chroma DB를 초기화했습니다.")
#         else:
#             st.info("삭제할 Chroma DB가 없습니다.")


# st.divider()


# if uploaded_file is not None:
#     st.subheader("1. PDF 문서 처리")

#     if st.button("📌 문서 인덱싱 시작"):
#         with st.spinner("PDF 로딩 중..."):
#             docs = load_pdf(uploaded_file)

#         st.success(f"PDF 로드 완료: {len(docs)} pages")

#         with st.spinner("문서 chunking 중..."):
#             chunks = split_documents(docs)

#         st.success(f"Chunk 생성 완료: {len(chunks)} chunks")

#         with st.spinner("Embedding 생성 및 ChromaDB 저장 중... 처음 실행 시 시간이 걸릴 수 있습니다."):
#             vectorstore = build_vectorstore(chunks)

#         st.session_state["vectorstore_ready"] = True
#         st.success("Vector DB 생성 완료!")

# else:
#     st.info("왼쪽 사이드바에서 보험 약관 PDF를 업로드하세요.")


# st.subheader("2. 보험 약관 질문하기")

# query = st.text_input(
#     "질문을 입력하세요",
#     placeholder="예: 보험금의 지급사유는 무엇인가요?",
# )

# if st.button("🔍 검색 및 답변 생성"):
#     if not query:
#         st.warning("질문을 입력해주세요.")
#     elif not os.path.exists(DB_DIR):
#         st.error("먼저 PDF를 업로드하고 문서 인덱싱을 완료해주세요.")
#     else:
#         with st.spinner("관련 약관 조항 검색 중..."):
#             vectorstore = load_vectorstore()
#             retrieved_docs = retrieve_documents(vectorstore, query, k=k)

#         st.subheader("3. 답변")

#         answer = generate_rule_based_answer(query, retrieved_docs)
#         st.markdown(answer)

#         st.subheader("4. 검색된 근거 문서")

#         for i, doc in enumerate(retrieved_docs, start=1):
#             source = doc.metadata.get("source", "unknown")
#             page = doc.metadata.get("page", "unknown")

#             with st.expander(f"근거 {i} | Page: {page}"):
#                 st.markdown(f"**Source:** `{source}`")
#                 st.markdown(f"**Page:** `{page}`")
#                 st.write(doc.page_content)


# st.divider()

# st.caption(
#     "Disclaimer: 본 프로젝트는 기술 데모용 프로토타입이며, 실제 보험금 지급 판단이나 법적 해석에는 사용할 수 없습니다."
# )