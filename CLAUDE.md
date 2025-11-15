# Whisky RAG Chatbot Project

## 프로젝트 개요

위스키 리뷰 정보를 기반으로 한 RAG(Retrieval Augmented Generation) 챗봇 구현 프로젝트입니다.
히포위스키클럽의 고객 리뷰 데이터베이스를 활용하여, 사용자의 취향에 맞는 위스키를 추천하고 설명하는 AI 서비스를 개발합니다.

## 배경

- 위스키 시장 침체: WhiskyStats 지수 기준 약 35.8% 하락 (357.94 → 229.77)
- 오프라인 전문가 상담의 확장성 한계
- 1,000개 고품질 리뷰를 활용한 AI 기반 맞춤형 추천 서비스 필요

## 과제 요구사항

### Q1: 위스키 리뷰 데이터 벡터화 및 Vector Database 적재

**목표**: `whisky_reviews.csv` 데이터를 임베딩하여 Vector Database에 저장

**구현 단계**:
1. CSV 파일에서 데이터 로딩
2. 결측치 확인 및 필터링/보간
3. 임베딩 모델을 활용한 벡터화
4. Vector DB(Pinecone)에 업로드

**데이터 필드**:
- 위스키 이름
- 리뷰 링크
- 향(nose)
- 맛(flavor)
- 피니쉬(finish)

### Q2: RAG 기반 웹 챗봇 구현

**목표**: 사용자 질문에 대해 관련 위스키 정보를 검색하고 LLM으로 답변 생성

**구현 요소**:
- 사용자 질문 임베딩
- Vector DB에서 유사 문서 검색 (Retrieval)
- 검색된 정보를 컨텍스트로 LLM에 전달
- 최종 답변 생성 (Generation)
- UI 프레임워크 활용 (Gradio, Reflex, Streamlit 등)

### Q3: 챗봇 성능(정확도) 개선

**목표**: `whisky_qnas.csv`의 20개 FAQ를 기준으로 답변 품질 향상

**개선 방법** (자유롭게 적용):
- Prompt Engineering
- 임베딩 및 적재 방법 개선
- 검색 알고리즘 개선 (하이브리드 검색, 리랭킹 등)
- Retrieval 후 데이터 처리 과정 개선
- RAG 파이프라인 최적화

**평가**:
- 제공된 20개 FAQ 외에 약 80개의 추가 테스트 질문 사용 예정
- 성능 개선 정량화 시 가산점

**참고 자료**:
- [LangChain - Evaluating LLMs with OpenEvals](https://blog.langchain.dev/evaluating-llms-with-openevals/)
- [Confident AI - RAG Evaluation Guide](https://docs.confident-ai.com/guides/guides-rag-evaluation)
- [Qdrant - RAG Evaluation Guide](https://qdrant.tech/blog/rag-evaluation-guide/)
- [Pinecone - RAG Evaluation](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)

## 제출물

1. **리포트** (1-3페이지)
   - 프롬프트 설계
   - 생성 로직
   - 성능 개선 기법
   - 평가 방법 및 채택 사유

2. **Q1 코드**
   - 데이터 로딩 → 임베딩 → Vector DB 저장 전체 파이프라인

3. **Q2 코드**
   - RAG 기반 챗봇 구현
   - 예시 답변 시연 (스크린샷, 콘솔 로그)

4. **Q3 코드**
   - 성능 개선 적용 코드
   - 정량적 평가 지표 (가점 요소)

## 기술 스택 및 무료 API

### 선택된 구현 스택

이 프로젝트에서는 다음 기술 스택을 사용합니다:

- **Framework**: LangChain (RAG 파이프라인 구축)
- **Vector Database**: Pinecone (LangChain 통합)
- **LLM**: Google Gemini 1.5 Flash-8B
- **Embedding**: Google Gemini Embedding

### Vector Database
- [Pinecone Starter Plan](https://pinecone.io/) (무료)
- LangChain Pinecone Integration: `langchain-pinecone`

### LLM
- [Google Gemini 1.5 Flash-8B](https://ai.google.dev/gemini-api/docs/models/gemini)
- LangChain Google GenAI Integration: `langchain-google-genai`
- 무료 할당량 제공

### Embedding
- [Google Gemini Embedding](https://ai.google.dev/gemini-api/docs/embeddings?hl=ko)
- 모델: `models/text-embedding-004`
- LangChain 통합 가능

## 데이터 파일

- `whisky_reviews.csv`: 1,000개 위스키 리뷰 (Q1)
- `whisky_qnas.csv`: 20개 FAQ (Q3)

## 프로젝트 구조 (예상)

```
rag-assignment/
├── data/
│   ├── whisky_reviews.csv
│   └── whisky_qnas.csv
├── notebooks/
│   ├── 1.EDA.ipynb
│   ├── 2.vectorization.ipynb
│   └── 3.evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── embedding.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── chatbot.py
│   └── evaluation.py
├── app/
│   └── gradio_app.py (or streamlit_app.py)
├── tests/
├── requirements.txt
├── README.md
└── REPORT.md
```

## 개발 진행 순서

1. **EDA (Exploratory Data Analysis)**
   - `whisky_reviews.csv` 데이터 탐색
   - 결측치 패턴 분석
   - 데이터 품질 확인

2. **Q1: 벡터화 파이프라인 구축**
   - 데이터 전처리 및 필터링
   - 임베딩 모델 선택 및 벡터화
   - Pinecone 설정 및 업로드

3. **Q2: RAG 챗봇 구현**
   - Retrieval 로직 구현
   - LLM 통합
   - UI 개발 (Gradio/Streamlit)

4. **Q3: 성능 개선 및 평가**
   - Baseline 성능 측정
   - 개선 기법 적용
   - 정량적 평가 수행

5. **리포트 작성**
   - 구현 과정 정리
   - 결과 분석
   - 개선 사항 문서화

## 주의사항

- Q&A 내용을 직접 prompt에 하드코딩하지 말 것
- 언어/프레임워크 제약 없음
- AI 코딩 도구 사용 가능 (Copilot, ChatGPT 등)
- 모든 단계를 완벽히 구현할 필요 없음
- 문제 해결 과정과 사고방식이 중요

## 연락처

- 문의: people@vessl.ai
- API usage 부족 시 위 이메일로 연락

## 참고 링크

- [WhiskyStats Index Monitor](https://www.whiskystats.com/index-monitor/index/1)
