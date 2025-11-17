# Whisky recommendation AI Service

RAG 기반 위스키 추천 서비스


### 사전 요구사항

- Python 3.12.12
- `uv` 패키지 매니저


### 1. 환경 설정

환경 세팅
```bash
# 개발 환경: pre-commit hooks 포함
make init-dev
# 프로덕션환경
make init
```

템플릿으로 환경 파일 생성 후, API키 입력
```bash
cp .env.example .env
# .env 파일에 필수 api 키 입력
```

가상 환경 활성화

```bash
source .venv/bin/activate
```

### 2. 데모 애플리케이션 실행

```bash
cd demo
streamlit run main.py
```

**기본 로그인 정보**:
- 사용자명: `admin`
- 비밀번호: `admin`

> ⚠️ **주의**: 개발용으로 하드코딩된 인증 정보입니다. 배포 전에 환경 변수로 변경해야 합니다.

### 3. 프로젝트 구조

```
whisky recommendation/
│
├── src/rag_pkg/              # 메인 패키지
│
├── configs/                  # YAML 설정 파일(프롬프트 템플릿 포함)
│
├── demo/                     # Streamlit 데모 애플리케이션
│
├── data/                     # 데이터 디렉토리 
│   ├── raw/                  # 원시 데이터 (whisky_reviews.csv, whisky_qnas.csv)
│   ├── intermediate/         # 중간 처리 결과
│   └── processed/            # 최종 처리된 데이터셋
│ 
├── notebooks/                # Jupyter 노트북
│
└── logs/                     # 로그 파일 (RAG 설정 저장장)
```

### 4. 주요 프레임워크 및 라이브러리

**LLM & GenAI** : LangChain, Google Generative AI

**데이터 처리, DB** : pandas, Pincone/ FAISS/ Chroma

**데모 화면** : streamlit

### 5. 작성자
Kwang Myung Yu (sguys99@gmail.com, sguys99@naver.com)
