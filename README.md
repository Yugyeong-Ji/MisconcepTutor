# MisconcepTutor

- MisconcepTutor: 학습자의 수학 개념 이해를 돕는 맞춤형 학습 도구
- 학생들의 오답을 분석하여 관련된 misconception을 파악하고, 이를 바탕으로 맞춤형 연습 문제를 제공

## Key Features 🌟

- **맞춤형 문제 제공**: 10개의 랜덤 문제를 통해 학습자의 이해도를 평가
- **오답 분석**: 틀린 문제에 대한 상세한 misconception 분석 제공
- **연습 문제 생성**: misconception을 기반으로 한 맞춤형 연습 문제 생성

### How to install

1. 저장소 클론
```bash
git clone https://github.com/Jintonic92/MisconcepTutor.git
cd MisconcepTutor
```

2. 가상 환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### How to run

```bash
streamlit run MisconceptTutor.py
```

## Project Structure 📁

```
MisconcepTutor/
├── Data/
│   ├── train.csv              # 학습 데이터
│   └── misconception_mapping.csv  # Misconception 매핑 데이터
├── MisconceptTutor.py         # 메인 애플리케이션
├── requirements.txt           # 필요한 패키지 목록
└── README.md                  # 프로젝트 문서
```

## Tech Stack 🛠️

- **Frontend**: Streamlit
- **Backend**: Python
- **ML/AI**: 
  - Sentence Transformers (misconception 분석)
  - LLaMA (문제 생성)
- **Data Preprocessing**: Pandas, NumPy


## Future Plan 🔮

1. Misconception 추론 능력 향상
2. 생성된 문제의 정확도 개선
3. UI/UX 개선
4. 다양한 과목 지원

## Reference 🙏

- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [LLaMA](https://ai.meta.com/llama/)
