"""
슬기로운 논문생활 — API 백엔드
FastAPI + Anthropic Claude API
Render 배포용
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import anthropic

app = FastAPI(title="슬기로운 논문생활 API", version="1.0")

# CORS — GitHub Pages에서 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic 클라이언트
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ==================== 모델 ====================

class TopicRequest(BaseModel):
    topic: str
    field: Optional[str] = ""
    keywords: Optional[str] = ""
    purpose: Optional[str] = ""

class LitReviewRequest(BaseModel):
    topic: str
    field: Optional[str] = ""
    keywords: Optional[str] = ""
    scope: Optional[str] = "최근 5년"
    known_papers: Optional[str] = ""

class StructureRequest(BaseModel):
    topic: str
    field: Optional[str] = ""
    keywords: Optional[str] = ""
    paper_type: Optional[str] = "원저"
    methodology: Optional[str] = ""

class IntroRequest(BaseModel):
    topic: str
    field: Optional[str] = ""
    keywords: Optional[str] = ""
    language: Optional[str] = "한국어"

class AbstractRequest(BaseModel):
    topic: str
    field: Optional[str] = ""
    keywords: Optional[str] = ""
    word_count: Optional[int] = 250
    language: Optional[str] = "한국어"

class JournalRequest(BaseModel):
    topic: str
    field: Optional[str] = ""
    keywords: Optional[str] = ""
    index_type: Optional[str] = "SCI/SCIE"

class ReviewRequest(BaseModel):
    topic: str
    reviewer_comment: str
    language: Optional[str] = "한국어"


# ==================== N2B 시스템 프롬프트 ====================

SYSTEM_PROMPT = """당신은 N2B(Not-But-Because) 프레임워크 기반 논문 작성 전문 AI입니다.

## N2B 프레임워크란?
- Not: 현재 Best Practice(BP)의 빈틈, 한계, 미해결 문제
- But: 그럼에도 불구하고 가능한 새로운 접근, 기회
- Because: 그래서 이 연구가 필요한 이유, 근거

## 빅매치 메이커 철학
논문 작성자는 빅매치 메이커입니다:
1. 빅매치를 만들어라 (서론) — 현재 챔피언(BP)과 도전자(내 연구)의 대결 구도
2. 시합을 시켜라 (본론) — 공정한 조건에서 실제로 붙여봄
3. 결과를 발표하라 (결론) — 모두가 궁금해하는 그 결과

## 핵심 원칙
- 대립하는 이름 두 개가 붙어야 빅매치가 성립 (예: "사후탐지형" vs "사전예측형")
- 빈틈만 있으면 불만이고, 이름이 붙으면 연구 주제가 됨
- BP의 계보를 추적하면 연구의 맥락이 보임 (1세대→2세대→3세대→...)
- 항상 구체적이고 실제적인 내용으로 분석할 것
- 최신 연구 트렌드를 반영할 것

## 응답 형식
- 구조화된 텍스트로 응답 (마크다운 대신 텍스트 기호 사용)
- 이모지를 적절히 활용
- 한국어로 응답 (영어 요청 시 영어)
"""


# ==================== 엔드포인트 ====================

@app.get("/")
def root():
    return {"service": "슬기로운 논문생활 API", "status": "running", "version": "1.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/topic")
async def analyze_topic(req: TopicRequest):
    """Stage 0: 연구 주제 N2B 분석"""
    prompt = f"""다음 연구 주제를 N2B 프레임워크로 분석해주세요.

연구 주제: {req.topic}
분야: {req.field or '미지정'}
키워드: {req.keywords or '미지정'}
연구 목적: {req.purpose or '미지정'}

다음 구조로 분석해주세요:

1. 현재 Best Practice (BP) 3가지 — 이 분야에서 현재 가장 잘 되고 있는 것
2. N2B 구조 분석:
   - Not (현재 BP의 빈틈 5가지) — 구체적 문헌 인용 포함
   - But (그럼에도 불구하고 가능한 새 접근)
   - Because (그래서 이 연구가 필요한 이유, 4가지 방향 제시)
3. 논문화 가능성 (참신성/실현성/기여도/시의성 각 별점)
4. 추천 빅매치 구도 3가지 — 반드시 대립하는 이름 쌍으로 (예: "OO형" vs "XX형")
5. 다음 단계 안내

텍스트 기호(━, ❌, ⚡, ✅, 🏆, 🥊, ✦, 💡)를 활용하여 구조화해주세요."""

    return await call_claude(prompt)


@app.post("/api/literature")
async def literature_review(req: LitReviewRequest):
    """Stage 1: N2B 문헌리뷰 맵"""
    prompt = f"""다음 연구 주제에 대한 N2B 문헌리뷰 맵을 만들어주세요.

연구 주제: {req.topic}
분야: {req.field or '미지정'}
키워드: {req.keywords or '미지정'}
검색 범위: {req.scope}
연구자 지정 논문: {req.known_papers or '없음'}

다음 구조로 작성해주세요:

1. N2B 계보 (세대별 진화):
   - 1세대 (초기 접근): 성과 → Not(빈틈) → 대표문헌
   - 2세대 (방법론 발전): 성과 → Not(빈틈) → 대표문헌
   - 3세대 (현재 BP): 성과 → Not(빈틈) → 대표문헌
   - 4세대 (연구 기회): 가능성 → Not(미개척) → ⭐ 연구 기회!

2. 핵심 선행연구 분류 (분야별로 실제 저자명과 연도 포함)

3. 연구 갭 요약 — 핵심 빈틈 한 문장

각 세대 사이에 "↓ 빈틈이 동기가 되어..." 화살표로 연결해주세요."""

    return await call_claude(prompt)


@app.post("/api/structure")
async def paper_structure(req: StructureRequest):
    """Stage 2: 논문 구조 설계"""
    prompt = f"""다음 연구 주제에 대한 N2B 기반 논문 구조를 설계해주세요.

연구 주제: {req.topic}
분야: {req.field or '미지정'}
키워드: {req.keywords or '미지정'}
논문 유형: {req.paper_type}
방법론: {req.methodology or '미지정'}

다음 구조로 설계해주세요:

1. 논문 제목 (한국어 + 영어) — 3가지 후보
2. 전체 구조 (N2B 매핑):
   - 서론 (Not 영역): 배경 → 문제 제기 → 연구 목적
   - 이론적 배경/문헌리뷰: BP 계보
   - 연구 방법 (But 영역): 제안하는 방법론
   - 결과 및 분석: 빅매치 시합 결과
   - 고찰 (Because 영역): 의미와 기여
   - 결론
3. 각 장의 예상 분량 (페이지 수)
4. 핵심 Figure/Table 제안
5. 빅매치 구도 확인"""

    return await call_claude(prompt)


@app.post("/api/introduction")
async def write_introduction(req: IntroRequest):
    """Stage 3: 서론 작성"""
    prompt = f"""다음 연구 주제에 대한 N2B 기반 서론 초안을 작성해주세요.

연구 주제: {req.topic}
분야: {req.field or '미지정'}
키워드: {req.keywords or '미지정'}
언어: {req.language}

N2B 4단락 구조로 서론을 작성해주세요:

¶1-2 (배경 + 문제): 이 분야의 중요성과 현재 BP 소개
¶3 (Not — 빈틈): 기존 접근의 한계와 미해결 문제
¶4 (But/Because — 연구 목적): 본 연구의 접근 방식과 필요성

각 단락에 [N2B 구조 표시]를 포함하고, 참고문헌 위치를 (Author, Year) 형식으로 표시해주세요.
서론 뒤에 "N2B 흐름 분석"도 추가해주세요."""

    return await call_claude(prompt)


@app.post("/api/abstract")
async def generate_abstract(req: AbstractRequest):
    """Stage 4: 초록 생성"""
    prompt = f"""다음 연구 주제에 대한 N2B 기반 초록을 작성해주세요.

연구 주제: {req.topic}
분야: {req.field or '미지정'}
키워드: {req.keywords or '미지정'}
목표 단어 수: {req.word_count}단어
언어: {req.language}

N2B 초록 구조:
- 문장 1-2 (배경+문제): Not — 현재 상황의 빈틈
- 문장 3-5 (방법+결과): But — 본 연구의 접근과 주요 결과
- 문장 6-7 (의의): Because — 이 연구가 중요한 이유

초록 뒤에 추천 키워드 5개도 제시해주세요."""

    return await call_claude(prompt)


@app.post("/api/journal")
async def match_journal(req: JournalRequest):
    """Stage 5: 저널 매칭"""
    prompt = f"""다음 연구 주제에 적합한 학술지를 추천해주세요.

연구 주제: {req.topic}
분야: {req.field or '미지정'}
키워드: {req.keywords or '미지정'}
희망 인덱스: {req.index_type}

각 저널에 대해:
1. 저널명 (약칭)
2. 출판사
3. Impact Factor (최근)
4. 인덱스 (SCI/SCIE/SCOPUS/KCI)
5. 평균 심사 기간
6. 수락율 (추정)
7. 이 주제와의 적합도 (★ 표시)
8. 추천 이유

최소 5개 저널을 추천하되, 국제 저널과 국내 저널을 섞어주세요.
난이도 순서대로 (도전적 → 현실적 → 안전) 정렬해주세요."""

    return await call_claude(prompt)


@app.post("/api/review-response")
async def review_response(req: ReviewRequest):
    """Stage 6: 심사 대응"""
    prompt = f"""다음 심사 의견에 대한 N2B 기반 답변을 작성해주세요.

연구 주제: {req.topic}
심사 의견: {req.reviewer_comment}
언어: {req.language}

N2B 답변 구조:
1. Not (심사위원 지적 요약): 정확히 무엇을 지적했는가
2. But (수용/반박): 타당한 부분은 수용, 오해는 근거로 반박
3. Because (수정/보완 근거): 왜 이렇게 수정했는가 / 왜 원래가 맞는가

다음 형식으로 작성:
- Response to Reviewer: 답변 (공손하되 논리적으로)
- Action Taken: 수정 내용 (구체적으로)
- Revised Manuscript: 수정된 부분 표시

심사위원을 존중하면서도 연구의 가치를 지키는 균형 잡힌 답변을 작성해주세요."""

    return await call_claude(prompt)


# ==================== Claude API 호출 ====================

async def call_claude(prompt: str):
    """Claude API 호출 공통 함수"""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = message.content[0].text
        return {"status": "ok", "result": result}
    
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="API 키가 유효하지 않습니다")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="API 호출 한도 초과. 잠시 후 다시 시도해주세요")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 분석 중 오류: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
