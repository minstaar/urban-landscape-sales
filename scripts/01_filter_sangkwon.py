# -*- coding: utf-8 -*-
"""
01_filter_sangkwon.py
─────────────────────────────────────────────────────────────────────────────
상권 필터링 스크립트

실행 방법:
    python scripts/01_filter_sangkwon.py

출력:
    data/processed/filtered_sangkwon_list.csv
    data/interim/filter_log.csv

제외 기준:
    Step 1. 유형 필터     - 발달상권 + 골목상권만 유지
    Step 2. 면적 필터     - 50,000m2 미만 제외 (상한 없음: 발달+골목 중 2km2 초과 없음)
    Step 3. 실내 복합시설 - 코엑스, 지하 백화점·쇼핑센터
    Step 4. 교통 허브     - 공항·KTX/SRT 전용 터미널
    Step 5. 산업단지/도매 - 디지털단지, 도매시장, 전문 산업단지
    Step 6. 공원·여가형   - 공원·유원지·등산로가 상권명에 포함
    Step 7. 대학 캠퍼스   - 캠퍼스 자체가 상권인 경우 ('앞' 제외)
    Step 8. 아파트 단지   - 단지 내 편의시설 상권
    Step 9. 단독 병원     - 병원 자체가 앵커인 상권
"""

import pandas as pd
from pathlib import Path

# 경로 설정
ROOT         = Path(__file__).resolve().parents[1]
RAW_BOUNDARY = ROOT / "data/raw/boundaries/서울시 상권분석서비스(영역-상권).csv"
OUT_LIST     = ROOT / "data/processed/filtered_sangkwon_list.csv"
OUT_LOG      = ROOT / "data/interim/filter_log.csv"

# Step 3: 실내 복합시설 코드
INDOOR_CODES = {
    3120218,  # 코엑스 (강남구)
    3120025,  # 롯데백화점(시청광장 지하쇼핑센터) (중구)
}

# Step 4: 교통 허브 코드
TRANSPORT_HUB_CODES = {
    3120115,  # 김포공항역(김포공항) (강서구)
    3120224,  # 수서역 (강남구) - SRT 전용
}

# Step 5: 산업단지·도매 키워드
INDUSTRIAL_KEYWORDS = [
    "디지털단지", "산업단지", "도매", "농수산물시장", "전자상가", "공단",
]
INDUSTRIAL_CODES = {
    3120235,  # 가산디지털단지 (금천구)
    3120129,  # 구로디지털단지 (구로구)
    3120130,  # 구로디지털단지역 (구로구)
    3120126,  # 디지털단지오거리 (구로구)
    3120117,  # 강서농산물도매시장 (강서구)
}

# Step 6: 공원·여가형
PARK_KEYWORDS           = ["공원", "유원지", "산입구", "등산", "수목원"]
PARK_EXCEPTION_KEYWORDS = ["앞", "역", "옆", "입구", "근처"]

# Step 7: 대학 캠퍼스
CAMPUS_KEYWORDS           = ["대학교", "대학원"]
CAMPUS_EXCEPTION_KEYWORDS = ["앞"]

# Step 8: 아파트 단지
APT_KEYWORDS = ["아파트", "주공"]

# Step 9: 단독 병원
HOSPITAL_KEYWORDS           = ["병원"]
HOSPITAL_EXCEPTION_KEYWORDS = ["역"]


def has_kw(name, keywords):
    return any(kw in str(name) for kw in keywords)

def has_exc(name, exceptions):
    return any(ex in str(name) for ex in exceptions)


def main():
    df = pd.read_csv(RAW_BOUNDARY, encoding="cp949")
    log = []

    def record(step, reason, excluded):
        for _, row in excluded.iterrows():
            log.append({
                "step": step,
                "reason": reason,
                "상권_코드": row["상권_코드"],
                "상권_코드_명": row["상권_코드_명"],
                "상권_구분_코드_명": row["상권_구분_코드_명"],
                "자치구_코드_명": row["자치구_코드_명"],
                "영역_면적": row["영역_면적"],
            })

    print("-" * 60)
    print(f"초기 상권 수: {len(df):,}개")
    print("-" * 60)

    # Step 1: 유형 필터
    mask = ~df["상권_구분_코드_명"].isin(["발달상권", "골목상권"])
    record("Step1", "유형제외(전통시장·관광특구)", df[mask])
    df = df[~mask].copy()
    print(f"Step 1  유형 필터 후 (발달+골목상권만):    {len(df):,}개")

    # Step 2: 면적 필터 (하한만)
    # 상한(2.0km2)은 발달+골목상권 중 해당 없어 불필요
    mask = df["영역_면적"] < 50_000
    record("Step2", "면적이상(50,000m2 미만)", df[mask])
    df = df[~mask].copy()
    print(f"Step 2  면적 필터 후 (0.05km2 이상):       {len(df):,}개")

    # Step 3: 실내 복합시설
    mask = df["상권_코드"].isin(INDOOR_CODES)
    record("Step3", "실내복합시설(코엑스·지하쇼핑센터)", df[mask])
    df = df[~mask].copy()
    print(f"Step 3  실내 복합시설 제외:                {len(df):,}개")

    # Step 4: 교통 허브
    mask = df["상권_코드"].isin(TRANSPORT_HUB_CODES)
    record("Step4", "교통허브(공항·KTX/SRT 전용)", df[mask])
    df = df[~mask].copy()
    print(f"Step 4  교통 허브 제외:                    {len(df):,}개")

    # Step 5: 산업단지·도매
    mask = (
        df["상권_코드_명"].apply(lambda x: has_kw(x, INDUSTRIAL_KEYWORDS)) |
        df["상권_코드"].isin(INDUSTRIAL_CODES)
    )
    record("Step5", "산업단지·도매(키워드+코드)", df[mask])
    df = df[~mask].copy()
    print(f"Step 5  산업단지·도매 제외:                {len(df):,}개")

    # Step 6: 공원·여가형
    mask = df["상권_코드_명"].apply(
        lambda x: has_kw(x, PARK_KEYWORDS) and not has_exc(x, PARK_EXCEPTION_KEYWORDS)
    )
    record("Step6", "공원·여가형(공원명이 상권명)", df[mask])
    df = df[~mask].copy()
    print(f"Step 6  공원·여가형 제외:                  {len(df):,}개")

    # Step 7: 대학 캠퍼스
    mask = df["상권_코드_명"].apply(
        lambda x: has_kw(x, CAMPUS_KEYWORDS) and not has_exc(x, CAMPUS_EXCEPTION_KEYWORDS)
    )
    record("Step7", "대학캠퍼스(캠퍼스 자체 상권)", df[mask])
    df = df[~mask].copy()
    print(f"Step 7  대학 캠퍼스 제외:                  {len(df):,}개")

    # Step 8: 아파트 단지
    mask = df["상권_코드_명"].apply(lambda x: has_kw(x, APT_KEYWORDS))
    record("Step8", "아파트단지(단지 내 편의시설)", df[mask])
    df = df[~mask].copy()
    print(f"Step 8  아파트 단지 제외:                  {len(df):,}개")

    # Step 9: 단독 병원
    mask = df["상권_코드_명"].apply(
        lambda x: has_kw(x, HOSPITAL_KEYWORDS) and not has_exc(x, HOSPITAL_EXCEPTION_KEYWORDS)
    )
    record("Step9", "단독병원(병원이 앵커인 상권)", df[mask])
    df = df[~mask].copy()
    print(f"Step 9  단독 병원 제외:                    {len(df):,}개")

    print("-" * 60)
    print(f"필터링 후 잔여 상권:  {len(df):,}개")
    print(f"  발달상권: {(df['상권_구분_코드_명']=='발달상권').sum():,}개")
    print(f"  골목상권: {(df['상권_구분_코드_명']=='골목상권').sum():,}개")
    print("-" * 60)

    OUT_LIST.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_LIST, index=False, encoding="utf-8-sig")
    pd.DataFrame(log).to_csv(OUT_LOG, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료:")
    print(f"  최종 상권 리스트 -> {OUT_LIST}")
    print(f"  제외 상세 로그   -> {OUT_LOG}")
    print(f"\n다음 단계: 매출 1억 미만 제외는 02_aggregate_sales.py 에서 처리됩니다.")


if __name__ == "__main__":
    main()
