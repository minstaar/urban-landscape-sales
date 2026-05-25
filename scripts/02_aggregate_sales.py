"""
02_aggregate_sales.py
─────────────────────────────────────────────────────────────────────────────
추정매출 집계 스크립트

    - 2021~2025년 연도별 zip 파일 읽기 (총 20분기)
    - 업종별 행을 상권별 분기별 전체 매출로 합산
    - 01_filter_sangkwon.py 결과로 필터링
    - 분기 평균 매출 1억 원 미만 상권 추가 제외
    - 최종 패널 구조(상권 × 분기) CSV 저장

실행 방법:
    python scripts/02_aggregate_sales.py

입력:
    data/raw/sales/서울시*추정매출*.zip (2021~2025)
    data/processed/filtered_sangkwon_list.csv

출력:
    data/processed/sales_panel.csv      최종 Y 패널 (상권 × 분기)
    data/processed/final_sangkwon_list.csv  매출 필터까지 적용된 최종 상권 리스트
"""

import zipfile
import pandas as pd
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
SALES_DIR  = ROOT / "data/raw/sales"
IN_LIST    = ROOT / "data/processed/filtered_sangkwon_list.csv"
OUT_PANEL  = ROOT / "data/processed/sales_panel.csv"
OUT_LIST   = ROOT / "data/processed/final_sangkwon_list.csv"

# 분기 평균 매출 최소 기준 (원)
MIN_AVG_QUARTERLY_SALES = 100_000_000  # 1억

# ── 연도별 zip 파일 목록 ───────────────────────────────────────────────────
SALES_ZIPS = {
    2021: "서울시_상권분석서비스(추정매출-상권)_2021년.zip",
    2022: "서울시_상권분석서비스(추정매출-상권)_2022년.zip",
    2023: "서울시_상권분석서비스(추정매출-상권)_2023년.zip",
    2024: "서울시 상권분석서비스(추정매출-상권)_2024년.zip",
    2025: "서울시 상권분석서비스(추정매출-상권)_2025년.zip",
}

# 집계에 사용할 컬럼만 읽기 (메모리 절약)
USE_COLS = [
    "기준_년분기_코드",
    "상권_코드",
    "상권_코드_명",
    "상권_구분_코드_명",
    "당월_매출_금액",
]


def load_sales_zip(zip_path: Path) -> pd.DataFrame:
    """zip 내 CSV를 읽어 필요 컬럼만 반환"""
    with zipfile.ZipFile(zip_path) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(
            z.open(csv_name),
            encoding="cp949",
            usecols=USE_COLS,
        )
    return df


def main():
    print("=" * 60)
    print("02_aggregate_sales.py 시작")
    print("=" * 60)

    # ── 1. 필터링된 상권 리스트 로드 ─────────────────────────────────────
    sangkwon_df = pd.read_csv(IN_LIST)
    valid_codes = set(sangkwon_df["상권_코드"].astype(int))
    print(f"\n[1] 입력 상권 수 (01_filter 결과): {len(valid_codes):,}개")

    # ── 2. 연도별 매출 데이터 로드 및 합치기 ──────────────────────────────
    print("\n[2] 연도별 매출 데이터 로드 중...")
    frames = []
    for year, fname in SALES_ZIPS.items():
        zip_path = SALES_DIR / fname
        if not zip_path.exists():
            print(f"  ⚠️  {year}년 파일 없음: {fname}")
            continue
        df = load_sales_zip(zip_path)
        frames.append(df)
        quarters = sorted(df["기준_년분기_코드"].unique())
        print(f"  {year}년: {len(df):,}행, 분기={quarters}")

    raw = pd.concat(frames, ignore_index=True)
    print(f"\n  합산 행 수: {len(raw):,}행 ({raw['기준_년분기_코드'].nunique()}분기)")

    # ── 3. 상권 필터 적용 ─────────────────────────────────────────────────
    raw = raw[raw["상권_코드"].isin(valid_codes)].copy()
    print(f"\n[3] 상권 필터 적용 후: {len(raw):,}행")

    # ── 4. 업종별 → 상권별 분기별 합산 (Y = 전체 추정매출) ───────────────
    panel = (
        raw.groupby(
            ["기준_년분기_코드", "상권_코드", "상권_코드_명", "상권_구분_코드_명"],
            as_index=False,
        )["당월_매출_금액"]
        .sum()
        .rename(columns={"당월_매출_금액": "추정매출_합계"})
    )
    print(f"\n[4] 업종 합산 완료: {len(panel):,}행")
    print(f"    상권 수: {panel['상권_코드'].nunique():,}개")
    print(f"    분기 수: {panel['기준_년분기_코드'].nunique()}개")

    # ── 5. 분기 수 체크 — 15분기 미만 상권 제외 ──────────────────────────
    quarter_counts = panel.groupby("상권_코드")["기준_년분기_코드"].count()
    low_quarter    = quarter_counts[quarter_counts < 15].index
    panel = panel[~panel["상권_코드"].isin(low_quarter)].copy()
    print(f"\n[5] 분기 15개 미만 제외 후: {panel['상권_코드'].nunique():,}개 상권")

    # ── 6. 분기 평균 매출 1억 미만 상권 제외 ─────────────────────────────
    avg_sales    = panel.groupby("상권_코드")["추정매출_합계"].mean()
    low_sales    = avg_sales[avg_sales < MIN_AVG_QUARTERLY_SALES].index
    panel = panel[~panel["상권_코드"].isin(low_sales)].copy()
    print(f"\n[6] 분기 평균 매출 1억 미만 제외 후: {panel['상권_코드'].nunique():,}개 상권")

    # ── 7. 정렬 및 최종 통계 ─────────────────────────────────────────────
    panel = panel.sort_values(["상권_코드", "기준_년분기_코드"]).reset_index(drop=True)

    final_codes = panel["상권_코드"].unique()
    n_bal = panel[panel["상권_구분_코드_명"] == "발달상권"]["상권_코드"].nunique()
    n_gol = panel[panel["상권_구분_코드_명"] == "골목상권"]["상권_코드"].nunique()

    print(f"\n{'=' * 60}")
    print(f"최종 패널 현황")
    print(f"  상권 수 : {len(final_codes):,}개  (발달 {n_bal}개 + 골목 {n_gol}개)")
    print(f"  분기 수 : {panel['기준_년분기_코드'].nunique()}개")
    print(f"  총 행 수: {len(panel):,}행")
    print(f"  Y 범위  : {panel['추정매출_합계'].min():,.0f}원"
          f" ~ {panel['추정매출_합계'].max():,.0f}원")
    print(f"  Y 중앙값: {panel['추정매출_합계'].median():,.0f}원")
    print(f"{'=' * 60}")

    # ── 8. 저장 ──────────────────────────────────────────────────────────
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PANEL, index=False, encoding="utf-8-sig")

    # 최종 상권 리스트 (다음 스크립트에서 이미지 수집에 사용)
    final_df = sangkwon_df[sangkwon_df["상권_코드"].isin(final_codes)].copy()
    final_df.to_csv(OUT_LIST, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료:")
    print(f"  Y 패널 데이터    → {OUT_PANEL}")
    print(f"  최종 상권 리스트 → {OUT_LIST}")
    print(f"\n※ 다음 단계: scripts/03_collect_images.py 로 이미지 수집")


if __name__ == "__main__":
    main()
