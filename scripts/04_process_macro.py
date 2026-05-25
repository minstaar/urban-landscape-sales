# -*- coding: utf-8 -*-
"""
04_process_macro.py
─────────────────────────────────────────────────────
거시경제 변수 처리 스크립트

    - CPI        : 월별 총지수 → 분기 평균
    - 기준금리   : 월별 금리   → 분기 마지막 달 값

연구 기간: 2022Q2 ~ 2025Q4 (분기코드 20222 ~ 20254)

실행:
    python scripts/04_process_macro.py

출력:
    data/processed/macro_panel.csv
"""

import pandas as pd
from pathlib import Path
import glob

ROOT      = Path(__file__).resolve().parents[1]
MACRO_DIR = ROOT / "data/raw/macro"
OUT_PATH  = ROOT / "data/processed/macro_panel.csv"

STUDY_START = 20222
STUDY_END   = 20254


def find_file(keyword):
    matches = list(MACRO_DIR.glob(f"*{keyword}*"))
    if not matches:
        raise FileNotFoundError(f"'{keyword}' 포함 파일을 찾을 수 없습니다.")
    return matches[0]


def wide_to_long(df, value_name, date_col_start=2):
    """
    wide format(열=연월) → long format(행=연월)
    date_col_start: 날짜 컬럼이 시작하는 인덱스
    """
    date_cols = df.columns[date_col_start:]
    values    = df.iloc[0, date_col_start:].values

    long = pd.DataFrame({"연월_str": date_cols, value_name: values})
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long


def to_quarter_code(year, quarter):
    return int(f"{year}{quarter}")


def assign_quarter(dt):
    return dt.year * 10 + ((dt.month - 1) // 3 + 1)


def main():
    print("=" * 60)
    print("04_process_macro.py 시작")
    print("=" * 60)

    # ── 1. CPI ────────────────────────────────────────────────
    print("\n[1] CPI 처리 중...")
    cpi_file = find_file("소비자물가")
    df_cpi   = pd.read_csv(cpi_file, encoding="cp949")
    print(f"    파일: {cpi_file.name}")

    # 날짜 컬럼: 인덱스 2부터 (시도별, 품목성질별 제외)
    cpi_long = wide_to_long(df_cpi, "CPI", date_col_start=2)

    # 날짜 파싱 (2019.01 → datetime)
    cpi_long["연월"] = pd.to_datetime(
        cpi_long["연월_str"].str.replace(".", "-") + "-01", format="%Y-%m-%d"
    )
    cpi_long["분기코드"] = cpi_long["연월"].apply(assign_quarter)

    # 분기 평균
    cpi_q = (cpi_long
             .dropna(subset=["CPI"])
             .groupby("분기코드", as_index=False)["CPI"]
             .mean()
             .round(4))

    print(f"    분기 수: {len(cpi_q)}, 범위: {cpi_q['분기코드'].min()}~{cpi_q['분기코드'].max()}")

    # ── 2. 기준금리 ───────────────────────────────────────────
    print("\n[2] 기준금리 처리 중...")
    rate_file = find_file("기준금리")
    df_rate   = pd.read_csv(rate_file, encoding="utf-8-sig")
    print(f"    파일: {rate_file.name}")

    # 날짜 컬럼: 인덱스 4부터 (통계표, 계정항목, 단위, 변환 제외)
    rate_long = wide_to_long(df_rate, "기준금리", date_col_start=4)

    # 날짜 파싱 (2019/01 → datetime)
    rate_long["연월"] = pd.to_datetime(
        rate_long["연월_str"].str.replace("/", "-") + "-01", format="%Y-%m-%d"
    )
    rate_long["분기코드"] = rate_long["연월"].apply(assign_quarter)

    # 분기 마지막 달 값 (분기말 실효금리)
    rate_q = (rate_long
              .dropna(subset=["기준금리"])
              .sort_values("연월")
              .groupby("분기코드", as_index=False)["기준금리"]
              .last())

    print(f"    분기 수: {len(rate_q)}, 범위: {rate_q['분기코드'].min()}~{rate_q['분기코드'].max()}")

    # ── 3. 병합 및 연구 기간 필터 ────────────────────────────
    macro = cpi_q.merge(rate_q, on="분기코드", how="inner")
    macro = macro[
        (macro["분기코드"] >= STUDY_START) &
        (macro["분기코드"] <= STUDY_END)
    ].reset_index(drop=True)

    # 컬럼명 통일 (다른 패널과 동일한 키 사용)
    macro = macro.rename(columns={"분기코드": "기준_년분기_코드"})

    print(f"\n{'=' * 60}")
    print(f"최종 거시변수 패널")
    print(f"  분기 수: {len(macro)}개 ({macro['기준_년분기_코드'].min()}~{macro['기준_년분기_코드'].max()})")
    print(macro.to_string(index=False))
    print("=" * 60)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_PATH}")
    print("다음 단계: scripts/05_merge_panel.py")


if __name__ == "__main__":
    main()
