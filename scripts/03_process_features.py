# -*- coding: utf-8 -*-
"""
03_process_features.py
─────────────────────────────────────────────────────
시변 구조화 변수 처리 스크립트

처리 변수:
    - 유동인구 (총_유동인구_수)
    - 직장인구 (총_직장_인구_수)
    - 상주인구 (총_상주인구_수)
    - 점포_수  (업종 합산)
    - 개업률   (개업_점포_수 합 / 점포_수 합)
    - 폐업률   (폐업_점포_수 합 / 점포_수 합)

연구 기간: 2022Q2 ~ 2025Q4 (분기코드 20222 ~ 20254)

실행:
    python scripts/03_process_features.py

출력:
    data/processed/features_panel.csv
"""

import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
POP_DIR   = ROOT / "data/raw/population"
STORE_DIR = ROOT / "data/raw/store"
LIST_PATH = ROOT / "data/processed/final_sangkwon_list.csv"
OUT_PATH  = ROOT / "data/processed/features_panel.csv"

# 연구 기간 분기코드
STUDY_START = 20222
STUDY_END   = 20254

# 점포 연도별 파일 목록 (2022Q2부터 필요하므로 2022~2025)
STORE_FILES = [
    "서울시_상권분석서비스(점포-상권)_2022년.csv",
    "서울시_상권분석서비스(점포-상권)_2023년.csv",
    "서울시 상권분석서비스(점포-상권)_2024년.csv",
    "서울시 상권분석서비스(점포-상권)_2025년.csv",
]


# 2025년 점포 파일 영문 컬럼 → 한글 컬럼 매핑
STORE_ENG_TO_KOR = {
    "stdr_yyqu_cd" : "기준_년분기_코드",
    "trdar_cd"     : "상권_코드",
    "stor_co"      : "점포_수",
    "opbiz_stor_co": "개업_점포_수",
    "clsbiz_stor_co": "폐업_점포_수",
}

STORE_USE_COLS_KOR = ["기준_년분기_코드", "상권_코드", "점포_수", "개업_점포_수", "폐업_점포_수"]
STORE_USE_COLS_ENG = ["stdr_yyqu_cd", "trdar_cd", "stor_co", "opbiz_stor_co", "clsbiz_stor_co"]


def load_population(fname, col_map):
    path = POP_DIR / fname
    df = pd.read_csv(path, encoding="cp949",
                     usecols=["기준_년분기_코드", "상권_코드"] + list(col_map.keys()))
    df = df.rename(columns=col_map)
    return df


def load_store(fpath: Path, valid_codes: set, study_start: int, study_end: int) -> pd.DataFrame:
    """점포 CSV 로드 — 2025년 영문 컬럼 자동 처리 + 조기 필터링으로 속도 개선"""
    header = pd.read_csv(fpath, encoding="cp949", nrows=0)
    is_eng = "stdr_yyqu_cd" in header.columns

    use_cols  = STORE_USE_COLS_ENG if is_eng else STORE_USE_COLS_KOR
    q_col     = "stdr_yyqu_cd"     if is_eng else "기준_년분기_코드"
    code_col  = "trdar_cd"         if is_eng else "상권_코드"

    chunks = []
    for chunk in pd.read_csv(fpath, encoding="cp949", usecols=use_cols, chunksize=50_000):
        chunk = chunk[
            (chunk[q_col] >= study_start) &
            (chunk[q_col] <= study_end) &
            (chunk[code_col].isin(valid_codes))
        ]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    if is_eng:
        df = df.rename(columns=STORE_ENG_TO_KOR)
    return df


def main():
    print("=" * 60)
    print("03_process_features.py 시작")
    print("=" * 60)

    # 최종 상권 리스트
    valid_codes = set(pd.read_csv(LIST_PATH)["상권_코드"].astype(int))
    print(f"\n대상 상권: {len(valid_codes):,}개")

    # ── 1. 유동인구 ──────────────────────────────────────────
    print("\n[1] 유동인구 로드 중...")
    pop_flow = load_population(
        "서울시 상권분석서비스(길단위인구-상권).csv",
        {"총_유동인구_수": "유동인구"}
    )
    print(f"    전체: {len(pop_flow):,}행, 분기 {pop_flow['기준_년분기_코드'].nunique()}개")

    # ── 2. 직장인구 ──────────────────────────────────────────
    print("[2] 직장인구 로드 중...")
    pop_work = load_population(
        "서울시 상권분석서비스(직장인구-상권).csv",
        {"총_직장_인구_수": "직장인구"}
    )
    print(f"    전체: {len(pop_work):,}행")

    # ── 3. 상주인구 ──────────────────────────────────────────
    print("[3] 상주인구 로드 중...")
    pop_resi = load_population(
        "서울시 상권분석서비스(상주인구-상권).csv",
        {"총_상주인구_수": "상주인구"}
    )
    print(f"    전체: {len(pop_resi):,}행")

    # ── 4. 인구 세 변수 병합 ─────────────────────────────────
    key = ["기준_년분기_코드", "상권_코드"]
    pop = (pop_flow
           .merge(pop_work, on=key, how="outer")
           .merge(pop_resi,  on=key, how="outer"))

    # ── 5. 점포수 / 개업률 / 폐업률 ─────────────────────────
    print("[4] 점포 데이터 로드 및 업종 합산 중...")
    frames = []
    for fname in STORE_FILES:
        fpath = STORE_DIR / fname
        if not fpath.exists():
            print(f"    ⚠ 파일 없음: {fname}")
            continue
        df = load_store(fpath, valid_codes, STUDY_START, STUDY_END)
        frames.append(df)
        qs = sorted(df["기준_년분기_코드"].unique())
        print(f"    {fname[-9:-4]}: {qs}")

    store_raw = pd.concat(frames, ignore_index=True)

    # 업종별 → 상권별 분기별 합산
    store = (store_raw
             .groupby(["기준_년분기_코드", "상권_코드"], as_index=False)
             .agg(점포_수=("점포_수", "sum"),
                  개업_점포_수=("개업_점포_수", "sum"),
                  폐업_점포_수=("폐업_점포_수", "sum")))

    # 개업률 / 폐업률 재계산
    store["개업률"] = (store["개업_점포_수"] / store["점포_수"].replace(0, float("nan")) * 100).round(4)
    store["폐업률"] = (store["폐업_점포_수"] / store["점포_수"].replace(0, float("nan")) * 100).round(4)
    store = store.drop(columns=["개업_점포_수", "폐업_점포_수"])

    # ── 6. 전체 병합 ─────────────────────────────────────────
    panel = pop.merge(store, on=key, how="outer")

    # ── 7. 연구 기간 및 상권 필터 ────────────────────────────
    panel = panel[
        (panel["기준_년분기_코드"] >= STUDY_START) &
        (panel["기준_년분기_코드"] <= STUDY_END) &
        (panel["상권_코드"].isin(valid_codes))
    ].copy()

    panel = panel.sort_values(key).reset_index(drop=True)

    # ── 8. 결과 요약 ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"최종 패널 현황")
    print(f"  행 수  : {len(panel):,}")
    print(f"  상권 수: {panel['상권_코드'].nunique():,}개")
    print(f"  분기 수: {panel['기준_년분기_코드'].nunique()}개")
    print(f"\n결측값 현황:")
    for col in ["유동인구", "직장인구", "상주인구", "점포_수", "개업률", "폐업률"]:
        n_miss = panel[col].isna().sum()
        pct    = n_miss / len(panel) * 100
        print(f"  {col:12s}: {n_miss:,}개 ({pct:.1f}%)")
    print("=" * 60)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_PATH}")
    print("다음 단계: scripts/04_process_macro.py")


if __name__ == "__main__":
    main()
