# -*- coding: utf-8 -*-
"""
05_merge_panel.py
─────────────────────────────────────────────────────
전체 패널 병합 스크립트

    Y   : 추정매출_합계   (sales_panel.csv)
    X1  : 유동인구, 직장인구, 상주인구, 점포_수, 개업률, 폐업률 (features_panel.csv)
    X2  : CPI, 기준금리  (macro_panel.csv)
    Meta: 상권_유형_더미, 상권_면적, 자치구_코드, 분기_더미 (final_sangkwon_list.csv)

실행:
    python scripts/05_merge_panel.py

출력:
    data/processed/panel_final.csv   완성 패널 (모델 입력용)
    data/interim/merge_report.txt    결측값 보고서
"""

import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
PROC      = ROOT / "data/processed"
INTERIM   = ROOT / "data/interim"
OUT_PATH  = PROC  / "panel_final.csv"
REPORT    = INTERIM / "merge_report.txt"

STUDY_START = 20222
STUDY_END   = 20254


def main():
    print("=" * 60)
    print("05_merge_panel.py 시작")
    print("=" * 60)

    # ── 1. 각 파일 로드 ────────────────────────────────────────
    print("\n[1] 데이터 로드 중...")

    sales = pd.read_csv(PROC / "sales_panel.csv")
    sales = sales[["기준_년분기_코드", "상권_코드", "추정매출_합계"]]
    print(f"    Y (추정매출)  : {sales.shape}")

    features = pd.read_csv(PROC / "features_panel.csv")
    print(f"    X 시변 변수   : {features.shape}")

    macro = pd.read_csv(PROC / "macro_panel.csv")
    # 첫 컬럼을 기준_년분기_코드로 강제 통일 (BOM/명칭 차이 무관)
    macro.columns = ["기준_년분기_코드", "CPI", "기준금리"]
    print(f"    X 거시 변수   : {macro.shape}")

    sangkwon = pd.read_csv(PROC / "final_sangkwon_list.csv")
    sangkwon = sangkwon[[
        "상권_코드", "상권_코드_명", "상권_구분_코드_명",
        "자치구_코드", "자치구_코드_명", "영역_면적"
    ]].drop_duplicates()
    print(f"    상권 메타정보 : {sangkwon.shape}")

    # ── 2. 연구 기간 필터 ──────────────────────────────────────
    key = ["기준_년분기_코드", "상권_코드"]
    sales = sales[
        (sales["기준_년분기_코드"] >= STUDY_START) &
        (sales["기준_년분기_코드"] <= STUDY_END)
    ]
    print(f"\n[2] 연구 기간 필터 후 Y: {len(sales):,}행")

    # ── 3. Y × 시변 X 병합 ───────────────────────────────────
    panel = sales.merge(features, on=key, how="left")
    print(f"[3] Y + 시변 X 병합: {len(panel):,}행")

    # ── 4. 거시 X 병합 (분기코드 기준) ───────────────────────
    panel = panel.merge(macro, on="기준_년분기_코드", how="left")
    print(f"[4] + 거시 X 병합: {len(panel):,}행")

    # ── 5. 상권 메타정보 병합 ─────────────────────────────────
    panel = panel.merge(sangkwon, on="상권_코드", how="left")
    print(f"[5] + 상권 메타 병합: {len(panel):,}행")

    # ── 6. 파생 변수 추가 ─────────────────────────────────────
    # 상권 유형 더미 (발달상권=1, 골목상권=0)
    panel["상권_유형_더미"] = (panel["상권_구분_코드_명"] == "발달상권").astype(int)

    # 분기 더미 (Q1=1, Q2=2, Q3=3, Q4=4)
    panel["분기"] = panel["기준_년분기_코드"] % 10

    # 면적 단위: m² → km²
    panel["면적_km2"] = (panel["영역_면적"] / 1_000_000).round(6)

    # 컬럼 순서 정리
    col_order = [
        "기준_년분기_코드", "분기",
        "상권_코드", "상권_코드_명",
        "상권_구분_코드_명", "상권_유형_더미",
        "자치구_코드", "자치구_코드_명",
        "면적_km2",
        # Y
        "추정매출_합계",
        # 시변 X
        "유동인구", "직장인구", "상주인구",
        "점포_수", "개업률", "폐업률",
        # 거시 X
        "CPI", "기준금리",
    ]
    panel = panel[col_order].sort_values(key).reset_index(drop=True)

    # ── 7. 결측값 보고서 ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("결측값 현황:")
    report_lines = ["결측값 보고서\n" + "=" * 40]
    x_cols = ["유동인구", "직장인구", "상주인구",
              "점포_수", "개업률", "폐업률", "CPI", "기준금리"]
    for col in x_cols:
        n    = panel[col].isna().sum()
        pct  = n / len(panel) * 100
        line = f"  {col:12s}: {n:4,}개 ({pct:5.1f}%)"
        print(line)
        report_lines.append(line)

    total_rows = len(panel)
    complete   = panel[x_cols + ["추정매출_합계"]].dropna().shape[0]
    print(f"\n  완전 관측치: {complete:,} / {total_rows:,} ({complete/total_rows*100:.1f}%)")
    report_lines.append(f"\n완전 관측치: {complete:,} / {total_rows:,}")

    print(f"\n최종 패널:")
    print(f"  상권 수 : {panel['상권_코드'].nunique():,}개")
    print(f"  분기 수 : {panel['기준_년분기_코드'].nunique()}개")
    print(f"  총 행 수: {len(panel):,}행")
    print("=" * 60)

    # ── 8. 저장 ───────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)

    panel.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n저장 완료:")
    print(f"  최종 패널    -> {OUT_PATH}")
    print(f"  결측 보고서  -> {REPORT}")
    print(f"\n다음 단계: scripts/06_collect_images.py (Street View 이미지 수집)")


if __name__ == "__main__":
    main()
