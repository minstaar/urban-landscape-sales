# -*- coding: utf-8 -*-
"""
07_cross_sectional_baseline.py
─────────────────────────────────────────────────────────────────────────────
Cross-sectional Baseline — 타뷸러 데이터만 사용 (이미지 없음)

[연구 설계 논거]
Panel LSTM(06_baseline_rnn.py)의 R² ~0.97은 상권 간 매출 규모 차이
(cross-sectional variance)가 전체 분산의 ~95%를 차지하기 때문이다.
이 구조에서 정적인 Street View 이미지를 추가해도 개선 여지가 3%에 불과하다.

가로경관 이미지는 "어떤 상권이 매출이 높은가"(cross-sectional)를 설명하는 데
적합하다. 따라서 분석 단위를 분기별 패널 → 상권별 단일 관측값으로 변경한다.

[데이터 구성]
    X (구조적 변수): 훈련 기간(20222~20241) 상권별 평균
        유동인구, 직장인구, 상주인구, 점포_수, 개업률, 폐업률
        CPI, 기준금리, 상권_유형_더미, 면적_km2
        자치구 원-핫 (25개)
    Y (예측 타겟): 테스트 기간(20251~20254) 상권별 평균 log(점포당 매출)
        = log1p(추정매출_합계 / 점포_수)  분기별 평균
        → 점포_수의 규모 효과 제거, 순수한 상권 생산성(productivity) 측정
        → 가로경관 → "점포당 매출이 높은 상권인가?" 로 연구 질문 명료화

    * Phase 2에서 X에 이미지 피처(DINO-ViT 768차원)를 추가해 ΔR² 측정

[모델]
    A. Linear Regression     (선형 기준선)
    B. Ridge (α=10)          (정규화 선형)
    C. Random Forest         (비선형, 변수 중요도 제공)
    D. Gradient Boosting     (비선형, 성능 상한)

[평가]
    5-Fold CV (상권 단위 분할) → R², RMSE(log), MAE(log)
    Random Forest 변수 중요도 출력

실행:
    python scripts/07_cross_sectional_baseline.py

출력:
    data/processed/cross_sectional_data.csv   (Phase 2 입력용, 상권별 1행)
    reports/cross_sectional_metrics.txt
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
DATA     = ROOT / "data/processed/panel_final.csv"
OUT_DATA = ROOT / "data/processed/cross_sectional_data.csv"
OUT_RPT  = ROOT / "reports/cross_sectional_metrics.txt"

# ── 기간 설정 ──────────────────────────────────────────────────────────────────
TRAIN_START = 20222
TRAIN_END   = 20241
TEST_START  = 20251
TEST_END    = 20254

# ── 피처 컬럼 ──────────────────────────────────────────────────────────────────
FEAT_NUM = [
    "유동인구", "직장인구", "상주인구",
    "점포_수", "개업률", "폐업률",
    "CPI", "기준금리",
    "상권_유형_더미", "면적_km2",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Cross-sectional 데이터 구성
# ══════════════════════════════════════════════════════════════════════════════
def build_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    패널 → 상권별 단일 관측값
        X: 훈련 기간(20222~20241) 평균
        Y: 테스트 기간(20251~20254) 평균 log_sales
    """
    df = df.copy()
    df["log_sales"] = np.log1p(df["추정매출_합계"])

    train = df[(df["기준_년분기_코드"] >= TRAIN_START) & (df["기준_년분기_코드"] <= TRAIN_END)]
    test  = df[(df["기준_년분기_코드"] >= TEST_START)  & (df["기준_년분기_코드"] <= TEST_END)]

    # X: 훈련 기간 수치 변수 평균
    x_agg = train.groupby("상권_코드")[FEAT_NUM].mean()

    # 메타 정보 (상권별 고정값)
    meta = (train.groupby("상권_코드")[["상권_코드_명", "상권_구분_코드_명",
                                        "자치구_코드", "자치구_코드_명"]]
            .first())

    # Y: 테스트 기간 점포당 매출 (분기별 산출 후 평균)
    #    점포_수=0 인 행은 제외 (log1p 왜곡 방지)
    test = test.copy()
    test["sales_per_store"] = (
        test["추정매출_합계"] / test["점포_수"].replace(0, np.nan)
    )
    test["log_sales_per_store"] = np.log1p(test["sales_per_store"])
    y_agg = (test.groupby("상권_코드")["log_sales_per_store"]
             .mean()
             .rename("log_sales_test"))

    # 병합
    cs = x_agg.join(meta).join(y_agg)
    before = len(cs)
    cs = cs.dropna()
    after = len(cs)
    if before != after:
        print(f"    ⚠ 결측 제거: {before} → {after}개 상권")

    return cs.reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# 2. 피처 행렬 구성 (자치구 원-핫 포함)
# ══════════════════════════════════════════════════════════════════════════════
def make_feature_matrix(cs: pd.DataFrame):
    """수치 변수 + 자치구 원-핫 → numpy array 반환"""
    dist_dummies = pd.get_dummies(cs["자치구_코드"], prefix="gu", drop_first=True)
    X = pd.concat([cs[FEAT_NUM], dist_dummies], axis=1).astype(float)
    y = cs["log_sales_test"].values
    feat_names = X.columns.tolist()
    return X.values, y, feat_names


# ══════════════════════════════════════════════════════════════════════════════
# 3. 지표 계산
# ══════════════════════════════════════════════════════════════════════════════
def cv_evaluate(model, X, y, n_splits=5):
    """5-Fold CV → R², RMSE, MAE (log scale)"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2s, rmses, maes = [], [], []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[val_idx])
        r2s.append(r2_score(y[val_idx], pred))
        rmses.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        maes.append(mean_absolute_error(y[val_idx], pred))
    return {
        "R2_mean":   np.mean(r2s),
        "R2_std":    np.std(r2s),
        "RMSE_mean": np.mean(rmses),
        "MAE_mean":  np.mean(maes),
    }


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("07_cross_sectional_baseline.py")
    print("  분석 단위: 상권 (854개) × 1행  |  Y: 테스트 기간 평균 매출")
    print("=" * 65)

    # ── 1. 데이터 로드 및 집계 ────────────────────────────────────────────────
    print("\n[1] 데이터 로드 및 Cross-sectional 집계 중...")
    df = pd.read_csv(DATA)
    cs = build_cross_section(df)
    print(f"    최종 상권 수: {len(cs):,}개")
    print(f"    Y (log 점포당 매출) — 평균: {cs['log_sales_test'].mean():.3f}, "
          f"표준편차: {cs['log_sales_test'].std():.3f}")

    # ── 2. 피처 행렬 ─────────────────────────────────────────────────────────
    X, y, feat_names = make_feature_matrix(cs)
    print(f"    피처 차원: {X.shape[1]}개 "
          f"(수치 {len(FEAT_NUM)}개 + 자치구 더미 {X.shape[1]-len(FEAT_NUM)}개)")

    # ── 3. 모델 정의 ──────────────────────────────────────────────────────────
    print("\n[2] 5-Fold CV 평가 중...\n")

    models = {
        "A. Linear Regression ": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "B. Ridge (α=10)      ": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0)),
        ]),
        "C. Random Forest     ": RandomForestRegressor(
            n_estimators=300, max_depth=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
        "D. Gradient Boosting ": GradientBoostingRegressor(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=42,
        ),
    }

    results = {}
    for name, model in models.items():
        res = cv_evaluate(model, X, y)
        results[name] = res
        print(f"  {name}  R²={res['R2_mean']:.4f} ±{res['R2_std']:.4f}  "
              f"RMSE={res['RMSE_mean']:.4f}  MAE={res['MAE_mean']:.4f}")

    # ── 4. 변수 중요도 (Random Forest, 전체 데이터 재학습) ───────────────────
    print("\n[3] Random Forest 변수 중요도 (상위 10개)...")
    rf_full = RandomForestRegressor(
        n_estimators=300, max_depth=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1,
    )
    rf_full.fit(X, y)
    importance = (pd.Series(rf_full.feature_importances_, index=feat_names)
                  .sort_values(ascending=False))
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"    {feat:25s}: {imp:.4f}  {bar}")

    # ── 5. Panel LSTM vs Cross-sectional 비교 요약 ────────────────────────────
    best_r2 = max(v["R2_mean"] for v in results.values())
    best_model = max(results, key=lambda k: results[k]["R2_mean"]).strip()

    print(f"\n{'=' * 65}")
    print("[ Panel LSTM vs Cross-sectional 비교 ]")
    print(f"{'=' * 65}")
    print(f"  Panel LSTM (06_baseline_rnn.py)")
    print(f"    - Val R² ≈ 0.97  /  Test R² ≈ 0.95")
    print(f"    - 높은 R²의 원인: 상권 간 매출 규모 차이(cross-sectional variance)가")
    print(f"      전체 분산의 ~95%를 차지 → 이미지가 기여할 여지 < 3%")
    print(f"    - 정적 Street View 이미지는 temporal 변화를 설명 불가")
    print(f"    → 이미지 추가 효과를 보이기 부적합한 구조")
    print()
    print(f"  Cross-sectional (이 스크립트, 타뷸러만)")
    print(f"    - 최고 모델({best_model}): R² ≈ {best_r2:.4f}")
    print(f"    - 상권별 단일 관측 → 이미지(static)가 설명해야 할 분산이 충분히 남아있음")
    print(f"    → Phase 2: 이미지 피처 추가 시 ΔR² 측정 가능")
    print(f"{'=' * 65}")

    # ── 6. 저장 ───────────────────────────────────────────────────────────────
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_RPT.parent.mkdir(parents=True, exist_ok=True)

    # Phase 2를 위한 데이터 저장 (상권별 1행, 이미지 피처 컬럼 추가 예정)
    cs_save = cs.copy()
    cs_save.to_csv(OUT_DATA, index=False, encoding="utf-8-sig")

    # 보고서 저장
    report_lines = [
        "Cross-sectional Baseline (타뷸러 데이터만) — 성능 보고서",
        "=" * 50,
        f"분석 단위  : 상권별 1행 (총 {len(cs):,}개 상권)",
        f"X 기간     : 훈련 기간 (20222~20241) 평균",
        f"Y 기간     : 테스트 기간 (20251~20254) 평균 log(점포당 매출)",
        f"피처 차원  : {X.shape[1]}개 (수치 {len(FEAT_NUM)} + 자치구 더미)",
        f"평가 방법  : 5-Fold CV (상권 단위)",
        "",
        "── 모델별 성능 (5-Fold CV)",
        f"  {'모델':<24}  {'R² 평균':>8}  {'R² 표준편차':>10}  {'RMSE':>8}  {'MAE':>8}",
    ]
    for name, res in results.items():
        report_lines.append(
            f"  {name:<24}  {res['R2_mean']:>8.4f}  {res['R2_std']:>10.4f}  "
            f"{res['RMSE_mean']:>8.4f}  {res['MAE_mean']:>8.4f}"
        )
    report_lines += [
        "",
        "── Random Forest 변수 중요도 (상위 10개)",
    ]
    for feat, imp in importance.head(10).items():
        report_lines.append(f"  {feat:25s}: {imp:.4f}")
    report_lines += [
        "",
        "── 비교 분석 메모",
        "  Panel LSTM R²(val)≈0.97: cross-sectional variance 선점으로 인한 ceiling effect",
        "  Cross-sectional R²: 이미지 추가 시 ΔR² 측정을 위한 적정 기준선",
        "",
        "── Phase 2 계획",
        "  08_collect_images.py 실행 후 Street View 이미지 수집",
        "  09_extract_image_features.py: DINO-ViT CLS token (768차원)",
        "  cross_sectional_data.csv에 이미지 피처 컬럼 추가",
        "  동일 모델로 재학습 → ΔR² 측정",
    ]
    OUT_RPT.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n저장 완료:")
    print(f"  Cross-sectional 데이터 → {OUT_DATA}")
    print(f"  성능 보고서            → {OUT_RPT}")
    print(f"\n다음 단계: scripts/08_collect_images.py (Street View 이미지 수집)")


if __name__ == "__main__":
    main()
