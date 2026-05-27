# -*- coding: utf-8 -*-
"""
06_baseline_rnn.py
─────────────────────────────────────────────────────────────────────────────
Panel LSTM Baseline — 타뷸러 데이터만 사용 (lag Y 제외)

연구 설계 원칙:
    이전 분기 매출(lag Y)을 입력에서 제외한다.
    목적이 "예측 정확도 극대화"가 아니라
    "가로경관 이미지가 구조적 변수 위에 추가 설명력을 갖는가"이기 때문이다.
    lag Y를 포함하면 이미지의 기여가 나머지 ~3%에 불과해 논문 주장이 약해진다.

모델 입력:
    시변 X : 유동인구, 직장인구, 상주인구, 점포_수, 개업률, 폐업률,
             CPI, 기준금리, Q_sin, Q_cos  (lag Y 없음)
    Static : 상권_유형_더미, 면적_km2  → LSTM 마지막 hidden에 concat
    자치구  : nn.Embedding(25, 8)

예측 타겟:
    log_sales를 StandardScaler로 정규화 → 모델 예측 → 역변환 → expm1
    (scale 불일치 해결, lag Y 없이도 학습 안정)

Naive 기준선:
    각 상권의 훈련 기간 평균 log_sales를 모든 분기에 예측
    ("역사적 평균 상권 매출" — lag 없이 쓸 수 있는 가장 단순한 기준)

데이터 분할:
    Train : 20222 ~ 20241  (8분기)
    Val   : 20242 ~ 20244  (3분기)
    Test  : 20251 ~ 20254  (4분기)

실행:
    python scripts/06_baseline_rnn.py

출력:
    data/processed/baseline_predictions.csv
    models/baseline_lstm.pt
    reports/baseline_metrics.txt
"""

import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ── 재현성 시드 ───────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "data/processed/panel_final.csv"
OUT_PRED  = ROOT / "data/processed/baseline_predictions.csv"
OUT_MODEL = ROOT / "models/baseline_lstm.pt"
OUT_RPT   = ROOT / "reports/baseline_metrics.txt"

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
LOOKBACK   = 4          # 입력 시퀀스 길이 (분기)
HIDDEN_DIM = 64
N_LAYERS   = 2
EMBED_DIM  = 8          # 자치구 임베딩 차원
DROPOUT    = 0.2
LR         = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 300
PATIENCE   = 30

# ── 분기 코드 ─────────────────────────────────────────────────────────────────
TRAIN_START = 20222
TRAIN_END   = 20241
VAL_START   = 20242
VAL_END     = 20244
TEST_START  = 20251
TEST_END    = 20254

# ── 시변 X 컬럼 (lag Y 없음) ──────────────────────────────────────────────────
TIME_COLS = [
    "유동인구", "직장인구", "상주인구",
    "점포_수", "개업률", "폐업률",
    "CPI", "기준금리",
]


def quarter_sin_cos(q_code: int):
    q = q_code % 10
    return math.sin(2 * math.pi * q / 4), math.cos(2 * math.pi * q / 4)


# ══════════════════════════════════════════════════════════════════════════════
# 전처리
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    df = df.copy().sort_values(["상권_코드", "기준_년분기_코드"]).reset_index(drop=True)

    # 분기 인코딩
    enc = df["기준_년분기_코드"].apply(
        lambda x: pd.Series(quarter_sin_cos(x), index=["Q_sin", "Q_cos"])
    )
    df = pd.concat([df, enc], axis=1)

    # log_sales
    df["log_sales"] = np.log1p(df["추정매출_합계"])

    # 자치구 index
    districts   = sorted(df["자치구_코드"].unique())
    dist_map    = {d: i for i, d in enumerate(districts)}
    n_districts = len(districts)
    df["자치구_idx"] = df["자치구_코드"].map(dist_map)

    time_cols_all  = TIME_COLS + ["Q_sin", "Q_cos"]

    # Train 기간 마스크
    train_mask = (df["기준_년분기_코드"] >= TRAIN_START) & (df["기준_년분기_코드"] <= TRAIN_END)

    # 정적 입력: 상권유형더미, 면적_km2
    static_num_cols = ["상권_유형_더미", "면적_km2"]

    # ── Scaler fit (Train 기간만) ──────────────────────────────────────────
    df[time_cols_all] = df[time_cols_all].fillna(df[time_cols_all].median())

    scaler_time = StandardScaler()
    scaler_time.fit(df.loc[train_mask, time_cols_all])
    df[time_cols_all] = scaler_time.transform(df[time_cols_all])

    scaler_static = StandardScaler()
    scaler_static.fit(df.loc[train_mask, static_num_cols])
    df[static_num_cols] = scaler_static.transform(df[static_num_cols])

    # ── Y 정규화 (log_sales → 평균 0, 표준편차 1) ─────────────────────────
    # lag Y 없이도 학습 안정화; 역변환 후 expm1으로 원단위 복원
    scaler_y = StandardScaler()
    scaler_y.fit(df.loc[train_mask, ["log_sales"]])
    df["log_sales_sc"] = scaler_y.transform(df[["log_sales"]])

    # ── 상권별 훈련 평균 (Naive baseline용) ────────────────────────────────
    train_mean = (
        df.loc[train_mask, ["상권_코드", "log_sales"]]
        .groupby("상권_코드")["log_sales"].mean()
    )

    # ── 상권별 시계열 딕셔너리 ────────────────────────────────────────────
    sangkwon_data = {}
    for code, grp in df.groupby("상권_코드"):
        grp = grp.sort_values("기준_년분기_코드").reset_index(drop=True)
        sangkwon_data[code] = {
            "quarters"      : grp["기준_년분기_코드"].values,
            "time_x"        : grp[time_cols_all].values.astype(np.float32),
            "static_num"    : grp[static_num_cols].iloc[0].values.astype(np.float32),
            "dist_idx"      : int(grp["자치구_idx"].iloc[0]),
            "log_sales_sc"  : grp["log_sales_sc"].values.astype(np.float32),
            "log_sales_raw" : grp["log_sales"].values.astype(np.float32),   # 역변환용
            "raw_sales"     : grp["추정매출_합계"].values.astype(np.float64),
            "naive_log"     : float(train_mean.get(code, grp["log_sales"].mean())),
            "name"          : grp["상권_코드_명"].iloc[0],
        }

    return sangkwon_data, n_districts, scaler_y


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class PanelDataset(Dataset):
    def __init__(self, sangkwon_data: dict, q_start: int, q_end: int):
        self.samples = []
        for code, d in sangkwon_data.items():
            quarters = d["quarters"]
            for i in range(LOOKBACK, len(quarters)):
                tq = quarters[i]
                if not (q_start <= tq <= q_end):
                    continue
                self.samples.append({
                    "x_seq"       : torch.tensor(d["time_x"][i - LOOKBACK: i]),
                    "x_static_num": torch.tensor(d["static_num"]),
                    "x_dist"      : torch.tensor(d["dist_idx"], dtype=torch.long),
                    "y_sc"        : torch.tensor(d["log_sales_sc"][i]),
                    "y_raw"       : d["raw_sales"][i],
                    "naive_log"   : d["naive_log"],
                    "code"        : code,
                    "q_code"      : tq,
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate(batch):
    return {
        "x_seq"       : torch.stack([b["x_seq"]        for b in batch]),
        "x_static_num": torch.stack([b["x_static_num"] for b in batch]),
        "x_dist"      : torch.stack([b["x_dist"]       for b in batch]),
        "y_sc"        : torch.stack([b["y_sc"]         for b in batch]),
        "y_raw"       : np.array([b["y_raw"]           for b in batch]),
        "naive_log"   : np.array([b["naive_log"]       for b in batch]),
        "code"        : [b["code"]                     for b in batch],
        "q_code"      : [b["q_code"]                   for b in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
class PanelLSTM(nn.Module):
    """
    Panel LSTM (lag Y 미사용)
      x_seq → LSTM → h_last
      concat [h_last | static_num | district_embed]
      → FC → normalized log_sales
    """
    def __init__(self, input_dim, hidden_dim, n_layers,
                 static_num_dim, n_districts, embed_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.district_embed = nn.Embedding(n_districts, embed_dim)
        self.dropout = nn.Dropout(dropout)

        fc_in = hidden_dim + static_num_dim + embed_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_in // 2, 1),
        )

    def forward(self, x_seq, x_static_num, x_dist):
        _, (h_n, _) = self.lstm(x_seq)
        h_last   = self.dropout(h_n[-1])
        d_emb    = self.district_embed(x_dist)
        combined = torch.cat([h_last, x_static_num, d_emb], dim=-1)
        return self.fc(combined).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 지표 계산
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(preds_raw, trues_raw):
    err     = preds_raw - trues_raw
    abs_err = np.abs(err)
    rmse  = np.sqrt(np.mean(err ** 2))
    mae   = np.mean(abs_err)
    smape = np.mean(
        2 * abs_err / (np.abs(trues_raw) + np.abs(preds_raw) + 1e-8)
    ) * 100
    mdape = np.median(abs_err / (np.abs(trues_raw) + 1e-8)) * 100
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((trues_raw - trues_raw.mean()) ** 2)
    r2    = 1 - ss_res / (ss_tot + 1e-8)
    return {"RMSE": rmse, "MAE": mae, "sMAPE": smape, "MdAPE": mdape, "R2": r2}


# ══════════════════════════════════════════════════════════════════════════════
# 평가
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, loader, device, scaler_y):
    model.eval()
    preds_raw, trues_raw, naive_raw = [], [], []
    codes, q_codes = [], []

    with torch.no_grad():
        for batch in loader:
            x_seq = batch["x_seq"].to(device)
            x_sn  = batch["x_static_num"].to(device)
            x_d   = batch["x_dist"].to(device)

            pred_sc = model(x_seq, x_sn, x_d).cpu().numpy()

            # 역정규화: scaled log_sales → log_sales → 원단위
            pred_log = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
            pred_r   = np.expm1(pred_log)

            # Naive: 훈련 기간 상권별 평균 log_sales → 원단위
            naive_r  = np.expm1(batch["naive_log"])

            preds_raw.extend(pred_r)
            trues_raw.extend(batch["y_raw"])
            naive_raw.extend(naive_r)
            codes.extend(batch["code"])
            q_codes.extend(batch["q_code"])

    preds_raw = np.array(preds_raw)
    trues_raw = np.array(trues_raw)
    naive_raw = np.array(naive_raw)

    metrics       = compute_metrics(preds_raw, trues_raw)
    naive_metrics = compute_metrics(naive_raw, trues_raw)

    records = pd.DataFrame({
        "상권_코드"      : codes,
        "기준_년분기_코드": q_codes,
        "실제매출"       : trues_raw,
        "예측매출_LSTM"  : preds_raw,
        "예측매출_Naive" : naive_raw,
    })
    return metrics, naive_metrics, records


# ══════════════════════════════════════════════════════════════════════════════
# 학습 루프
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x_seq = batch["x_seq"].to(device)
        x_sn  = batch["x_static_num"].to(device)
        x_d   = batch["x_dist"].to(device)
        y_sc  = batch["y_sc"].to(device)

        optimizer.zero_grad()
        pred = model(x_seq, x_sn, x_d)
        loss = criterion(pred, y_sc)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_sc)

    return total_loss / len(loader.dataset)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("06_baseline_rnn.py  (lag Y 제외 버전)")
    print(f"  Device: {device}")
    print("=" * 60)

    # 1. 전처리
    print("\n[1] 데이터 전처리 중...")
    df = pd.read_csv(DATA)
    sangkwon_data, n_districts, scaler_y = preprocess(df)
    print(f"    상권: {len(sangkwon_data):,}개 | 자치구: {n_districts}개")

    input_dim      = len(TIME_COLS) + 2   # Q_sin, Q_cos
    static_num_dim = 2                    # 상권유형더미, 면적_km2

    # 2. Dataset
    print("\n[2] Dataset 구성 중...")
    train_ds = PanelDataset(sangkwon_data, TRAIN_START, TRAIN_END)
    val_ds   = PanelDataset(sangkwon_data, VAL_START,   VAL_END)
    test_ds  = PanelDataset(sangkwon_data, TEST_START,  TEST_END)
    print(f"    Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=0)

    # 3. 모델
    print("\n[3] 모델 초기화...")
    model = PanelLSTM(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        static_num_dim=static_num_dim, n_districts=n_districts,
        embed_dim=EMBED_DIM, dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    파라미터 수: {n_params:,}개")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    # 4. 학습
    print(f"\n[4] 학습 (max {MAX_EPOCHS} epochs, patience {PATIENCE})...")
    best_val_r2  = -float("inf")
    best_state   = None
    no_improve   = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m, _, _ = evaluate(model, val_loader, device, scaler_y)
        scheduler.step(val_m["RMSE"])

        if val_m["R2"] > best_val_r2:
            best_val_r2 = val_m["R2"]
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"Val sMAPE {val_m['sMAPE']:.2f}% | "
                  f"Val MdAPE {val_m['MdAPE']:.2f}% | "
                  f"Val R² {val_m['R2']:.4f}")

        if no_improve >= PATIENCE:
            print(f"  → Early stopping at epoch {epoch}")
            break

    # 5. 최종 평가
    print("\n[5] 최종 평가...")
    model.load_state_dict(best_state)
    val_m,  val_n,  val_df  = evaluate(model, val_loader,  device, scaler_y)
    test_m, test_n, test_df = evaluate(model, test_loader, device, scaler_y)

    header = f"  {'':12s}  {'LSTM':>10s}  {'Naive(mean)':>12s}"
    rows = [
        ("RMSE(억)", "RMSE",  1e8),
        ("MAE(억)",  "MAE",   1e8),
        ("sMAPE(%)", "sMAPE", 1),
        ("MdAPE(%)", "MdAPE", 1),
        ("R²",       "R2",    1),
    ]

    def print_block(label, m, n):
        print(f"\n── {label}")
        print(header)
        for name, key, div in rows:
            print(f"  {name:12s}  {m[key]/div:>10.4f}  {n[key]/div:>12.4f}")

    print(f"\n{'='*60}")
    print_block("Validation (20242~20244)", val_m,  val_n)
    print_block("Test       (20251~20254)", test_m, test_n)
    print(f"\n{'='*60}")

    # 6. 저장
    for p in [OUT_PRED, OUT_MODEL, OUT_RPT]:
        p.parent.mkdir(parents=True, exist_ok=True)

    pd.concat([
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ]).to_csv(OUT_PRED, index=False, encoding="utf-8-sig")

    torch.save({
        "model_state": best_state,
        "hyperparams": {
            "lookback": LOOKBACK, "hidden_dim": HIDDEN_DIM,
            "n_layers": N_LAYERS, "embed_dim": EMBED_DIM,
            "n_districts": n_districts, "input_dim": input_dim,
        },
        "scaler_y_mean": scaler_y.mean_[0],
        "scaler_y_std" : scaler_y.scale_[0],
    }, OUT_MODEL)

    def fmt(m, n):
        lines = [f"  {'':12s}  {'LSTM':>10s}  {'Naive(mean)':>12s}"]
        for name, key, div in rows:
            lines.append(f"  {name:12s}  {m[key]/div:>10.4f}  {n[key]/div:>12.4f}")
        return lines

    report = [
        "Panel LSTM Baseline (lag Y 제외) — 성능 보고서",
        "=" * 44,
        f"Lookback   : {LOOKBACK}분기",
        f"Hidden dim : {HIDDEN_DIM}",
        f"LSTM layers: {N_LAYERS}",
        f"파라미터 수 : {n_params:,}",
        f"Naive 기준  : 상권별 훈련 기간 평균 매출",
        "",
        "── Validation (20242~20244)",
    ] + fmt(val_m, val_n) + [
        "",
        "── Test (20251~20254)",
    ] + fmt(test_m, test_n)

    OUT_RPT.write_text("\n".join(report), encoding="utf-8")

    print(f"\n저장 완료:")
    print(f"  예측 결과   → {OUT_PRED}")
    print(f"  모델 가중치 → {OUT_MODEL}")
    print(f"  성능 보고서 → {OUT_RPT}")
    print("\n다음 단계: scripts/07_collect_images.py (Street View 이미지 수집)")


if __name__ == "__main__":
    main()
