# -*- coding: utf-8 -*-
"""
09_filter_images.py — CLIP 기반 이미지 필터링
─────────────────────────────────────────────────────────────────────────────
[역할]
    08_collect_images.py 이후 단계.
    각 샘플 포인트당 수집된 두 방향(A/B) 이미지 중:
      ① 상업 전면(commercial storefront)에 가까운 방향 자동 선택
      ② 선택된 이미지도 품질 기준 미달이면 rejected로 이동
    → 최종적으로 data/images/ 에는 상권당 품질 보장된 이미지만 남음

[처리 흐름]
    1. data/images/{상권코드}/*.jpg 로드
    2. (lat, lng) 기준으로 이미지 쌍 그룹핑
    3. CLIP 코사인 유사도 계산 (text: "commercial storefront")
    4. 쌍 중 높은 점수 이미지 선택 (방향 자동 결정)
    5. 점수 < CLIP_THRESHOLD → data/images_rejected/{상권코드}/ 로 이동
    6. 상권별 유효 이미지 수 집계 → valid_image_sangkwon.csv 저장

[출력]
    data/images/{상권코드}/          ← 유효 이미지만 남음
    data/images_rejected/{상권코드}/ ← 거부 이미지 (삭제 안 함, 수동 검토 가능)
    data/processed/valid_image_sangkwon.csv
        columns: 상권_코드, valid_count, flagged
        flagged=True: 유효 이미지 < MIN_VALID_IMAGES → 11번에서 제외

[CLIP 텍스트 프롬프트]
    한국 상권 맥락 반영:
    - Positive: "street view of Korean commercial district with shops and store signs"
    - Negative: "empty road with no stores or pedestrians"

[수동 복원]
    images_rejected/ 내 이미지를 images/ 로 직접 복사하면
    11번 스크립트에 자동 반영됨 (폴더 내 모든 jpg 스캔)

실행:
    python scripts/09_filter_images.py

의존성:
    pip install torch transformers Pillow --break-system-packages
"""

import os
import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
IMG_DIR       = ROOT / "data/images"
REJECTED_DIR  = ROOT / "data/images_rejected"
VALID_CSV     = ROOT / "data/processed/valid_image_sangkwon.csv"

# ── CLIP 파라미터 ──────────────────────────────────────────────────────────────
CLIP_MODEL_ID   = "openai/clip-vit-base-patch32"
CLIP_THRESHOLD  = 0.22    # 이 점수 미만 → rejected (0~1 범위, 경험적 기준)
MIN_VALID_IMAGES = 5      # 이 미만 상권 → flagged (11번에서 제외)

# ── CLIP 텍스트 프롬프트 ───────────────────────────────────────────────────────
POSITIVE_PROMPT = "street view of Korean commercial district with shops and store signs"
NEGATIVE_PROMPT = "empty road highway with no stores or pedestrians"


# ══════════════════════════════════════════════════════════════════════════════
# CLIP 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_clip(device):
    print(f"  CLIP 모델 로드 중: {CLIP_MODEL_ID}  (device: {device})")
    model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor


@torch.no_grad()
def clip_score(image_path, model, processor, text_features, device):
    """이미지 1장에 대한 CLIP 코사인 유사도 반환 (positive 방향)"""
    try:
        img    = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        img_feat = model.get_image_features(**inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        # positive - negative 차이로 상업성 점수 계산
        score = (img_feat @ text_features.T).squeeze()
        pos_score = score[0].item()
        neg_score = score[1].item()
        return pos_score - neg_score * 0.5   # positive 강조
    except Exception:
        return -1.0


# ══════════════════════════════════════════════════════════════════════════════
# 파일명 파싱: {lat}_{lng}_{heading}.jpg
# ══════════════════════════════════════════════════════════════════════════════
def parse_filename(fname):
    """
    파일명에서 (lat, lng, heading) 추출
    예: '37.556789_126.923456_90.jpg' → ('37.556789', '126.923456', '90')
    """
    m = re.match(r"^(-?\d+\.\d+)_(-?\d+\.\d+)_(\d+)\.jpg$", fname)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("09_filter_images.py — CLIP 기반 이미지 필터링")
    print("=" * 65)

    if not IMG_DIR.exists():
        print(f"  ⚠ {IMG_DIR} 없음. 08_collect_images.py를 먼저 실행하세요.")
        return

    # ── 디바이스 설정 ─────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── CLIP 로드 ─────────────────────────────────────────────────────────────
    model, processor = load_clip(device)

    # 텍스트 피처 사전 계산
    texts  = [POSITIVE_PROMPT, NEGATIVE_PROMPT]
    t_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**t_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"\n  Positive: '{POSITIVE_PROMPT}'")
    print(f"  Negative: '{NEGATIVE_PROMPT}'")
    print(f"  임계값: {CLIP_THRESHOLD}  |  최소 유효 이미지: {MIN_VALID_IMAGES}장\n")

    # ── 상권 폴더 순회 ────────────────────────────────────────────────────────
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    sangkwon_dirs = sorted([d for d in IMG_DIR.iterdir() if d.is_dir()])
    print(f"[1] 처리 대상 상권: {len(sangkwon_dirs)}개\n")

    records = []
    total_kept     = 0
    total_rejected = 0

    for sq_dir in tqdm(sangkwon_dirs, desc="상권 필터링"):
        code   = sq_dir.name
        images = sorted(sq_dir.glob("*.jpg"))

        if not images:
            records.append({"상권_코드": code, "valid_count": 0, "flagged": True})
            continue

        # ── (lat, lng) 기준으로 이미지 쌍 그룹핑 ────────────────────────────
        groups = defaultdict(list)   # key: (lat_str, lng_str) → [Path, ...]
        singles = []                  # 쌍 없이 단독으로 존재하는 이미지

        for img_path in images:
            lat, lng, heading = parse_filename(img_path.name)
            if lat is None:
                singles.append(img_path)
                continue
            groups[(lat, lng)].append(img_path)

        # ── 각 그룹에서 CLIP 점수 계산 → 방향 선택 ──────────────────────────
        rejected_dir = REJECTED_DIR / code
        kept_count   = 0

        all_group_images = list(groups.values()) + [[s] for s in singles]

        for group in all_group_images:
            # 그룹 내 모든 이미지 점수 계산
            scored = []
            for img_path in group:
                score = clip_score(img_path, model, processor, text_features, device)
                scored.append((score, img_path))

            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_path = scored[0]

            if best_score >= CLIP_THRESHOLD:
                # 최고 점수 이미지 보존, 나머지 rejected로
                kept_count += 1
                total_kept += 1
                for score, img_path in scored[1:]:
                    rejected_dir.mkdir(exist_ok=True)
                    shutil.move(str(img_path), str(rejected_dir / img_path.name))
                    total_rejected += 1
            else:
                # 전부 rejected
                for score, img_path in scored:
                    rejected_dir.mkdir(exist_ok=True)
                    shutil.move(str(img_path), str(rejected_dir / img_path.name))
                    total_rejected += 1

        flagged = kept_count < MIN_VALID_IMAGES
        records.append({
            "상권_코드"  : code,
            "valid_count": kept_count,
            "flagged"    : flagged,
        })

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(records).sort_values("상권_코드")
    df.to_csv(VALID_CSV, index=False, encoding="utf-8-sig")

    flagged_df = df[df["flagged"]]
    normal_df  = df[~df["flagged"]]

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("필터링 완료 요약")
    print(f"  유효 이미지 유지       : {total_kept:,}장")
    print(f"  rejected 이동          : {total_rejected:,}장  →  {REJECTED_DIR}")
    print(f"  정상 상권 ({MIN_VALID_IMAGES}장 이상)   : {len(normal_df):,}개")
    print(f"  플래그 상권 ({MIN_VALID_IMAGES}장 미만)  : {len(flagged_df):,}개  ← 11번에서 자동 제외")
    if len(flagged_df) > 0:
        print(f"\n  [플래그 상권 목록]")
        for _, r in flagged_df.iterrows():
            print(f"    {r['상권_코드']}  유효 {r['valid_count']}장")
    print(f"\n  결과 저장              → {VALID_CSV}")
    print(f"{'=' * 65}")
    print("\n수동 검토:")
    print(f"  {REJECTED_DIR} 폴더에서 이미지 확인")
    print(f"  복원할 이미지는 해당 상권 폴더(data/images/)로 직접 복사")
    print("\n다음 단계: python scripts/10_extract_features.py")


if __name__ == "__main__":
    main()
