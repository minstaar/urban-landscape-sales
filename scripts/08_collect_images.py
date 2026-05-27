# -*- coding: utf-8 -*-
"""
08_collect_images.py
─────────────────────────────────────────────────────────────────────────────
Street View 이미지 수집 — 도로 엣지 수직 샘플링 (test_08_images.py 검증 완료)

[수집 전략]
    1. 상권 폴리곤 내 상업 도로 엣지 추출 (osmnx)
    2. 엣지 중점을 SAMPLE_INTERVAL_M 간격으로 분할 → 후보 포인트 생성
    3. 공간 분산 필터(MIN_SPREAD_M) → n_pts개 선택
    4. 각 포인트에서 도로에 수직인 두 방향(±90°) 모두 이미지 수집
       → 어느 쪽이 상업 전면인지는 09_filter_images.py(CLIP)에서 자동 결정
    5. 메타데이터 API로 실제 파노라마 위치 확인 → MAX_SNAP_M 초과 시 스킵

[방향 결정 근거]
    선행 연구 대부분은 방향 기준 미명시("at intervals along roads").
    본 연구는 두 수직 방향 모두 수집 후 CLIP으로 자동 선택 → 방법론적으로
    더 엄밀하며 논문에 명확히 기술 가능.
    (e.g. "The commercially-oriented image was automatically selected using
    CLIP cosine similarity to 'commercial storefront'; the alternative is
    retained for manual inspection.")

[샘플 수]
    n_pts = max(8, min(15, int(area_m2 / 25_000)))
    각 포인트 × 2방향 → 상권당 16~30장 수집 (CLIP 필터링 전)

[비용 추정]
    852개 상권 × 11포인트(평균) × 2방향 ≈ 18,700장
    Google $200 무료 크레딧 / 장당 $0.007 → 무료 크레딧 내 처리 가능

[보안]
    API 키는 반드시 .env 파일에서 로드
    GOOGLE_STREETVIEW_API_KEY=your_key_here
    절대 코드에 하드코딩 금지

[파일명 규칙]
    {lat:.6f}_{lng:.6f}_{heading}.jpg
    → 좌표·방향 동일 = 파일명 동일 → 중복 다운로드 방지
    → 재실행 안전 (이미 존재하는 파일 자동 스킵)

[이어받기]
    파일이 이미 존재하면 자동 스킵

실행:
    python scripts/08_collect_images.py

출력:
    data/images/{상권_코드}/{lat}_{lng}_{heading}.jpg
    data/processed/image_sampling_log.csv
"""

import argparse
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 환경변수 ───────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_STREETVIEW_API_KEY")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
SHP_PATH  = ROOT / "data/raw/boundaries/서울시 상권분석서비스(영역-상권).shp"
LIST_PATH = ROOT / "data/processed/final_sangkwon_list.csv"
IMG_DIR   = ROOT / "data/images"
LOG_PATH  = ROOT / "data/processed/image_sampling_log.csv"

# ── Street View API 설정 ───────────────────────────────────────────────────────
SV_BASE   = "https://maps.googleapis.com/maps/api/streetview"
SV_META   = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMG_SIZE  = "640x640"
FOV       = 90
PITCH     = -10   # 살짝 아래 → 간판·쇼윈도 중심
SV_RADIUS = 30    # 파노라마 탐색 반경(m) — 좁게 유지해 실내 스냅 방지

# ── 샘플링 파라미터 ───────────────────────────────────────────────────────────
MAX_SNAP_M        = 30   # 파노라마-요청좌표 거리 초과 시 실내 스냅으로 간주, 스킵
SAMPLE_INTERVAL_M = 20   # 긴 엣지 분할 샘플링 간격(m)
MAX_PTS_PER_EDGE  = 2    # 엣지당 최대 포인트 수 — 긴 도로 독점 방지
MIN_SPREAD_M      = 30   # 포인트 간 최소 거리(m) — 공간 분산
MAJOR_ROAD_QUOTA  = 2    # 상권당 primary/secondary 최대 포인트 수 (층화 샘플링)

# ── 도로 타입 분류 ─────────────────────────────────────────────────────────────
MAJOR_HIGHWAYS = {"primary", "primary_link", "secondary", "secondary_link"}
MINOR_HIGHWAYS = {
    "tertiary", "tertiary_link",
    "residential", "living_street",
    "pedestrian", "unclassified",
}

# ── 도로 타입 우선순위 (낮을수록 먼저 선택) ────────────────────────────────────
ROAD_PRIORITY = {
    "pedestrian"    : 0,
    "living_street" : 1,
    "residential"   : 2,
    "unclassified"  : 3,
    "tertiary"      : 4,
    "tertiary_link" : 4,
    "secondary"     : 5,
    "secondary_link": 5,
    "primary"       : 6,
    "primary_link"  : 6,
}

# ── 상업 도로 타입 필터 ────────────────────────────────────────────────────────
COMMERCIAL_HIGHWAYS = {
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential",
    "living_street",
    "pedestrian",
    "unclassified",
}


# ══════════════════════════════════════════════════════════════════════════════
# 유틸 함수
# ══════════════════════════════════════════════════════════════════════════════
def n_sample_points(area_m2: float) -> int:
    """면적(m²) 기반 샘플 포인트 수: 25,000m²당 1개, 최소 8개 / 최대 15개"""
    return max(8, min(15, int(area_m2 / 25_000)))


def get_highway(val):
    return val[0] if isinstance(val, list) else val


def compute_heading(from_lat, from_lng, to_lat, to_lng):
    """두 점 간 방위각 (0~360°)"""
    dlon = math.radians(to_lng - from_lng)
    lat1 = math.radians(from_lat)
    lat2 = math.radians(to_lat)
    x    = math.sin(dlon) * math.cos(lat2)
    y    = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def perp_both(edge_bearing):
    """
    엣지에 수직인 두 방향 모두 반환 (±90°)
    → CLIP 필터링 단계에서 상업 전면 방향 자동 선택
    """
    return round((edge_bearing + 90) % 360), round((edge_bearing + 270) % 360)


def dist_m(lat1, lng1, lat2, lng2):
    """두 좌표 간 거리(m) — 서울 위도 근사"""
    return math.hypot(lat1 - lat2, lng1 - lng2) * 111_000


# ══════════════════════════════════════════════════════════════════════════════
# 도로 엣지 샘플링
# ══════════════════════════════════════════════════════════════════════════════
def sample_edges_in_polygon(edges_gdf, nodes_gdf, polygon):
    """
    폴리곤 내 도로 엣지에서 샘플 포인트 + 수직 두 heading + 도로타입 추출

    Returns: [(lat, lng, heading_A, heading_B, road_type), ...]
    """
    candidates = []

    for (u, v, _), edge in edges_gdf.iterrows():
        if u not in nodes_gdf.index or v not in nodes_gdf.index:
            continue

        road_type = get_highway(edge.get("highway", "unclassified"))
        if road_type not in ROAD_PRIORITY:
            road_type = "unclassified"

        u_node = nodes_gdf.loc[u]
        v_node = nodes_gdf.loc[v]
        u_lat, u_lng = float(u_node.geometry.y), float(u_node.geometry.x)
        v_lat, v_lng = float(v_node.geometry.y), float(v_node.geometry.x)

        mid_lat = (u_lat + v_lat) / 2
        mid_lng = (u_lng + v_lng) / 2
        if not polygon.contains(Point(mid_lng, mid_lat)):
            continue

        edge_len_m = dist_m(u_lat, u_lng, v_lat, v_lng)
        if edge_len_m < SAMPLE_INTERVAL_M:
            sample_fracs = [0.5]
        else:
            n_seg = max(1, int(edge_len_m / SAMPLE_INTERVAL_M))
            n_seg = min(n_seg, MAX_PTS_PER_EDGE)
            sample_fracs = [(i + 0.5) / n_seg for i in range(n_seg)]

        edge_bearing = compute_heading(u_lat, u_lng, v_lat, v_lng)
        h_a, h_b = perp_both(edge_bearing)

        for t in sample_fracs:
            pt_lat = u_lat + t * (v_lat - u_lat)
            pt_lng = u_lng + t * (v_lng - u_lng)
            candidates.append((pt_lat, pt_lng, h_a, h_b, road_type))

    return candidates


def spread_sample(pts, n, min_dist_m):
    """
    공간 분산 샘플링 — 위치(lat, lng)만 기준으로 거리 계산
    pts: [(lat, lng, heading_A, heading_B, road_type), ...]
    """
    min_deg  = min_dist_m / 111_000
    selected = []
    for pt in pts:
        if len(selected) >= n:
            break
        if not selected or min(
            math.hypot(pt[0] - s[0], pt[1] - s[1]) for s in selected
        ) >= min_deg:
            selected.append(pt)
    return selected


def stratified_sample(candidates, n_pts, min_dist_m):
    """
    층화 샘플링:
      1. minor pool (tertiary/residential/pedestrian): 우선순위 정렬 후 최대 (n_pts - MAJOR_ROAD_QUOTA)개
      2. major pool (primary/secondary): 최대 MAJOR_ROAD_QUOTA개 보충
      3. minor가 부족하면 major로 남은 자리 채움

    → 골목/이면도로 우선 + 대로변 최소 보장
    """
    minor = sorted(
        [c for c in candidates if c[4] in MINOR_HIGHWAYS],
        key=lambda x: ROAD_PRIORITY.get(x[4], 3)
    )
    major = sorted(
        [c for c in candidates if c[4] in MAJOR_HIGHWAYS],
        key=lambda x: ROAD_PRIORITY.get(x[4], 5)
    )

    # minor 먼저 채우기
    minor_quota = n_pts - MAJOR_ROAD_QUOTA
    selected_minor = spread_sample(minor, max(minor_quota, 0), min_dist_m)

    # major에서 최대 MAJOR_ROAD_QUOTA개 추가
    # 이미 선택된 포인트와 거리 조건 만족하는 것만
    selected_major = []
    min_deg = min_dist_m / 111_000
    for pt in major:
        if len(selected_major) >= MAJOR_ROAD_QUOTA:
            break
        all_selected = selected_minor + selected_major
        if not all_selected or min(
            math.hypot(pt[0] - s[0], pt[1] - s[1]) for s in all_selected
        ) >= min_deg:
            selected_major.append(pt)

    result = selected_minor + selected_major

    # minor가 부족해서 n_pts 미달이면 major로 추가 보충
    if len(result) < n_pts:
        remaining_major = [p for p in major if p not in selected_major]
        extra = spread_sample(remaining_major, n_pts - len(result), min_dist_m)
        result += extra

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Street View API
# ══════════════════════════════════════════════════════════════════════════════
def get_pano_location(lat, lng):
    """메타데이터 API → 실제 파노라마 위치 반환 / 없으면 None"""
    try:
        r = requests.get(
            SV_META,
            params={"location": f"{lat},{lng}", "radius": SV_RADIUS, "key": API_KEY},
            timeout=10,
        )
        data = r.json()
        if data.get("status") != "OK":
            return None
        loc = data["location"]
        return float(loc["lat"]), float(loc["lng"])
    except Exception:
        return None


def download_sv_image(lat, lng, heading, save_path):
    """Street View Static API 이미지 다운로드 (5KB 미만은 에러 이미지로 간주)"""
    try:
        r = requests.get(
            SV_BASE,
            params={
                "size"    : IMG_SIZE,
                "location": f"{lat},{lng}",
                "heading" : heading,
                "fov"     : FOV,
                "pitch"   : PITCH,
                "key"     : API_KEY,
            },
            timeout=20,
        )
        if r.status_code == 200 and len(r.content) > 5_000:
            save_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part",  type=int, default=1,
                        help="실행할 파트 번호 (1 또는 2). 기본값: 1 (전체)")
    parser.add_argument("--total", type=int, default=1,
                        help="총 파트 수. 기본값: 1 (전체). 2로 설정하면 절반씩 분할")
    args = parser.parse_args()

    if not API_KEY:
        raise ValueError(
            "GOOGLE_STREETVIEW_API_KEY가 .env에 없습니다.\n"
            "프로젝트 루트의 .env 파일에 아래 줄을 추가하세요:\n"
            "GOOGLE_STREETVIEW_API_KEY=your_key_here"
        )

    print("=" * 65)
    print("08_collect_images.py — 도로 엣지 수직 샘플링 (전체 상권)")
    if args.total > 1:
        print(f"  분할 모드: {args.part}/{args.total} 파트")
    print("=" * 65)

    # ── 1. 상권 폴리곤 로드 ────────────────────────────────────────────────────
    print("\n[1] 상권 경계 Shapefile 로드 중...")
    gdf = gpd.read_file(SHP_PATH, encoding="cp949")
    gdf = gdf.set_crs("EPSG:5181", allow_override=True).to_crs("EPSG:4326")
    gdf = gdf.rename(columns={"TRDAR_CD": "상권_코드", "RELM_AR": "영역_면적"})
    gdf["상권_코드"] = gdf["상권_코드"].astype(str).str.strip()

    valid_codes = set(pd.read_csv(LIST_PATH)["상권_코드"].astype(str).str.strip())
    gdf = gdf[gdf["상권_코드"].isin(valid_codes)].reset_index(drop=True)

    # ── 파트 분할 ─────────────────────────────────────────────────────────────
    if args.total > 1:
        chunk_size = len(gdf) // args.total
        start_idx  = (args.part - 1) * chunk_size
        end_idx    = start_idx + chunk_size if args.part < args.total else len(gdf)
        gdf = gdf.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"    전체 상권 중 {start_idx}~{end_idx-1}번 담당")

    print(f"    대상 상권: {len(gdf):,}개")

    # ── 2. 서울 도로망 로드 ───────────────────────────────────────────────────
    print("\n[2] 서울 도로망 로드 중 (osmnx 캐시 활용)...")
    ox.settings.use_cache   = True
    ox.settings.log_console = False

    bounds = gdf.total_bounds
    buf    = 0.002
    try:
        G = ox.graph_from_bbox(
            (bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf),
            network_type="all", simplify=False,
        )
    except TypeError:
        G = ox.graph_from_bbox(
            north=bounds[3]+buf, south=bounds[1]-buf,
            east=bounds[2]+buf,  west=bounds[0]-buf,
            network_type="all", simplify=False,
        )

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # 상업 도로 엣지만 필터링
    commercial_mask = edges_gdf["highway"].apply(
        lambda hw: get_highway(hw) in COMMERCIAL_HIGHWAYS
    )
    commercial_edges = edges_gdf[commercial_mask]
    print(f"    전체 엣지: {len(edges_gdf):,}개 → 상업 도로 엣지: {len(commercial_edges):,}개")

    # ── 3. 이어받기 준비 ──────────────────────────────────────────────────────
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    existing_count = sum(1 for _ in IMG_DIR.rglob("*.jpg"))
    if existing_count:
        print(f"\n    기존 이미지 {existing_count:,}장 감지 → 이어받기 모드")

    # ── 4. 상권별 이미지 수집 ────────────────────────────────────────────────
    print(f"\n[3] 이미지 수집 시작 ({len(gdf):,}개 상권)...\n")
    new_records     = []
    total_download  = 0
    total_skip      = 0
    total_no_sv     = 0
    total_snap_skip = 0
    total_no_road   = 0

    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="상권"):
        code    = row["상권_코드"]
        area    = float(row["영역_면적"])
        polygon = row["geometry"]
        n_pts   = n_sample_points(area)

        # ── 도로 엣지 샘플링 (층화: minor 우선 + major 최대 2개) ──────────
        candidates = sample_edges_in_polygon(commercial_edges, nodes_gdf, polygon)

        if not candidates:
            candidates = sample_edges_in_polygon(edges_gdf, nodes_gdf, polygon)

        if not candidates:
            total_no_road += 1
            continue

        sampled = stratified_sample(candidates, n_pts, MIN_SPREAD_M)
        if not sampled:
            total_no_road += 1
            continue

        img_dir = IMG_DIR / code
        img_dir.mkdir(exist_ok=True)

        for pt_lat, pt_lng, h_a, h_b in sampled:

            # 두 방향 모두 이미 존재하면 포인트 전체 스킵
            lat_s = f"{pt_lat:.6f}"
            lng_s = f"{pt_lng:.6f}"
            path_a = img_dir / f"{lat_s}_{lng_s}_{h_a}.jpg"
            path_b = img_dir / f"{lat_s}_{lng_s}_{h_b}.jpg"
            if path_a.exists() and path_b.exists():
                total_skip += 2
                continue

            # 메타데이터 API → 실제 파노라마 위치 확인 (포인트당 1회)
            pano = get_pano_location(pt_lat, pt_lng)
            if pano is None:
                total_no_sv += 1
                continue

            pano_lat, pano_lng = pano
            snap_d = dist_m(pt_lat, pt_lng, pano_lat, pano_lng)
            if snap_d > MAX_SNAP_M:
                total_snap_skip += 1
                continue

            # 두 방향 모두 다운로드
            for heading, img_path in [(h_a, path_a), (h_b, path_b)]:
                if img_path.exists():
                    total_skip += 1
                    continue

                success = download_sv_image(pano_lat, pano_lng, heading, img_path)
                new_records.append({
                    "상권_코드": code,
                    "filename" : img_path.name,
                    "lat"      : round(pt_lat, 6),
                    "lng"      : round(pt_lng, 6),
                    "heading"  : heading,
                    "success"  : success,
                })
                if success:
                    total_download += 1

                time.sleep(0.05)

    # ── 5. 로그 저장 (누적 append) ───────────────────────────────────────────
    new_df = pd.DataFrame(new_records)
    if not new_df.empty:
        if LOG_PATH.exists():
            old_df  = pd.read_csv(LOG_PATH)
            all_log = pd.concat([old_df, new_df], ignore_index=True)
        else:
            all_log = new_df
        all_log = all_log.drop_duplicates(subset=["상권_코드", "filename"])
        all_log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    # ── 6. 결과 요약 ─────────────────────────────────────────────────────────
    total_files = sum(1 for _ in IMG_DIR.rglob("*.jpg"))
    print(f"\n{'=' * 65}")
    print("수집 완료 요약")
    print(f"  신규 다운로드          : {total_download:,}장")
    print(f"  이미 존재 (스킵)       : {total_skip:,}장")
    print(f"  Street View 없음       : {total_no_sv:,}개 포인트")
    print(f"  스냅 초과 (실내 추정)  : {total_snap_skip:,}개 포인트")
    print(f"  도로 없는 상권         : {total_no_road:,}개")
    print(f"  총 이미지 파일         : {total_files:,}장  →  {IMG_DIR}")
    print(f"  수집 로그              → {LOG_PATH}")
    print(f"{'=' * 65}")
    print("\n다음 단계: python scripts/09_filter_images.py  ← CLIP 필터링")


if __name__ == "__main__":
    main()
