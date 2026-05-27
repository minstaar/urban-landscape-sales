# -*- coding: utf-8 -*-
"""
test_06_images.py — 도로 엣지 수직 샘플링 방식 검증
─────────────────────────────────────────────────────────────────────────────
[수집 전략]
    POI 기반(OSM 데이터 품질 의존) → 도로 엣지 수직 샘플링으로 변경

    1. 상권 폴리곤 내 상업 도로 엣지 추출
    2. 각 엣지 중점(들)을 샘플 포인트로 사용
    3. 도로에 수직인 두 방향 (±90°) 모두 이미지 수집
       → centroid 휴리스틱 없이, 두 장 모두 저장
       → 07_filter_images.py (CLIP) 에서 상업 전면 방향 자동 선택
    4. 공간 분산 (MIN_SPREAD_M 이상 간격 보장)
    5. 메타데이터 API로 실제 파노라마 위치 확인
       → MAX_SNAP_M 초과 시 실내/지하 스냅으로 간주, 스킵

[방향 결정 근거]
    선행 연구 대부분("at 50-m intervals along roads")은 방향 기준 미명시.
    본 연구는 두 수직 방향 모두 수집 후 CLIP으로 자동 선택 → 방법론적으로
    더 엄밀하며 논문에 명확히 기술 가능.
    (e.g. "The commercially-oriented image was selected via CLIP cosine
    similarity to 'commercial storefront'; the alternative is retained for
    manual inspection.")

[장점 vs POI 기반]
    - OSM 데이터 품질 의존성 없음 → 재현성 높음
    - 도로에 수직 = 항상 건물 전면을 향함 → POI centroid 오차 없음
    - 국제 논문에서 방법론 방어 용이 ("uniform road-edge sampling")

[대상 상권]
    발달상권 3개 / 골목상권 3개 — 총 6개

실행:
    python scripts/test_06_images.py

출력:
    data/images_test/{상권_코드}/{lat}_{lng}_{heading}.jpg
    (각 샘플 포인트당 2장: _A=heading1, _B=heading2)
"""

import math
import os
import time
import warnings

import numpy as np
import requests
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from dotenv import load_dotenv
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 환경변수 ────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_STREETVIEW_API_KEY")

# ── 경로 설정 ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
SHP_PATH = ROOT / "data/raw/boundaries/서울시 상권분석서비스(영역-상권).shp"
IMG_DIR  = ROOT / "data/images_test"

# ── 테스트 대상 상권 ──────────────────────────────────────────────────────────────
TEST_CODES = {
    "3120102": "서교동(홍대) — 발달상권",
    "3120046": "이태원(이태원역) — 발달상권",
    "3120105": "상수역(홍대) — 발달상권",
    "3110009": "자하문터널 — 골목상권",
    "3110017": "정독도서관 — 골목상권",
    "3110006": "청운동 — 골목상권",
}

# ── 샘플링 파라미터 ───────────────────────────────────────────────────────────────
MAX_POINTS        = 15   # 상권당 최대 샘플 포인트 수
MIN_SPREAD_M      = 30   # 포인트 간 최소 거리(m) — 공간 분산
MAX_SNAP_M        = 30   # 파노라마-요청좌표 거리 초과 시 실내 스냅으로 간주, 스킵
SAMPLE_INTERVAL_M = 20   # 긴 엣지 분할 샘플링 간격(m)
MAX_PTS_PER_EDGE  = 2    # 엣지당 최대 포인트 수 — primary 긴 도로 독점 방지

# ── Street View 설정 ─────────────────────────────────────────────────────────────
SV_BASE   = "https://maps.googleapis.com/maps/api/streetview"
SV_META   = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMG_SIZE  = "640x640"
FOV       = 90
PITCH     = -10    # 살짝 아래 → 간판·쇼윈도 중심
SV_RADIUS = 30     # 파노라마 탐색 반경(m) — 좁게 유지해 실내 스냅 방지

# ── 상업 도로 타입 ────────────────────────────────────────────────────────────────
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
    perp1 = round((edge_bearing + 90)  % 360)
    perp2 = round((edge_bearing + 270) % 360)
    return perp1, perp2


def dist_m(lat1, lng1, lat2, lng2):
    """두 좌표 간 거리(m) — 서울 위도 근사"""
    return math.hypot(lat1 - lat2, lng1 - lng2) * 111_000


def spread_sample(pts, n, min_dist_m):
    """
    공간 분산 샘플링
    pts: [(lat, lng, heading_A, heading_B), ...]
    위치(lat, lng)만 기준으로 거리 계산
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


# ══════════════════════════════════════════════════════════════════════════════
# 도로 엣지 샘플링
# ══════════════════════════════════════════════════════════════════════════════
def sample_edges_in_polygon(edges_gdf, nodes_gdf, polygon):
    """
    폴리곤 내 상업 도로 엣지에서 샘플 포인트 + 수직 두 heading 추출

    긴 엣지(> SAMPLE_INTERVAL_M)는 SAMPLE_INTERVAL_M 간격으로 분할 샘플링.
    Returns: [(lat, lng, heading_A, heading_B), ...]
             heading_A = edge_bearing + 90°
             heading_B = edge_bearing + 270°
    """
    candidates = []

    for (u, v, _), edge in edges_gdf.iterrows():
        # 두 노드 모두 존재하는지 확인
        if u not in nodes_gdf.index or v not in nodes_gdf.index:
            continue

        u_node = nodes_gdf.loc[u]
        v_node = nodes_gdf.loc[v]
        u_lat, u_lng = float(u_node.geometry.y), float(u_node.geometry.x)
        v_lat, v_lng = float(v_node.geometry.y), float(v_node.geometry.x)

        # 엣지 중점이 폴리곤 내부인지 확인
        mid_lat = (u_lat + v_lat) / 2
        mid_lng = (u_lng + v_lng) / 2
        if not polygon.contains(Point(mid_lng, mid_lat)):
            continue

        # 엣지 길이에 따라 단일 중점 or 다중 샘플
        edge_len_m = dist_m(u_lat, u_lng, v_lat, v_lng)
        if edge_len_m < SAMPLE_INTERVAL_M:
            sample_fracs = [0.5]                        # 중점 1개
        else:
            n_seg = max(1, int(edge_len_m / SAMPLE_INTERVAL_M))
            sample_fracs = [(i + 0.5) / n_seg for i in range(n_seg)]

        edge_bearing = compute_heading(u_lat, u_lng, v_lat, v_lng)
        h_a, h_b = perp_both(edge_bearing)             # 두 방향

        edge_len_m = dist_m(u_lat, u_lng, v_lat, v_lng)
        if edge_len_m < SAMPLE_INTERVAL_M:
            sample_fracs = [0.5]
        else:
            n_seg = max(1, int(edge_len_m / SAMPLE_INTERVAL_M))
            n_seg = min(n_seg, MAX_PTS_PER_EDGE)
            sample_fracs = [(i + 0.5) / n_seg for i in range(n_seg)]

        for t in sample_fracs:
            pt_lat = u_lat + t * (v_lat - u_lat)
            pt_lng = u_lng + t * (v_lng - u_lng)
            candidates.append((pt_lat, pt_lng, h_a, h_b))

    return candidates


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
    if not API_KEY:
        raise ValueError("GOOGLE_STREETVIEW_API_KEY가 .env에 없습니다.")

    print("=" * 65)
    print("test_06_images.py — 도로 엣지 수직 샘플링 방식")
    print("=" * 65)

    # ── 1. SHP 로드 ──────────────────────────────────────────────────────────
    print("\n[1] Shapefile 로드 중...")
    gdf = gpd.read_file(SHP_PATH, encoding="cp949")
    gdf = gdf.set_crs("EPSG:5181", allow_override=True).to_crs("EPSG:4326")
    gdf = gdf.rename(columns={"TRDAR_CD": "상권_코드", "RELM_AR": "영역_면적"})
    gdf["상권_코드"] = gdf["상권_코드"].astype(str).str.strip()
    gdf = gdf[gdf["상권_코드"].isin(TEST_CODES.keys())].reset_index(drop=True)

    if gdf.empty:
        print("  ⚠ 테스트 상권 코드를 찾을 수 없습니다.")
        return

    matched = set(gdf["상권_코드"].tolist())
    for code, name in TEST_CODES.items():
        tag = "✓" if code in matched else "✗ (없음, 스킵)"
        print(f"  {tag}  [{code}] {name}")
    print(f"\n  수집 대상: {len(gdf)}개 상권")

    # ── 2. OSM 도로망 로드 ───────────────────────────────────────────────────
    print("\n[2] OSM 도로망 로드 중...")
    ox.settings.use_cache   = True
    ox.settings.log_console = False

    bounds = gdf.total_bounds
    buf    = 0.005
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
    print(f"  전체 엣지: {len(edges_gdf):,}개 → 상업 도로 엣지: {len(commercial_edges):,}개")

    # ── 3. 상권별 이미지 수집 ────────────────────────────────────────────────
    print(f"\n[3] 이미지 수집 시작...\n")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    total_saved  = 0
    total_skip   = 0
    total_no_sv  = 0

    for _, row in gdf.iterrows():
        code    = row["상권_코드"]
        label   = TEST_CODES.get(code, code)
        polygon = row["geometry"]
        centroid     = polygon.centroid
        c_lat, c_lng = float(centroid.y), float(centroid.x)

        print(f"  ▶ [{code}] {label}")

        # ── 도로 엣지 샘플링 ──────────────────────────────────────────────
        candidates = sample_edges_in_polygon(
            commercial_edges, nodes_gdf, polygon
        )
        print(f"    엣지 후보 포인트: {len(candidates)}개", end="")

        if not candidates:
            # 골목상권 등 상업 도로가 없는 경우 → 전체 도로로 재시도
            print(f" → 상업 도로 없음, 전체 도로로 재시도")
            candidates = sample_edges_in_polygon(
                edges_gdf, nodes_gdf, polygon
            )
            print(f"    전체 도로 후보: {len(candidates)}개", end="")

        # 공간 분산 샘플링 (위치 기준, 포인트당 2장 수집)
        sampled = spread_sample(candidates, MAX_POINTS, MIN_SPREAD_M)
        print(f" → 공간분산 후 {len(sampled)}개 포인트 (최대 {len(sampled)*2}장)")

        if not sampled:
            print(f"    ⚠ 샘플링 실패 (폴리곤 내 도로 없음)")
            continue

        img_dir = IMG_DIR / code
        img_dir.mkdir(exist_ok=True)

        for pt_lat, pt_lng, h_a, h_b in sampled:

            # 메타데이터로 실제 파노라마 위치 확인 (위치당 1회 호출)
            pano = get_pano_location(pt_lat, pt_lng)

            if pano is None:
                print(f"    ({pt_lat:.5f}, {pt_lng:.5f}) → SV 없음")
                total_no_sv += 1
                continue

            pano_lat, pano_lng = pano
            snap_d = dist_m(pt_lat, pt_lng, pano_lat, pano_lng)

            if snap_d > MAX_SNAP_M:
                print(f"    ({pt_lat:.5f}, {pt_lng:.5f}) → "
                      f"스냅 {snap_d:.0f}m 초과, 스킵")
                total_skip += 1
                continue

            print(f"    ({pt_lat:.5f}, {pt_lng:.5f}) | "
                  f"파노: ({pano_lat:.5f}, {pano_lng:.5f}) | "
                  f"스냅: {snap_d:.0f}m | headings: {h_a}° / {h_b}°")

            # 두 방향 모두 다운로드
            for heading, tag in [(h_a, "A"), (h_b, "B")]:
                img_path = img_dir / f"{pt_lat:.6f}_{pt_lng:.6f}_{heading}.jpg"
                if img_path.exists():
                    print(f"      [{tag}] → 이미 존재 (스킵)")
                    continue

                ok     = download_sv_image(pano_lat, pano_lng, heading, img_path)
                status = "✓ 저장" if ok else "✗ 실패"
                print(f"      [{tag}] {heading}° → {status}  ({img_path.name})")
                if ok:
                    total_saved += 1
                time.sleep(0.1)

    # ── 4. 요약 ──────────────────────────────────────────────────────────────
    saved_files = list(IMG_DIR.rglob("*.jpg"))
    print(f"\n{'=' * 65}")
    print(f"수집 완료")
    print(f"  저장: {total_saved}장 | SV 없음: {total_no_sv}개 | 스냅 초과: {total_skip}개")
    print(f"  총 파일: {len(saved_files)}장  →  {IMG_DIR}")
    print(f"  ※ 각 포인트당 2장(A/B방향) 수집 → CLIP 필터링으로 상업 전면 선택")
    print(f"{'=' * 65}")
    print("\n다음 단계:")
    print("  1. data/images_test/ 폴더에서 수집 결과 확인")
    print("  2. python scripts/07_filter_images.py  ← CLIP 필터링 (방향 자동 선택)")
    print("  3. 확인 후 → python scripts/06_collect_images.py 전체 852 상권 수집")


if __name__ == "__main__":
    main()
