"""
Google Street View Static API 이미지 수집 스크립트
------------------------------------------------------
사용법:
    python src/data_collection/streetview.py

설명:
    상권별로 정의된 좌표에서 동서남북 4방향 이미지를 수집합니다.
    API 키는 .env 파일에서 불러옵니다. (코드에 직접 넣지 마세요)
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_STREETVIEW_API_KEY")

# ──────────────────────────────────────────────
# 상권별 샘플링 좌표 (위도, 경도)
# 각 상권당 주요 도로 교차점 기준으로 추후 확장 예정
# ──────────────────────────────────────────────
SANGKWON = {
    "홍대": [
        (37.5563, 126.9236),
        (37.5571, 126.9251),
        (37.5550, 126.9220),
    ],
    "성수": [
        (37.5446, 127.0556),
        (37.5452, 127.0572),
        (37.5438, 127.0545),
    ],
    "강남": [
        (37.4979, 127.0276),
        (37.4990, 127.0290),
        (37.4967, 127.0262),
    ],
}

# 4방향: 북(0), 동(90), 남(180), 서(270)
HEADINGS = [0, 90, 180, 270]
HEADING_NAMES = {0: "N", 90: "E", 180: "S", 270: "W"}

IMAGE_SIZE = "640x640"
FOV = 90           # 화각 (도)
PITCH = 0          # 상하 각도 (0=수평)
DELAY = 0.3        # API 호출 간격 (초) - 과부하 방지

BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "images"


def check_image_exists(lat: float, lng: float) -> bool:
    """해당 좌표에 Street View 이미지가 존재하는지 확인"""
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lng}", "key": API_KEY}
    res = requests.get(url, params=params)
    data = res.json()
    return data.get("status") == "OK"


def fetch_image(lat: float, lng: float, heading: int, save_path: Path) -> bool:
    """이미지 한 장 수집 후 저장"""
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "location": f"{lat},{lng}",
        "size": IMAGE_SIZE,
        "heading": heading,
        "pitch": PITCH,
        "fov": FOV,
        "key": API_KEY,
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        save_path.write_bytes(res.content)
        return True
    return False


def collect_all():
    """전체 상권 이미지 수집 실행"""
    total, success, skipped = 0, 0, 0

    for name, coords in SANGKWON.items():
        save_dir = BASE_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{name}] 수집 시작 — {len(coords)}개 지점 × 4방향")

        for i, (lat, lng) in enumerate(coords):
            # 해당 좌표에 이미지가 있는지 먼저 확인
            if not check_image_exists(lat, lng):
                print(f"  지점 {i+1}: Street View 없음, 건너뜀")
                skipped += 1
                continue

            for heading in HEADINGS:
                filename = f"point{i+1:02d}_{HEADING_NAMES[heading]}.jpg"
                save_path = save_dir / filename

                if save_path.exists():
                    print(f"  {filename}: 이미 존재, 건너뜀")
                    success += 1
                    continue

                ok = fetch_image(lat, lng, heading, save_path)
                status = "저장 완료" if ok else "실패"
                print(f"  {filename}: {status}")
                total += 1
                if ok:
                    success += 1
                time.sleep(DELAY)

    print(f"\n완료: {success}/{total + skipped}장 수집 ({skipped}개 지점 이미지 없음)")


if __name__ == "__main__":
    if not API_KEY:
        print("오류: .env 파일에 GOOGLE_STREETVIEW_API_KEY가 없습니다.")
    else:
        collect_all()
