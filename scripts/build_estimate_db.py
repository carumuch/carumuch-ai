# 기존의 견적서 json 파일들을 SQLite 단일 파일로 변경하는 코드
import os
import json
import sqlite3
import argparse
from typing import Tuple, List, Optional


def parse_int_money(s: Optional[str]) -> int:
    """'44,750' / '' / None 같은 문자열을 int로 안전 변환"""
    if s is None:
        return 0
    s = str(s).strip()
    if s == "":
        return 0
    # "10%(112,113)" 같은 케이스는 여기선 안 들어온다고 가정하지만, 혹시 모르면 숫자만 추출
    # 다만 close_cost는 '부품가격','공임'만 사용하므로 보통 '44,750' 형태
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def compute_estimate_from_json(data: dict) -> Tuple[List[str], int]:
    """
    기존 close_cost()와 동일한 규칙:
    - data['수리내역'] 순회
    - item에 '손해사정후' 키가 있으면 그 내부의 '부품가격','공임' 사용
      없으면 item의 '부품가격','공임' 사용
    - total_cost = 부품가격 + 공임 합
    - list_repair = '작업항목 및 부품명' 누적
    """
    total_cost = 0
    list_repair: List[str] = []

    repair_details = data.get("수리내역")
    if not isinstance(repair_details, list):
        return list_repair, total_cost

    for item in repair_details:
        if not isinstance(item, dict):
            continue

        # 작업 항목명
        part_name = item.get("작업항목 및 부품명")
        if isinstance(part_name, str) and part_name.strip():
            list_repair.append(part_name.strip())

        # 비용 추출 (기아 일부 데이터: 손해사정후)
        if "손해사정후" in item and isinstance(item.get("손해사정후"), dict):
            after = item["손해사정후"]
            part_cost = parse_int_money(after.get("부품가격"))
            labor_cost = parse_int_money(after.get("공임"))
        else:
            part_cost = parse_int_money(item.get("부품가격"))
            labor_cost = parse_int_money(item.get("공임"))

        total_cost += (part_cost + labor_cost)

    return list_repair, total_cost


def iter_json_files(folder: str):
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".json"):
                yield os.path.join(root, name)


def image_name_from_json_filename(json_path: str) -> str:
    base = os.path.splitext(os.path.basename(json_path))[0]
    return f"{base}.jpg"


def ensure_schema(conn: sqlite3.Connection):
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # 약 200MB 캐시(환경에 따라 조정)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS estimate_map (
          manufacturer TEXT NOT NULL,
          image_name TEXT NOT NULL,
          total_cost INTEGER NOT NULL,
          list_repair_json TEXT NOT NULL,
          PRIMARY KEY (manufacturer, image_name)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_estimate_map_mfg ON estimate_map(manufacturer);")


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        # 혹시 인코딩 섞였으면 시도
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None


def build_for_manufacturer(conn: sqlite3.Connection, manufacturer: str, folder: str, batch_size: int = 1000):
    files = list(iter_json_files(folder))
    total = len(files)
    print(f"[{manufacturer}] json files: {total} (folder={folder})")

    insert_sql = """
        INSERT OR REPLACE INTO estimate_map (manufacturer, image_name, total_cost, list_repair_json)
        VALUES (?, ?, ?, ?)
    """

    cur = conn.cursor()
    buf = []
    ok = 0
    fail = 0

    for idx, path in enumerate(files, 1):
        data = load_json(path)
        if not isinstance(data, dict):
            fail += 1
            continue

        image_name = image_name_from_json_filename(path)
        list_repair, total_cost = compute_estimate_from_json(data)

        buf.append((manufacturer, image_name, int(total_cost), json.dumps(list_repair, ensure_ascii=False)))
        ok += 1

        if len(buf) >= batch_size:
            cur.executemany(insert_sql, buf)
            conn.commit()
            buf.clear()

        if idx % 2000 == 0 or idx == total:
            print(f"[{manufacturer}] {idx}/{total} processed (ok={ok}, fail={fail})")

    if buf:
        cur.executemany(insert_sql, buf)
        conn.commit()
        buf.clear()

    print(f"[{manufacturer}] done. ok={ok}, fail={fail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kia-dir", default="kia_견적서")
    ap.add_argument("--hyundai-dir", default="hyundai_견적서")
    ap.add_argument("--out", default=os.path.join("artifacts", "estimate.db"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    conn = sqlite3.connect(args.out)
    try:
        ensure_schema(conn)

        if os.path.isdir(args.kia_dir):
            build_for_manufacturer(conn, "kia", args.kia_dir)
        else:
            print(f"[kia] skip (not found): {args.kia_dir}")

        if os.path.isdir(args.hyundai_dir):
            build_for_manufacturer(conn, "hyundai", args.hyundai_dir)
        else:
            print(f"[hyundai] skip (not found): {args.hyundai_dir}")

        # 통계
        cur = conn.cursor()
        cur.execute("SELECT manufacturer, COUNT(*) FROM estimate_map GROUP BY manufacturer")
        rows = cur.fetchall()
        print("[summary] rows:", rows)
        print(f"[summary] db saved: {args.out}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()