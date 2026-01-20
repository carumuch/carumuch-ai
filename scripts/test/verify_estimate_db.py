# scripts/test/verify_estimate_db.py
import os
import json
import random
import sqlite3
import argparse
from typing import Optional, Tuple, List


def to_int_money(x: Optional[str]) -> int:
    s = str(x or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None


def calc_from_json(json_path: str) -> Tuple[List[str], int]:
    """
    build_estimate_db.py에서 쓴 로직과 동일해야 합니다.
    - '손해사정후' 있으면 그 내부의 부품가격/공임
    - 없으면 item의 부품가격/공임
    """
    data = load_json(json_path)
    if not isinstance(data, dict):
        return [], 0

    total = 0
    repairs: List[str] = []

    for it in (data.get("수리내역") or []):
        if not isinstance(it, dict):
            continue

        nm = it.get("작업항목 및 부품명")
        if isinstance(nm, str) and nm.strip():
            repairs.append(nm.strip())

        src = it.get("손해사정후") if isinstance(it.get("손해사정후"), dict) else it
        total += to_int_money(src.get("부품가격")) + to_int_money(src.get("공임"))

    return repairs, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join("artifacts", "estimate.db"))
    ap.add_argument("--mfg", choices=["kia", "hyundai", "all"], default="kia")
    ap.add_argument("--kia-dir", default="kia_견적서")
    ap.add_argument("--hyundai-dir", default="hyundai_견적서")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--show", type=int, default=10, help="불일치 샘플 몇 개 출력할지")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    def verify_one(mfg: str, folder: str):
        rows = cur.execute(
            "SELECT image_name, total_cost, list_repair_json FROM estimate_map WHERE manufacturer=?",
            (mfg,),
        ).fetchall()

        if not rows:
            print(f"[{mfg}] DB rows not found.")
            return

        sample = random.sample(rows, min(args.samples, len(rows)))
        mismatch = 0
        shown = 0
        missing_json = 0

        for image_name, db_total, db_list_json in sample:
            json_path = os.path.join(folder, os.path.splitext(image_name)[0] + ".json")
            if not os.path.exists(json_path):
                missing_json += 1
                continue

            file_list, file_total = calc_from_json(json_path)
            db_list = json.loads(db_list_json)

            if int(db_total) != int(file_total) or db_list != file_list:
                mismatch += 1
                if shown < args.show:
                    shown += 1
                    print(f"\n[{mfg}] MISMATCH: {image_name}")
                    print(f"  db_total   : {db_total}")
                    print(f"  file_total : {file_total}")
                    print(f"  db_list_len   : {len(db_list)}")
                    print(f"  file_list_len : {len(file_list)}")
                    # 앞쪽 몇 개만 비교 출력
                    print(f"  db_list_head   : {db_list[:5]}")
                    print(f"  file_list_head : {file_list[:5]}")

        print(f"\n[{mfg}] checked={len(sample)} mismatch={mismatch} missing_json={missing_json}")

    if args.mfg in ("kia", "all"):
        verify_one("kia", args.kia_dir)

    if args.mfg in ("hyundai", "all"):
        verify_one("hyundai", args.hyundai_dir)

    conn.close()


if __name__ == "__main__":
    main()
