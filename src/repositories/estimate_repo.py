# 견적서 데이터 질의 함수
import os
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

ESTIMATE_DB_PATH = os.getenv("ESTIMATE_DB_PATH", "artifacts/estimate.db")

@dataclass
class EstimateNotFoundError(Exception):
    manufacturer: str
    image_name: str

def get_estimate_or_raise(manufacturer: str, image_name: str) -> Tuple[List[str], int]:
    conn = sqlite3.connect(ESTIMATE_DB_PATH)
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT total_cost, list_repair_json FROM estimate_map WHERE manufacturer=? AND image_name=?",
            (manufacturer, image_name),
        ).fetchone()

        if row is None:
            raise EstimateNotFoundError(manufacturer=manufacturer, image_name=image_name)

        total_cost, list_repair_json = row
        return json.loads(list_repair_json), int(total_cost)
    finally:
        conn.close()
