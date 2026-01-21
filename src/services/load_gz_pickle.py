# src/services/load_gz_pickle.py
import os
import gzip
import pickle
from typing import Dict, Literal

Manufacturer = Literal["kia", "hyundai"]

KIA_FEATURES_GZ = os.getenv("KIA_FEATURES_GZ", "kia_features_dict.pkl.gz")
HYUNDAI_FEATURES_GZ = os.getenv("HYUNDAI_FEATURES_GZ", "h_features_dict.pkl.gz")

_features_cache: Dict[str, dict] = {}

def load_gz_pickle(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def get_features_dict(manufacturer: Manufacturer) -> dict:
    """
    manufacturer별 features dict를 최초 1회 로드 후 메모리 캐시 재사용
    """
    if manufacturer in _features_cache:
        return _features_cache[manufacturer]

    path = KIA_FEATURES_GZ if manufacturer == "kia" else HYUNDAI_FEATURES_GZ
    d = load_gz_pickle(path)
    _features_cache[manufacturer] = d
    return d
