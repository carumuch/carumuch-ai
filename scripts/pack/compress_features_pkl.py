# scripts/pack/compress_features_pkl.py
import os
import pickle
import gzip

def size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)

def compress(src: str, dst: str):
    with open(src, "rb") as f:
        obj = pickle.load(f)

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with gzip.open(dst, "wb", compresslevel=6) as gf:
        pickle.dump(obj, gf, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] {src} -> {dst}")
    print(f"  - before: {size_mb(src):.2f} MB")
    print(f"  - after : {size_mb(dst):.2f} MB")

if __name__ == "__main__":
    compress("kia_features_dict.pkl", "kia_features_dict.pkl.gz")
    compress("h_features_dict.pkl", "h_features_dict.pkl.gz")
