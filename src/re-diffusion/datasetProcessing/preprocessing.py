import os
import json
from datasets import load_dataset

# -------------------------
# CONFIG
# -------------------------
OUT_DIR = "src/datasetProcessing/data"
MIN_AESTHETIC_SCORE = 5.0
SHARD_SIZE = 20_000

# -------------------------
# DATASET FILTER
# -------------------------
def filtered_laion(dataset):
    for s in dataset:
        if s.get("AESTHETIC_SCORE", 0.0) >= MIN_AESTHETIC_SCORE:
            url = s.get("URL")
            if url is None:
                continue

            yield {
                "url": url,
                "caption": s.get("TEXT", ""),
                "width": s.get("WIDTH"),
                "height": s.get("HEIGHT"),
                "aesthetic": s.get("AESTHETIC_SCORE"),
            }

# -------------------------
# SHARD WRITER (PURE METADATA)
# -------------------------
def write_metadata_shards(dataset):
    os.makedirs(OUT_DIR, exist_ok=True)

    shard_id = 0
    count = 0
    total_written = 0
    total_seen = 0

    shard_path = os.path.join(OUT_DIR, f"meta-{shard_id:05d}.jsonl")
    f = open(shard_path, "w", buffering=1024 * 1024)

    for meta in dataset:
        total_seen += 1

        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        count += 1
        total_written += 1

        if count >= SHARD_SIZE:
            f.close()
            shard_id += 1
            count = 0
            shard_path = os.path.join(OUT_DIR, f"meta-{shard_id:05d}.jsonl")
            f = open(shard_path, "w", buffering=1024 * 1024)

        if total_written % 10_000 == 0:
            print(f"[written {total_written}]")

    f.close()

    print("===================================")
    print("Done.")
    print(f"Total seen:     {total_seen}")
    print(f"Total written:  {total_written}")
    print(f"Shards created: {shard_id + 1}")
    print("===================================")

# -------------------------
# MAIN
# -------------------------
def main():
    print("Loading LAION dataset (streaming, metadata only)...")

    dataset = load_dataset(
        "laion/aesthetics_v2_4.75",
        split="train",
        streaming=True,
    )

    dataset = filtered_laion(dataset)

    print("Writing pure-metadata shards (no network calls)...")
    write_metadata_shards(dataset)

if __name__ == "__main__":
    main()
