#!/usr/bin/env python3
"""
Load DoyyingFace/github-embeddings-doy from HuggingFace into Milvus.
Creates database write_test1, collection doyy, partitions by comment_length, and inserts data.
"""

from collections import defaultdict

from datasets import load_dataset
from pymilvus import DataType, MilvusClient

ENDPOINT = "http://localhost:19530"
DB_NAME = "write_test1"
COLLECTION_NAME = "doyy"
DATASET_NAME = "DoyyingFace/github-embeddings-doy"

# Partition names by comment_length range (inclusive min, inclusive max)
# Range starts at 0 so that empty comments (length 0) go into the first partition
# instead of falling through to the catch-all partition.
PARTITION_RANGES = [
    (0, 50, "c_0_50"),
    (51, 100, "c_51_100"),
    (101, 500, "c_101_500"),
    (501, 1000, "c_501_1000"),
]
PARTITION_GT_1000 = "c_gt_1000"

VARCHAR_LONG = 65535
VARCHAR_SHORT = 2048


def _partition_name(comment_length: int) -> str:
    for lo, hi, name in PARTITION_RANGES:
        if lo <= comment_length <= hi:
            return name
    return PARTITION_GT_1000


def _create_schema():
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="html_url", datatype=DataType.VARCHAR, max_length=VARCHAR_SHORT)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=VARCHAR_SHORT)
    schema.add_field(field_name="comments", datatype=DataType.VARCHAR, max_length=VARCHAR_LONG)
    schema.add_field(field_name="body", datatype=DataType.VARCHAR, max_length=VARCHAR_LONG)
    schema.add_field(field_name="comment_length", datatype=DataType.INT64)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=VARCHAR_LONG)
    schema.add_field(field_name="embeddings", datatype=DataType.FLOAT_VECTOR, dim=768)
    return schema


def _create_index_params():
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embeddings",
        index_type="IVF_FLAT",
        index_name="embeddings_index",
        metric_type="COSINE",
        params={"nlist": 1024},
    )
    index_params.add_index(
        field_name="title",
        index_type="Trie",
        index_name="title_index",
        metric_type="",
        params={},
    )
    return index_params


def _row_from_example(example: dict) -> dict:
    """Build a row for insert from a dataset example. Normalize column names and compute comment_length if needed."""
    # Common HuggingFace column name variants
    html_url = (
        example.get("html_url")
        or example.get("url")
        or ""
    )
    title = str(example.get("title", "") or "")
    comments = str(example.get("comments", "") or example.get("comment", "") or "")
    body = str(example.get("body", "") or "")
    text = str(example.get("text", "") or example.get("content", "") or body or comments)
    embeddings = example.get("embeddings") or example.get("embedding") or example.get("vector")
    if embeddings is None:
        raise ValueError("Example has no 'embeddings', 'embedding', or 'vector' field")
    if len(embeddings) != 768:
        raise ValueError(f"Expected embedding dimension 768, got {len(embeddings)}")
    comment_length = example.get("comment_length")
    if comment_length is None:
        comment_length = len(comments)
    else:
        comment_length = int(comment_length)
    return {
        "html_url": html_url[:VARCHAR_SHORT - 1],
        "title": title[:VARCHAR_SHORT - 1],
        "comments": comments[:VARCHAR_LONG - 1],
        "body": body[:VARCHAR_LONG - 1],
        "comment_length": comment_length,
        "text": text[:VARCHAR_LONG - 1],
        "embeddings": embeddings,
    }


def main():
    client = MilvusClient(uri=ENDPOINT, token="root:Milvus")

    # 1. Create database if it does not exist and use it
    dbs = client.list_databases()
    if DB_NAME not in dbs:
        client.create_database(db_name=DB_NAME)
    client.using_database(DB_NAME)

    # 2. Create collection with schema and indexes.
    # WARNING: This drops the existing collection and all its data without confirmation.
    # Guard with a user prompt to prevent accidental data loss on repeated runs.
    schema = _create_schema()
    index_params = _create_index_params()
    if client.has_collection(COLLECTION_NAME):
        answer = input(f"Collection '{COLLECTION_NAME}' already exists. Drop and recreate? [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return
        client.drop_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    # 3. Create partitions by comment_length
    for _, _, name in PARTITION_RANGES:
        client.create_partition(collection_name=COLLECTION_NAME, partition_name=name)
    client.create_partition(collection_name=COLLECTION_NAME, partition_name=PARTITION_GT_1000)

    # 4. Stream dataset and insert in batches per partition.
    # Instead of loading all rows into memory (which can OOM on large datasets),
    # we accumulate rows per partition up to batch_size, then flush immediately.
    ds = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    batch_size = 256

    # Buffers: partition_name -> list of row dicts (max batch_size before flush)
    buffers: dict[str, list[dict]] = defaultdict(list)
    inserted_counts: dict[str, int] = defaultdict(int)

    def _flush(part_name: str) -> None:
        """Insert buffered rows for a partition and clear the buffer."""
        rows = buffers[part_name]
        if not rows:
            return
        client.insert(
            collection_name=COLLECTION_NAME,
            data=rows,
            partition_name=part_name,
        )
        inserted_counts[part_name] += len(rows)
        buffers[part_name] = []

    for i, example in enumerate(ds):
        try:
            row = _row_from_example(dict(example))
        except Exception as e:
            print(f"Skip row {i}: {e}")
            continue
        part = _partition_name(row["comment_length"])
        buffers[part].append(row)
        # Flush when the buffer for this partition reaches batch_size
        if len(buffers[part]) >= batch_size:
            _flush(part)

    # 5. Flush any remaining rows in all partition buffers
    for part_name in list(buffers.keys()):
        _flush(part_name)

    for part_name, count in inserted_counts.items():
        print(f"Inserted {count} rows into partition {part_name}")

    print("Done.")


if __name__ == "__main__":
    main()
