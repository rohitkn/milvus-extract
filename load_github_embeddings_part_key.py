#!/usr/bin/env python3
"""
Load DoyyingFace/github-embeddings-doy from HuggingFace into Milvus.
Creates database write_test1, collection doyying with ``comment_length`` as the
Milvus partition key (server-managed routing), and inserts data.
"""

from datasets import load_dataset
from pymilvus import DataType, MilvusClient

ENDPOINT = "http://localhost:19530"
DB_NAME = "write_test1"
COLLECTION_NAME = "doyying2"
DATASET_NAME = "DoyyingFace/github-embeddings-doy"

VARCHAR_LONG = 65535
VARCHAR_SHORT = 2048


def _create_schema():
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
        partition_key_field="comment_length",
    )
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
    index_params.add_index( #by commenting these lines i was testing if the data is inserted without indexes,
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

    # 2. Create collection with schema and indexes
    schema = _create_schema()
    index_params = _create_index_params()
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    # 3. Load dataset; Milvus routes rows using partition key ``comment_length``
    ds = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    batch_size = 256
    buffer: list[dict] = []
    total = 0

    def flush_batch() -> None:
        nonlocal buffer, total
        if not buffer:
            return
        client.insert(collection_name=COLLECTION_NAME, data=buffer)
        total += len(buffer)
        print(f"Inserted batch of {len(buffer)} rows (total {total})")
        buffer = []

    for i, example in enumerate(ds):
        try:
            row = _row_from_example(dict(example))
        except Exception as e:
            print(f"Skip row {i}: {e}")
            continue
        buffer.append(row)
        if len(buffer) >= batch_size:
            flush_batch()

    flush_batch()

    print("Done.")


if __name__ == "__main__":
    main()
