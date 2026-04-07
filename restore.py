#!/usr/bin/env python3
"""
Restore Milvus database, collections, and partitions from JSON produced by
extract.py --dump-schema / --dump-schema-all (layout: db_dir/<collection>__schema.json,
<collection>__<partition>__part.json, and <collection>__indexes.json or <collection>_indexes.json).
"""

import json
from pathlib import Path
from typing import Any

import click
import yaml
from pymilvus import CollectionSchema, MilvusClient

CONNECT_DEFAULTS = {
    "endpoint": "https://localhost:19530",
    "api_key": "root:Milvus",
}

# Keys from describe_index / dump that are not index-build parameters
INDEX_DESCRIBE_META_KEYS = frozenset(
    {
        "field_name",
        "index_name",
        "index_type",
        "metric_type",
        "total_rows",
        "indexed_rows",
        "pending_index_rows",
        "state",
        "params",
    }
)


def _load_connect_config(path: Path) -> tuple[str, str]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "connect" not in data:
        raise click.BadParameter("YAML must have a root key 'connect'")
    raw: dict[str, Any] = data["connect"] or {}
    endpoint = raw.get("endpoint", CONNECT_DEFAULTS["endpoint"])
    api_key = raw.get("api_key", CONNECT_DEFAULTS["api_key"])
    if not isinstance(endpoint, str):
        raise click.BadParameter("connect.endpoint must be a string")
    if not isinstance(api_key, str):
        raise click.BadParameter("connect.api_key must be a string")
    return endpoint, api_key


def _collection_from_schema_filename(filename: str) -> str | None:
    if not filename.endswith("__schema.json"):
        return None
    return filename[: -len("__schema.json")]


def _parse_partition_filename(
    filename: str, collection_names: list[str]
) -> tuple[str, str] | None:
    if not filename.endswith("__part.json"):
        return None
    middle = filename[: -len("__part.json")]
    for coll in sorted(collection_names, key=len, reverse=True):
        prefix = coll + "__"
        if middle.startswith(prefix):
            return coll, middle[len(prefix) :]
    return None


def _indexes_path_for_collection(database_dir: Path, collection: str) -> Path | None:
    for suffix in ("__indexes.json"):
        p = database_dir / f"{collection}{suffix}"
        if p.is_file():
            return p
    return None


def _coerce_index_scalar(v: Any) -> Any:
    if isinstance(v, str) and v.isdigit():
        return int(v)
    if isinstance(v, str) and v.count(".") == 1:
        left, right = v.split(".", 1)
        if left.isdigit() and right.isdigit():
            return float(v)
    return v


def _append_index_entry_to_params(index_params: Any, entry: dict[str, Any]) -> None:
    field_name = entry["field_name"]
    index_type = entry["index_type"]
    index_name = entry.get("index_name") or ""
    metric_type = entry.get("metric_type")
    if metric_type is None:
        metric_type = ""

    merge: dict[str, Any] = {}
    if isinstance(entry.get("params"), dict):
        merge.update(
            {k: _coerce_index_scalar(v) for k, v in entry["params"].items()}
        )
    for k, v in entry.items():
        if k in INDEX_DESCRIBE_META_KEYS or k in merge:
            continue
        merge[k] = _coerce_index_scalar(v)

    index_params.add_index(
        field_name=field_name,
        index_type=index_type,
        index_name=index_name,
        metric_type=metric_type,
        params=merge,
    )


def _restore_collection_indexes(
    client: MilvusClient, collection: str, indexes_path: Path
) -> None:
    raw = json.loads(indexes_path.read_text())
    if not isinstance(raw, list):
        raise click.ClickException(
            f"{indexes_path.name} must be a JSON array of index objects"
        )
    existing = set(client.list_indexes(collection_name=collection))
    index_params = MilvusClient.prepare_index_params()
    to_add = False
    for item in raw:
        if not isinstance(item, dict):
            continue
        iname = item.get("index_name") or ""
        if iname in existing:
            click.echo(
                f"Index {iname!r} on collection {collection!r} already exists, skipping"
            )
            continue
        _append_index_entry_to_params(index_params, item)
        to_add = True
    if to_add:
        client.create_index(
            collection_name=collection, index_params=index_params
        )
        click.echo(f"Created indexes on collection {collection!r} from {indexes_path.name}")


def _restore_collections(
    client: MilvusClient, database_dir: Path, db_name: str
) -> None:
    json_files = [p for p in database_dir.iterdir() if p.is_file() and p.suffix == ".json"]
    schema_paths = sorted(
        p
        for p in json_files
        if _collection_from_schema_filename(p.name) is not None
    )
    if not schema_paths:
        raise click.ClickException(
            f"No *__schema.json files under {database_dir.resolve()}"
        )

    collection_names = [
        _collection_from_schema_filename(p.name) for p in schema_paths
    ]
    assert None not in collection_names

    dbs = client.list_databases()
    if db_name not in dbs:
        client.create_database(db_name=db_name)
        click.echo(f"Created database {db_name!r}")
    else:
        click.echo(f"Database {db_name!r} already exists")
    client.using_database(db_name)

    for path in schema_paths:
        coll_name = _collection_from_schema_filename(path.name)
        assert coll_name is not None
        desc = json.loads(path.read_text())
        json_name = desc.get("collection_name")
        if json_name is not None and json_name != coll_name:
            click.echo(
                f"WARN: {path.name} has collection_name {json_name!r} but filename implies "
                f"{coll_name!r}; using filename for create_collection and partition matching"
            )
        if client.has_collection(coll_name):
            click.echo(f"Collection {coll_name!r} already exists, skipping")
            continue
        schema = CollectionSchema.construct_from_dict(desc)
        client.create_collection(collection_name=coll_name, schema=schema)
        click.echo(f"Created collection {coll_name!r}")

    part_paths = sorted(p for p in json_files if p.name.endswith("__part.json"))
    for path in part_paths:
        parsed = _parse_partition_filename(path.name, list(collection_names))
        if parsed is None:
            raise click.ClickException(
                f"Could not match partition file {path.name!r} to a collection "
                f"(known collections: {collection_names})"
            )
        coll_name, partition_name = parsed
        if partition_name == "_default":
            click.echo(
                f"Skipping partition {partition_name!r} on {coll_name!r} (implicit default)"
            )
            continue
        if not client.has_collection(coll_name):
            raise click.ClickException(
                f"Partition file {path.name!r} references collection {coll_name!r} "
                "which was not created (missing schema file?)"
            )
        if client.has_partition(coll_name, partition_name):
            click.echo(
                f"Partition {partition_name!r} on {coll_name!r} already exists, skipping"
            )
            continue
        client.create_partition(
            collection_name=coll_name, partition_name=partition_name
        )
        click.echo(f"Created partition {partition_name!r} on collection {coll_name!r}")

    for coll_name in collection_names:
        idx_path = _indexes_path_for_collection(database_dir, coll_name)
        if idx_path is None:
            click.echo(
                f"WARN: No {coll_name}__indexes.json or {coll_name}_indexes.json under "
                f"{database_dir.resolve()}; skipping indexes for {coll_name!r}"
            )
            continue
        if not client.has_collection(coll_name):
            click.echo(
                f"WARN: Collection {coll_name!r} missing; skipping indexes from {idx_path.name}"
            )
            continue
        _restore_collection_indexes(client, coll_name, idx_path)


@click.command()
@click.option(
    "--database-dir",
    "database_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory named like the target Milvus database (basename = db name).",
)
@click.option(
    "-i",
    "connect_config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Milvus connection YAML (root key 'connect': endpoint, api_key).",
)
@click.option(
    "--restore-collections",
    is_flag=True,
    help="Create database, collections, partitions, and indexes from JSON in --database-dir.",
)
def main(
    database_dir: Path, connect_config_file: Path, restore_collections: bool
) -> None:
    """Restore Milvus metadata from dumped JSON (use --restore-collections)."""
    if not restore_collections:
        raise click.UsageError("Specify --restore-collections to run restore.")

    uri, token = _load_connect_config(connect_config_file)
    client = MilvusClient(uri=uri, token=token)
    db_name = database_dir.name
    _restore_collections(client, database_dir, db_name)


if __name__ == "__main__":
    main()
