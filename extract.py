#!/usr/bin/env python3
"""
Milvus extract tool: pulls data from a Milvus endpoint via query_iterator
and writes to disk (JSON or Parquet) at file:// or s3:// locations.
"""

import json
from pathlib import Path
from typing import Any

import click
import yaml


# Defaults from spec
DEFAULTS = {
    "database": "default",
    "collection": "",
    "endpoint": "https://localhost:19530",
    "api_key": "root:Milvus",
    "buffer": 2000,
    "filter_expression": "",
    "export_file_type": "JSON",
    "output_location": "",
}

CLOUD_STORAGE_DEFAULTS = {
    "endpoint": "localhost:9000",
    "access_key": "",
    "secret_key": "",
    "storage_root": "a-bucket",
}

EXPORT_TYPES = ("JSON", "PARQUET")
BUFFER_MIN, BUFFER_MAX = 1, 16384
CONFIG_ROOT = "extract"


def load_config(path: str) -> dict[str, Any]:
    """Load and validate config from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or CONFIG_ROOT not in data:
        raise click.BadParameter(f"YAML must have a root key '{CONFIG_ROOT}'")
    raw = data[CONFIG_ROOT] or {}

    # Apply defaults and type checks
    database = raw.get("database", DEFAULTS["database"])
    collection = raw.get("collection", DEFAULTS["collection"])
    endpoint = raw.get("endpoint", DEFAULTS["endpoint"])
    api_key = raw.get("api_key", DEFAULTS["api_key"])
    buffer = raw.get("buffer", DEFAULTS["buffer"])
    filter_expression = raw.get("filter_expression", DEFAULTS["filter_expression"])
    export_file_type = raw.get("export_file_type", DEFAULTS["export_file_type"])
    output_location = raw.get("output_location", DEFAULTS["output_location"])

    if not isinstance(database, str):
        raise click.BadParameter("database must be a string")
    if not isinstance(collection, str):
        raise click.BadParameter("collection must be a string")
    if not isinstance(endpoint, str):
        raise click.BadParameter("endpoint must be a string")
    if not isinstance(api_key, str):
        raise click.BadParameter("api_key must be a string")
    if not isinstance(filter_expression, str):
        raise click.BadParameter("filter_expression must be a string")
    if not isinstance(output_location, str):
        raise click.BadParameter("output_location must be a string")

    buffer = int(buffer)
    if not (BUFFER_MIN <= buffer <= BUFFER_MAX):
        raise click.BadParameter(
            f"buffer must be between {BUFFER_MIN} and {BUFFER_MAX}, got {buffer}"
        )

    export_file_type = str(export_file_type).upper()
    if export_file_type not in EXPORT_TYPES:
        raise click.BadParameter(
            f"export_file_type must be one of {list(EXPORT_TYPES)}, got {export_file_type}"
        )

    raw_csp = raw.get("cloud_storage_params") or {}
    csp_endpoint = raw_csp.get("endpoint", CLOUD_STORAGE_DEFAULTS["endpoint"])
    csp_access_key = raw_csp.get("access_key", CLOUD_STORAGE_DEFAULTS["access_key"])
    csp_secret_key = raw_csp.get("secret_key", CLOUD_STORAGE_DEFAULTS["secret_key"])
    csp_storage_root = raw_csp.get("storage_root", CLOUD_STORAGE_DEFAULTS["storage_root"])
    if not isinstance(csp_endpoint, str):
        raise click.BadParameter("cloud_storage_params.endpoint must be a string")
    if not isinstance(csp_access_key, str):
        raise click.BadParameter("cloud_storage_params.access_key must be a string")
    if not isinstance(csp_secret_key, str):
        raise click.BadParameter("cloud_storage_params.secret_key must be a string")
    if not isinstance(csp_storage_root, str):
        raise click.BadParameter("cloud_storage_params.storage_root must be a string")
    if output_location.strip().startswith("s3://"):
        if not csp_access_key:
            raise click.BadParameter("cloud_storage_params.access_key is mandatory for s3:// output")
        if not csp_secret_key:
            raise click.BadParameter("cloud_storage_params.secret_key is mandatory for s3:// output")
        if not csp_storage_root:
            raise click.BadParameter("cloud_storage_params.storage_root is mandatory for s3:// output")

    cloud_storage_params = {
        "endpoint": csp_endpoint,
        "access_key": csp_access_key,
        "secret_key": csp_secret_key,
        "storage_root": csp_storage_root,
    }

    return {
        "database": database,
        "collection": collection,
        "endpoint": endpoint,
        "api_key": api_key,
        "buffer": buffer,
        "filter_expression": filter_expression,
        "export_file_type": export_file_type,
        "output_location": output_location,
        "cloud_storage_params": cloud_storage_params,
    }


def run_extract(config: dict[str, Any]) -> None:
    """Connect to Milvus, iterate with query_iterator per partition, and write output."""
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=config["endpoint"],
        token=config["api_key"],
        db_name=config["database"],
    )

    out = config["output_location"]
    if not out:
        raise click.BadParameter("output_location is mandatory for export")

    is_s3 = out.strip().startswith("s3://")
    buffer_size = config["buffer"]
    export_file_type = config["export_file_type"]

    # Base path for output (no scheme, no trailing slash)
    if is_s3:
        base_path = out.replace("s3://", "").strip().rstrip("/") or "export"
        # If first path segment is the bucket (storage_root), remove it from remote_path
        storage_root = config["cloud_storage_params"]["storage_root"]
        parts = base_path.split("/")
        if parts and parts[0] == storage_root:
            base_path = "/".join(parts[1:]).rstrip("/") or "export"
    else:
        base_path = out.replace("file://", "/", 1).strip().rstrip("/") or "export"

    # Schema required for all bulk writers (Local and Remote, JSON and PARQUET)
    from pymilvus import CollectionSchema
    desc = client.describe_collection(collection_name=config["collection"])
    schema = CollectionSchema.construct_from_dict(desc)

    partition_names = client.list_partitions(collection_name=config["collection"])
    total_written = 0

    for partition_name in partition_names:
        collection_dir = f"col_{config['collection']}"
        partition_path = f"{base_path}/{collection_dir}/partition/{partition_name}"
        writer = None

        if is_s3:
            from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType
            csp = config["cloud_storage_params"]
            connect_param = RemoteBulkWriter.S3ConnectParam(
                bucket_name=csp["storage_root"],
                endpoint=csp["endpoint"],
                access_key=csp["access_key"],
                secret_key=csp["secret_key"],
            )
            file_type = (
                BulkFileType.JSON if export_file_type == "JSON" else BulkFileType.PARQUET
            )
            writer = RemoteBulkWriter(
                schema=schema,
                remote_path=partition_path,
                connect_param=connect_param,
                file_type=file_type,
            )
        else:
            from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType
            Path(partition_path).mkdir(parents=True, exist_ok=True)
            file_type = (
                BulkFileType.JSON if export_file_type == "JSON" else BulkFileType.PARQUET
            )
            writer = LocalBulkWriter(
                schema=schema,
                local_path=partition_path,
                file_type=file_type,
            )

        offset = 0
        while True:
            iterator = client.query_iterator(
                collection_name=config["collection"],
                batch_size=buffer_size,
                offset=offset,
                filter=config["filter_expression"] or "",
                output_fields=["*"],
                partition_names=[partition_name],
            )
            count_this_round = 0
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    for hit in batch:
                        row = hit.to_dict() if hasattr(hit, "to_dict") else dict(hit)
                        writer.append_row(row)
                        total_written += 1
                    count_this_round += len(batch)
            finally:
                iterator.close()
            if count_this_round < buffer_size:
                break
            offset += count_this_round

        if writer:
            writer.commit()
            click.echo(f"Wrote partition {partition_name} to {partition_path}")

    if total_written:
        click.echo(f"Wrote {total_written} rows total ({export_file_type})")


def _client(endpoint: str, api_key: str, db_name: str = "default"):
    from pymilvus import MilvusClient
    return MilvusClient(uri=endpoint, token=api_key, db_name=db_name)


def _run_list_databases(endpoint: str, api_key: str) -> None:
    client = _client(endpoint, api_key)
    for name in client.list_databases():
        click.echo(name)


def _run_list_collections(endpoint: str, api_key: str, database: str) -> None:
    client = _client(endpoint, api_key, db_name=database)
    for name in client.list_collections():
        click.echo(name)


def _run_dump_schema(
    endpoint: str, api_key: str, database: str, collection: str, out_dir: Path
) -> None:
    client = _client(endpoint, api_key, db_name=database)
    desc = client.describe_collection(collection_name=collection)
    schema_dir = out_dir / database
    schema_dir.mkdir(parents=True, exist_ok=True)
    path = schema_dir / f"{collection}.json"
    path.write_text(json.dumps(desc, indent=2, default=str))
    click.echo(f"Wrote {path}")


def _run_dump_schema_all(endpoint: str, api_key: str, database: str, out_dir: Path) -> None:
    client = _client(endpoint, api_key, db_name=database)
    schema_dir = out_dir / database
    schema_dir.mkdir(parents=True, exist_ok=True)
    for name in client.list_collections():
        desc = client.describe_collection(collection_name=name)
        path = schema_dir / f"{name}.json"
        path.write_text(json.dumps(desc, indent=2, default=str))
        click.echo(f"Wrote {path}")


@click.command()
@click.option(
    "-e",
    "endpoint",
    "--endpoint",
    required=True,
    help="Milvus endpoint URL (e.g. https://localhost:19530)",
)
@click.option(
    "-d",
    "database",
    "--database",
    default="default",
    help="Database name (default: default).",
)
@click.option(
    "-c",
    "collection",
    "--collection",
    default="",
    help="Collection name (default: empty).",
)
@click.option(
    "-f",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML config file (required for extract).",
)
@click.option("--list-databases", is_flag=True, help="Print all database names.")
@click.option("--list-collections", is_flag=True, help="Print all collection names for the given database.")
@click.option(
    "--dump-schema",
    is_flag=True,
    help="Dump the given collection schema as JSON in a directory named after the database. Use with -c/--collection to specify the collection.",
)
@click.option(
    "--dump-schema-all",
    is_flag=True,
    help="Dump all collection schemas as JSON in a directory named after the database.",
)
@click.option(
    "--schema-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Directory for schema JSON files (default: current directory).",
)
def main(
    endpoint: str,
    database: str,
    collection: str,
    config_file: Path | None,
    list_databases: bool,
    list_collections: bool,
    dump_schema: bool,
    dump_schema_all: bool,
    schema_dir: Path,
) -> None:
    """Pull data from a Milvus endpoint via query_iterator and write to disk."""
    actions = [list_databases, list_collections, dump_schema, dump_schema_all]
    if sum(actions) > 1:
        raise click.UsageError("At most one of --list-databases, --list-collections, --dump-schema, --dump-schema-all may be set.")
    api_key = DEFAULTS["api_key"]
    if config_file is not None:
        config = load_config(str(config_file))
        api_key = config["api_key"]
        if not any(actions):
            config["endpoint"] = endpoint
            config["database"] = database
            config["collection"] = collection or config["collection"]
            run_extract(config)
            return
    elif not any(actions):
        raise click.UsageError("Config file -f is required for data extract. Use --help for more available actions")
    if list_databases:
        _run_list_databases(endpoint, api_key)
        return
    if list_collections:
        _run_list_collections(endpoint, api_key, database)
        return
    if dump_schema:
        if not collection:
            raise click.UsageError("--dump-schema requires -c/--collection.")
        _run_dump_schema(endpoint, api_key, database, collection, schema_dir)
        return
    if dump_schema_all:
        _run_dump_schema_all(endpoint, api_key, database, schema_dir)
        return
    assert False, "unreachable"


if __name__ == "__main__":
    main()
