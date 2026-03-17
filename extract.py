#!/usr/bin/env python3
"""
Milvus extract tool: pulls data from a Milvus endpoint via query_iterator
and writes to disk (JSON or Parquet) at file:// or s3:// locations.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from pymilvus.exceptions import ConnectionConfigException
from pymilvus.exceptions import MilvusException
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
    "export_file_type": "PARQUET",
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
EXTRACT_CONFIG_ROOT = "extract"
CONNECT_CONFIG_ROOT = "connect"


def load_config(extract_config_path: str | None, connect_config_path: str | None = None) -> dict[str, Any]:
    """Load and validate configuration from YAML files.

    Exactly one of extract_config_path or connect_config_path must be provided.
    - connect_config_path: returns only connection credentials (endpoint + api_key).
    - extract_config_path: returns full extraction config including output, buffer, etc.

    Raises click.BadParameter if neither path is provided or if validation fails.
    """
    if not extract_config_path and not connect_config_path:
        raise click.BadParameter("Either extract_config_path or connect_config_path must be provided")

    # Load connection credentials from connect_config_path
    if connect_config_path:
        with open(connect_config_path) as f:
            connect_data = yaml.safe_load(f)
        if not connect_data or CONNECT_CONFIG_ROOT not in connect_data:
            raise click.BadParameter(f"YAML must have a root key '{CONNECT_CONFIG_ROOT}'")
        connect_raw = connect_data[CONNECT_CONFIG_ROOT] or {}
        endpoint = connect_raw.get("endpoint", DEFAULTS["endpoint"])
        api_key = connect_raw.get("api_key", DEFAULTS["api_key"])

        if not isinstance(endpoint, str):
            raise click.BadParameter("endpoint must be a string")
        if not isinstance(api_key, str):
            raise click.BadParameter("api_key must be a string")
        return {
        "endpoint": endpoint,
        "api_key": api_key
        }

    # Apply defaults and type checks
    # Load connection credentials from extract_config
    if extract_config_path:
        with open(extract_config_path) as f:
            data = yaml.safe_load(f)
        if not data or EXTRACT_CONFIG_ROOT not in data:
            raise click.BadParameter(f"YAML must have a root key '{EXTRACT_CONFIG_ROOT}'")
        raw = data[EXTRACT_CONFIG_ROOT] or {}
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
        if not database:
            raise click.BadParameter("database must not be empty")
        if not isinstance(collection, str):
            raise click.BadParameter("collection must be a string")
        if not collection:
            raise click.BadParameter("collection must not be empty")
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

    try:
        client = MilvusClient(
            uri=config["endpoint"],
            token=config["api_key"],
            db_name=config["database"],
        )
    except (ConnectionConfigException, MilvusException) as e:
        click.echo(f"ERROR: Failed to connect to Milvus: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"ERROR: Failed to connect to Milvus: {e}")
        sys.exit(1)

    out = config["output_location"]
    if not out:
        raise click.BadParameter("output_location is mandatory for export")

    is_s3 = out.strip().startswith("s3://")
    buffer_size = config["buffer"]
    export_file_type = config["export_file_type"]

    # Base path for output (no scheme, no trailing slash)
    if is_s3:
        base_path = out.removeprefix("s3://").strip().rstrip("/") or "export"
        # If first path segment is the bucket (storage_root), remove it from remote_path
        storage_root = config["cloud_storage_params"]["storage_root"]
        parts = base_path.split("/")
        if parts and parts[0] == storage_root:
            base_path = "/".join(parts[1:]).rstrip("/") or "export"
    else:
        # Strip the "file://" scheme prefix without adding an extra slash.
        # e.g. "file:///tmp/export" -> "/tmp/export", "file://tmp/export" -> "tmp/export"
        base_path = out.removeprefix("file://").strip().rstrip("/") or "export"

    # Schema required for all bulk writers (Local and Remote, JSON and PARQUET)
    from pymilvus import CollectionSchema
    try:
        desc = client.describe_collection(collection_name=config["collection"])
        schema = CollectionSchema.construct_from_dict(desc)
        partition_names = client.list_partitions(collection_name=config["collection"])
    except MilvusException as e:
        click.echo(f"ERROR: Failed to read collection '{config['collection']}': {e}")
        sys.exit(1)

    total_written = 0

    try:
        client.load_collection(collection_name=config["collection"])
    except MilvusException as e:
        click.echo(f"ERROR: Failed to load collection '{config['collection']}': {e}")
        sys.exit(1)
    # Use a human-readable timestamp for the output directory name
    # e.g. col_my_collection_20260317_153012
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_dir = f"col_{config['collection']}_{timestamp}"
    remove_auto_id_field = schema.primary_field.name if schema.primary_field is not None and schema.auto_id == True and schema.primary_field.auto_id == True else None
    for partition_name in partition_names:
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

        # Use a single query_iterator per partition. The iterator internally manages
        # its own cursor/offset, so we just call next() until it returns an empty batch.
        # No outer offset loop is needed -- that would cause double-pagination and
        # potentially skip or duplicate rows.
        iterator = client.query_iterator(
            collection_name=config["collection"],
            batch_size=buffer_size,
            filter=config["filter_expression"] or "",
            output_fields=["*"],
            partition_names=[partition_name],
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for hit in batch:
                    row = hit.to_dict() if hasattr(hit, "to_dict") else dict(hit)
                    # Safely remove auto-id field if present; pop() avoids KeyError
                    # when the field is unexpectedly absent from the row dict.
                    if remove_auto_id_field:
                        row.pop(remove_auto_id_field, None)
                    writer.append_row(row)
                    total_written += 1
        finally:
            iterator.close()

        # Always commit, even for empty partitions — this preserves the source
        # partition structure in the output so that the destination side can
        # recreate the same partitions on import.
        if writer:
            writer.commit()
            click.echo(f"Wrote partition {partition_name} to {partition_path} with size {writer.total_row_count}")

    click.echo(f"Wrote {total_written} rows total ({export_file_type})")


def _client(endpoint: str, api_key: str, db_name: str = "default"):
    from pymilvus import MilvusClient
    return MilvusClient(uri=endpoint, token=api_key, db_name=db_name)


def _run_list_databases(endpoint: str, api_key: str) -> None:
    try:
        client = _client(endpoint, api_key)
        databases = client.list_databases()
    except (ConnectionConfigException, MilvusException) as e:
        click.echo(f"ERROR: {e}")
        return
    except Exception as e:
        click.echo(f"ERROR: Exception connecting to Milvus: {e}")
        return
    for name in databases:
        click.echo(f"Database: {name}")


def _run_list_collections(endpoint: str, api_key: str, database: str) -> None:
    try:
        client = _client(endpoint, api_key, db_name=database)
        collections = client.list_collections()
    except (ConnectionConfigException, MilvusException) as e:
        click.echo(f"ERROR: {e}")
        return
    except Exception as e:
        click.echo(f"ERROR: Exception connecting to Milvus: {e}")
        return
    for name in collections:
        click.echo(f"Collection: {name}")


def _run_dump_schema(
    endpoint: str, api_key: str, database: str, collection: str, out_dir: Path
) -> None:
    try:
        client = _client(endpoint, api_key, db_name=database)
        desc = client.describe_collection(collection_name=collection)
    except (ConnectionConfigException, MilvusException) as e:
        click.echo(f"ERROR: {e}")
        return
    except Exception as e:
        click.echo(f"ERROR: Exception connecting to Milvus: {e}")
        return
    schema_dir = out_dir / database
    schema_dir.mkdir(parents=True, exist_ok=True)
    path = schema_dir / f"{collection}__schema.json"
    path.write_text(json.dumps(desc, indent=2, default=str))
    click.echo(f"Wrote {path}")


def _run_dump_schema_all(endpoint: str, api_key: str, database: str, out_dir: Path) -> None:
    try:
        client = _client(endpoint, api_key, db_name=database)
        collections = client.list_collections()
    except (ConnectionConfigException, MilvusException) as e:
        click.echo(f"ERROR: {e}")
        return
    except Exception as e:
        click.echo(f"ERROR: Exception connecting to Milvus: {e}")
        return
    schema_dir = out_dir / database
    schema_dir.mkdir(parents=True, exist_ok=True)
    for name in collections:
        try:
            desc = client.describe_collection(collection_name=name)
        except MilvusException as e:
            click.echo(f"ERROR: Failed to describe collection '{name}': {e}, skipping")
            continue
        path = schema_dir / f"{name}__schema.json"
        path.write_text(json.dumps(desc, indent=2, default=str))
        click.echo(f"Wrote {path}")


def _run_dump_indexes(
    endpoint: str, api_key: str, database: str, collection: str, out_dir: Path
) -> None:
    try:
        client = _client(endpoint, api_key, db_name=database)
        indexes = client.list_indexes(collection_name=collection)
    except (ConnectionConfigException, MilvusException) as e:
        click.echo(f"ERROR: {e}")
        return
    except Exception as e:
        click.echo(f"ERROR: Exception connecting to Milvus: {e}")
        return
    indexes_dir = out_dir / database
    indexes_dir.mkdir(parents=True, exist_ok=True)
    # Collect index descriptions as dicts so we can write valid JSON.
    # Previous code used str() which produces Python repr, not JSON.
    # Also use write mode "w" instead of append "a" to avoid corrupted
    # output from repeated runs.
    index_desc = []
    for index in indexes:
        try:
            index_desc.append(client.describe_index(collection_name=collection, index_name=index))
        except MilvusException as e:
            click.echo(f"ERROR: Failed to describe index '{index}': {e}, skipping")
            continue
    path = indexes_dir / f"{collection}__indexes.json"
    path.write_text(json.dumps(index_desc, indent=2, default=str))
    click.echo(f"Wrote {path}")

@click.command()
@click.option(
    "-d",
    "database",
    "--database",
    default=None,
    help="Database name (default: None).",
)
@click.option(
    "-c",
    "collection",
    "--collection",
    default=None,
    help="Collection name (default: None).",
)
@click.option(
    "-f",
    "extract_config_file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML config file (required for extract).",
)
@click.option(
    "-i",
    "connect_config_file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML Milvus connection credentials file (for actions other than data extract)",
)
@click.option("--list-databases", is_flag=True, help="Print all database names.")
@click.option("--list-collections", is_flag=True, help="Print all collection names for the given database.")
@click.option(
    "--dump-schema",
    is_flag=True,
    help="Write given collection schema as JSON in schema_dir/dbname/ Use with -d/--database and -c/--collection",
)
@click.option(
    "--dump-schema-all",
    is_flag=True,
    help="Write all collection schemas as JSON in schema_dir/dbname/ Use with -d/--database",
)
@click.option(
    "--dump-indexes",
    is_flag=True,
    help="Write given collection's indexes as JSON in schema_dir/dbname/ Use with -d/--database and -c/--collection"
)
@click.option(
    "--schema-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Directory for schema JSON files (default: current directory).",
)
def main(
    database: str,
    collection: str,
    extract_config_file: Path | None,
    connect_config_file: Path | None,
    list_databases: bool,
    list_collections: bool,
    dump_schema: bool,
    dump_schema_all: bool,
    dump_indexes: bool,
    schema_dir: Path,
) -> None:
    """Pull data from a Milvus endpoint via query_iterator and write to disk."""
    actions = [list_databases, list_collections, dump_schema, dump_schema_all, dump_indexes]
    if sum(actions) > 1:
        raise click.UsageError(
            "At most one of --list-databases, --list-collections, --dump-schema, --dump-schema-all, --dump-indexes may be set."
        )

    # Mutual exclusion: -i and -f cannot be used together
    if connect_config_file is not None and extract_config_file is not None:
        raise click.UsageError("Cannot use -i/--connect-config-file and -f/--extract-config-file together")

    # Data extraction mode: requires -f, no action flags allowed
    if extract_config_file is not None:
        if any(actions):
            raise click.UsageError("Action flags (--list-databases, etc.) can only be used with -i/--connect-config-file")
        config = load_config(str(extract_config_file), None)
        run_extract(config)
        return

    # Action mode: requires -i for connection credentials and at least one action flag.
    # Without -i, there is no endpoint to connect to, which previously caused an
    # UnboundLocalError when reaching the action handlers below.
    if not any(actions):
        raise click.UsageError("Config file -f is required for data extract. Use --help for more available actions")
    if connect_config_file is None:
        raise click.UsageError("Connection config -i is required for action flags (--list-databases, etc.)")

    connect_config = load_config(None, str(connect_config_file))
    endpoint = connect_config["endpoint"]
    api_key = connect_config["api_key"]

    if list_databases:
        _run_list_databases(endpoint, api_key)
        return
    if list_collections:
        if database is None:
            database = DEFAULTS["database"]
            click.echo(f"WARN: Database -d db_name not provided on command line. Using default database {DEFAULTS['database']}")
        _run_list_collections(endpoint, api_key, database)
        return
    if dump_schema:
        if not collection or not database:
            raise click.UsageError("--dump-schema requires -c/--collection and -d/--database.")
        _run_dump_schema(endpoint, api_key, database, collection, schema_dir)
        return
    if dump_schema_all:
        if database is None:
            database = DEFAULTS["database"]
            click.echo(f"WARN: Database -d db_name not provided on command line. Using default database {DEFAULTS['database']}")
        _run_dump_schema_all(endpoint, api_key, database, schema_dir)
        return
    if dump_indexes:
        if not collection or not database:
            raise click.UsageError("--dump-indexes requires -c/--collection and -d/--database.")
        _run_dump_indexes(endpoint, api_key, database, collection, schema_dir)
        return
    assert False, "unreachable"


if __name__ == "__main__":
    main()
