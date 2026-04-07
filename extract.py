#!/usr/bin/env python3
"""
Milvus extract tool: pulls data from a Milvus endpoint via query_iterator
and writes to disk (JSON or Parquet) at file:// or s3:// locations.
"""

import json
from pathlib import Path
from typing import Any
from grpc.aio import UsageError
from numpy import extract
from pymilvus.exceptions import ConnectionConfigException
from pymilvus.exceptions import MilvusException
import click
import yaml
import time


# Defaults from spec
DEFAULTS = {
    "database": "default",
    "collection": "",
    "endpoint": "https://localhost:19530",
    "api_key": "root:Milvus",
    "buffer": 2000,
    "max_rows_per_file": 10000,
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


def load_config(extract_config_path: str, connect_config_path: str | None = None) -> dict[str, Any]:
    
    # Load connection credentials from connect_config_path
    if connect_config_path:
        with open(connect_config_path) as f:
            connect_data = yaml.safe_load(f)
        if not connect_data or "connect" not in connect_data:
            raise click.BadParameter(f"YAML must have a root key 'connect'")
        connect_raw = connect_data["connect"] or {}
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
        max_rows_per_file = raw.get("max_rows_per_file", DEFAULTS["max_rows_per_file"])
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
        max_rows_per_file = int(max_rows_per_file)
        if not (1 <= max_rows_per_file <= 100000):
            raise click.BadParameter(
                f"max_rows_per_file must be between 1 and 100000, got {max_rows_per_file}"
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
        "max_rows_per_file": max_rows_per_file,
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
    max_rows_per_file = config["max_rows_per_file"]
    desc = client.describe_collection(collection_name=config["collection"])
    schema = CollectionSchema.construct_from_dict(desc)

    partition_names = client.list_partitions(collection_name=config["collection"])

    total_fetched = 0
    try:
        client.load_collection(collection_name=config["collection"])
    except MilvusException as e:
        click.echo(f"ERROR: Exception raised: {e}, Please check collection name")
        return
    collection_dir = f"col_{config['collection']}_{time.time()}"
    remove_auto_id_field = schema.primary_field.name if schema.primary_field is not None and schema.auto_id == True and schema.primary_field.auto_id == True else None
    for partition_name in partition_names:
        partition_path = f"{base_path}/{collection_dir}/partition/{partition_name}"
        total_this_partition = 0
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

        click.echo(f"Loading data for partition: {partition_name}")
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
                    if remove_auto_id_field:
                        del(row[remove_auto_id_field])
                    writer.append_row(row)
                    total_fetched += 1
                    total_this_partition += 1
                    if total_this_partition % max_rows_per_file == 0:
                        writer.commit()
                        click.echo(f"Wrote {max_rows_per_file} to partition {partition_name} in {partition_path}. Total row count {writer.total_row_count}")
        finally:
            iterator.close()
        if writer:
            writer.commit()
            click.echo(f"Wrote partition {partition_name} to {partition_path} with size {writer.total_row_count}")

    if total_fetched:
        click.echo(f"Wrote {total_fetched} rows total ({export_file_type})")


def _client(endpoint: str, api_key: str, db_name: str = "default"):
    from pymilvus import MilvusClient
    return MilvusClient(uri=endpoint, token=api_key, db_name=db_name)


def _run_list_databases(endpoint: str, api_key: str) -> None:
    try: 
        client = _client(endpoint, api_key)
    except ConnectionConfigException as e1:
        click.echo(f"ERROR: Exception raised: {e1}, Please check endpoint and api_key")
        return
    except MilvusException as e2:
        click.echo(f"ERROR: Exception raised: {e2}, Please check endpoint and api_key")
        return
    except Exception as e:
        click.echo(f"ERROR: Exception connecting to Milvus: {e}")
        return
    for name in client.list_databases():
        click.echo(f"Database: {name}")


def _run_list_collections(endpoint: str, api_key: str, database: str) -> None:
    try: 
        client = _client(endpoint, api_key, db_name=database)
    except ConnectionConfigException as e1:
        click.echo(f"ERROR: Exception raised: {e1}, Please check endpoint and api_key")
        return
    except MilvusException as e2:
        click.echo(f"ERROR: Exception raised: {e2}, Please check endpoint and api_key")
        return
    except Exception as e:
        click.echo(f"ERROR: Exception connecting to Milvus: {e}")
        return
    for name in client.list_collections():
        click.echo(f"Collection: {name}")


def _run_dump_schema(
    endpoint: str, api_key: str, database: str, collection: str, out_dir: Path
) -> None:
    client = _client(endpoint, api_key, db_name=database)
    desc = client.describe_collection(collection_name=collection)
    schema_dir = out_dir / database
    schema_dir.mkdir(parents=True, exist_ok=True)
    path = schema_dir / f"{collection}__schema.json"
    path.write_text(json.dumps(desc, indent=2, default=str))
    click.echo(f"Wrote {path}")
    list_of_partitions = client.list_partitions(collection_name=collection)
    for partition in list_of_partitions:
        partition_desc = client.get_partition_stats(collection_name=collection, partition_name=partition)
        path = schema_dir / f"{collection}__{partition}__part.json"
        path.write_text(json.dumps(partition_desc, indent=2, default=str))
        click.echo(f"Wrote {path}")
    
    


def _run_dump_schema_all(endpoint: str, api_key: str, database: str, out_dir: Path) -> None:
    client = _client(endpoint, api_key, db_name=database)
    schema_dir = out_dir / database
    schema_dir.mkdir(parents=True, exist_ok=True)
    for name in client.list_collections():
        desc = client.describe_collection(collection_name=name)
        path = schema_dir / f"{name}__schema.json"
        path.write_text(json.dumps(desc, indent=2, default=str))
        click.echo(f"Wrote {path}")
        list_of_partitions = client.list_partitions(collection_name=name)
        for partition in list_of_partitions:
            partition_desc = client.get_partition_stats(collection_name=name, partition_name=partition)
            path = schema_dir / f"{name}__{partition}__part.json"
            path.write_text(json.dumps(partition_desc, indent=2, default=str))
            click.echo(f"Wrote {path}")


def _run_dump_indexes(
    endpoint: str, api_key: str, database: str, collection: str, out_dir: Path
) -> None:
    client = _client(endpoint, api_key, db_name=database)
    indexes = client.list_indexes(collection_name=collection)
    indexes_dir = out_dir / database
    indexes_dir.mkdir(parents=True, exist_ok=True)
    index_desc = []
    for index in indexes:
        index_desc.append(str(client.describe_index(collection_name=collection, index_name=index)))
    
    with open(indexes_dir / f"{collection}__indexes.json", "a") as f:
        f.write('\n'.join(index_desc))
    click.echo(f"Wrote {indexes_dir / f"{collection}__indexes.json"}")

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
    api_key = DEFAULTS["api_key"]
    if connect_config_file is not None:
        if extract_config_file is not None:
            raise click.UsageError("Cannot use -i/--connect-config-file and -f/--extract-config-file together")
            
        connect_config = load_config(None, str(connect_config_file))
        endpoint = connect_config["endpoint"]
        api_key = connect_config["api_key"]
        
    if extract_config_file is not None:
        if connect_config_file is not None:
            raise click.UsageError("Cannot use -i/--connect-config-file and -f/--extract-config-file together")
            
        if any(actions):
            raise click.UsageError("actions can only be used with -i/--connect-config-file")
            
        config = load_config(str(extract_config_file), None)
        run_extract(config)
        return
    elif not any(actions):
        raise click.UsageError("Config file -f is required for data extract. Use --help for more available actions")
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
