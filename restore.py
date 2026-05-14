#!/usr/bin/env python3
"""
Restore Milvus database, collections, and partitions from JSON produced by
extract.py --dump-schema / --dump-schema-all (layout: db_dir/<collection>__schema.json,
optional <collection>__<partition>__part.json when the collection has no partition key,
and <collection>__indexes.json).

Collections with a partition key in __schema.json omit __part.json dumps; metadata
restore skips create_partition from any stray __part.json for those collections.

Also restore bulk-exported rows with --restore-collection-data from file://, s3://, or gs://
paths matching extract.py bulk writer layout (col_<name>__<timestamp>/partition/...,
using partition/<literal _partition_key>/... when the collection uses a Milvus partition key).
"""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click
import yaml
from dotenv import load_dotenv
from pymilvus import CollectionSchema, MilvusClient

CONNECT_DEFAULTS = {
    "endpoint": "https://localhost:19530",
}

COL_DIR_RE = re.compile(r"^col_(.+)__(\d+(?:\.\d+)?)$")
UUID_DIR_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
INSERT_BATCH = 500
PARTITION_KEY_EXPORT_DIR = "_partition_key"

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


def _load_connect_config(path: Path) -> str:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "connect" not in data:
        raise click.BadParameter("YAML must have a root key 'connect'")
    raw: dict[str, Any] = data["connect"] or {}
    endpoint = raw.get("endpoint", CONNECT_DEFAULTS["endpoint"])
    if not isinstance(endpoint, str):
        raise click.BadParameter("connect.endpoint must be a string")
    return endpoint


def _load_connection_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _cloud_storage_params_from_yaml(data: dict[str, Any]) -> dict[str, str]:
    raw = data.get("cloud_storage_params") or {}
    if not raw and "connect" in data:
        raw = (data.get("connect") or {}).get("cloud_storage_params") or {}
    endpoint = str(raw.get("endpoint", ""))
    storage_root = str(raw.get("storage_root", ""))
    access_key = str(
        raw.get("access_key")
        or os.getenv("CLOUD_STORAGE_ACCESS_KEY")
        or os.getenv("AWS_ACCESS_KEY_ID")
        or ""
    )
    secret_key = str(
        raw.get("secret_key")
        or os.getenv("CLOUD_STORAGE_SECRET_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
        or ""
    )
    return {
        "endpoint": endpoint,
        "storage_root": storage_root,
        "access_key": access_key,
        "secret_key": secret_key,
    }


def _cluster_id_from_zilliz_host(hostname: str) -> str:
    """
    Cluster id is the first hostname label (before the first dot), with a
    trailing '-privatelink' segment removed (case-insensitive on the suffix).
    """
    if not hostname:
        return ""
    first = hostname.split(".")[0]
    lower = first.lower()
    suf = "-privatelink"
    if lower.endswith(suf):
        return first[: -len(suf)]
    return first


def _derive_zilliz_bulk_import_url_and_cluster(milvus_endpoint: str) -> tuple[str | None, str | None]:
    """Derive REST bulk-import base URL and cluster id from Zilliz / Zilliz Cloud hostnames."""
    try:
        parsed = urlparse(milvus_endpoint)
        host_raw = parsed.hostname or ""
        host = host_raw.lower()
    except Exception:
        return None, None
    if not host_raw:
        return None, None

    cluster_id = _cluster_id_from_zilliz_host(host_raw)
    if not cluster_id:
        return None, None

    is_cn = host.endswith(".zillizcloud.com.cn") or host.endswith(".cloud.zilliz.com.cn")

    # Serverless: <cluster>.serverless.<region>.cloud.zilliz.com
    # e.g. in05-5a64a71f44fc942.serverless.aws-eu-central-1.cloud.zilliz.com
    if ".serverless." in host and ".cloud.zilliz.com" in host:
        parts = host.split(".")
        try:
            si = parts.index("serverless")
        except ValueError:
            return None, None
        if si + 1 >= len(parts):
            return None, None
        region = parts[si + 1]
        if is_cn:
            import_url = f"https://api.{region}.cloud.zilliz.com.cn"
        else:
            import_url = f"https://api.{region}.cloud.zilliz.com"
        return import_url, cluster_id

    # Global cluster: <cluster>.global-cluster.vectordb.zillizcloud.com
    if ".global-cluster.vectordb.zillizcloud." in host:
        if is_cn:
            import_url = "https://api.global-cluster.zillizcloud.com.cn"
        else:
            import_url = "https://api.global-cluster.zillizcloud.com"
        return import_url, cluster_id

    # Dedicated / BYOC: <cluster>.<region>.vectordb.zillizcloud.com[:port]
    # Region may be aws-*, gcp-*, az-* (Azure); cluster label may end with -privatelink (stripped above).
    if ".vectordb.zillizcloud." not in host:
        return None, None
    parts = host.split(".")
    try:
        vidx = parts.index("vectordb")
    except ValueError:
        return None, None
    if vidx < 2:
        return None, None
    region = parts[vidx - 1]
    if not (
        region.startswith("aws-")
        or region.startswith("gcp-")
        or region.startswith("az-")
    ):
        return None, None
    if is_cn:
        import_url = f"https://api.{region}.zillizcloud.com.cn"
    else:
        import_url = f"https://api.{region}.zillizcloud.com"
    return import_url, cluster_id


def _bulk_import_api_url(milvus_endpoint: str, data: dict[str, Any]) -> str:
    connect_raw = (data.get("connect") or {}) if isinstance(data, dict) else {}
    explicit = connect_raw.get("import_api_url") or connect_raw.get("bulk_import_url")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.rstrip("/")
    derived, _ = _derive_zilliz_bulk_import_url_and_cluster(milvus_endpoint)
    if derived:
        return derived.rstrip("/")
    parsed = urlparse(milvus_endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    return milvus_endpoint.rstrip("/")


def _bulk_import_cluster_id(milvus_endpoint: str, data: dict[str, Any]) -> str:
    connect_raw = (data.get("connect") or {}) if isinstance(data, dict) else {}
    cid = connect_raw.get("cluster_id")
    if isinstance(cid, str) and cid.strip():
        return cid.strip()
    _, derived = _derive_zilliz_bulk_import_url_and_cluster(milvus_endpoint)
    return derived or ""


def _parse_database_dir_uri(raw: str) -> tuple[str, str, dict[str, Any]]:
    """Return (scheme, db_name, details). details: path for file, bucket+prefix for s3/gs."""
    s = raw.strip()
    lowered = s.lower()
    if lowered.startswith("file://"):
        scheme = "file"
        parsed = urlparse(s)
        path = Path(parsed.path or "/").expanduser().resolve()
        db_name = path.name
        if not db_name:
            raise click.BadParameter("--database-dir must end with a database directory name")
        return scheme, db_name, {"local_path": path}
    if lowered.startswith("s3://"):
        scheme = "s3"
        parsed = urlparse(s if "://" in s else "s3://" + s[5:])
        bucket = parsed.netloc
        prefix = (parsed.path or "").lstrip("/").rstrip("/")
        if not bucket:
            raise click.BadParameter("Invalid s3:// URI (missing bucket)")
        db_name = prefix.split("/")[-1] if prefix else bucket
        if not db_name:
            raise click.BadParameter("--database-dir must include a database path segment")
        return scheme, db_name, {"bucket": bucket, "prefix": prefix}
    if lowered.startswith("gs://"):
        scheme = "gs"
        parsed = urlparse(s if "://" in s else "gs://" + s[5:])
        bucket = parsed.netloc
        prefix = (parsed.path or "").lstrip("/").rstrip("/")
        if not bucket:
            raise click.BadParameter("Invalid gs:// URI (missing bucket)")
        db_name = prefix.split("/")[-1] if prefix else bucket
        if not db_name:
            raise click.BadParameter("--database-dir must include a database path segment")
        return scheme, db_name, {"bucket": bucket, "prefix": prefix}
    raise click.BadParameter(
        "--database-dir must start with file:// then /<localpath> or s3://, or gs:// for --restore-collection-data"
    )


def _parse_col_dir_name(dirname: str) -> tuple[str, float] | None:
    m = COL_DIR_RE.match(dirname)
    if not m:
        return None
    try:
        ts = float(m.group(2))
    except ValueError:
        return None
    return m.group(1), ts


def _pick_latest_col_roots(col_dirs: list[Path]) -> dict[str, Path]:
    """One root Path per collection name (latest timestamp wins)."""
    best: dict[str, tuple[float, Path]] = {}
    for p in col_dirs:
        parsed = _parse_col_dir_name(p.name)
        if parsed is None:
            continue
        name, ts = parsed
        prev = best.get(name)
        if prev is None or ts >= prev[0]:
            best[name] = (ts, p)
    return {k: v[1] for k, v in best.items()}


def _ensure_database(client: MilvusClient, db_name: str) -> None:
    dbs = client.list_databases()
    if db_name not in dbs:
        client.create_database(db_name=db_name)
        click.echo(f"Created database {db_name!r}")
    else:
        click.echo(f"Database {db_name!r} already exists")
    client.using_database(db_name)


def _read_json_or_ndjson_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def _insert_batches(
    client: MilvusClient,
    collection_name: str,
    rows: list[dict[str, Any]],
    partition_name: str,
    *,
    omit_partition_name: bool = False,
) -> None:
    if not rows:
        return
    for i in range(0, len(rows), INSERT_BATCH):
        batch = rows[i : i + INSERT_BATCH]
        if omit_partition_name:
            client.insert(collection_name=collection_name, data=batch)
        else:
            pn = partition_name if partition_name != "_default" else ""
            client.insert(
                collection_name=collection_name,
                data=batch,
                partition_name=pn,
            )
        click.echo(
            f"Inserted {len(batch)} row(s) into {collection_name!r}"
            + (
                ""
                if omit_partition_name
                else (f" partition={partition_name!r}" if partition_name != "_default" else "")
            )
        )


def _restore_collection_data_file(
    client: MilvusClient, db_root: Path, db_name: str
) -> None:
    if not db_root.is_dir():
        raise click.ClickException(f"Not a directory: {db_root}")
    col_dirs = [p for p in db_root.iterdir() if p.is_dir() and _parse_col_dir_name(p.name)]
    if not col_dirs:
        raise click.ClickException(
            f"No col_<collection>__<timestamp> directories under {db_root}"
        )
    roots_by_coll = _pick_latest_col_roots(col_dirs)
    click.echo(
        f"Restoring data from {len(roots_by_coll)} collection(s) "
        f"(latest export per collection): {', '.join(sorted(roots_by_coll))}"
    )
    for collection_name, col_root in roots_by_coll.items():
        if not client.has_collection(collection_name):
            raise click.ClickException(
                f"Collection {collection_name!r} does not exist; create schema first"
            )
        part_root = col_root / "partition"
        if not part_root.is_dir():
            click.echo(f"WARN: No partition/ under {col_root}, skipping {collection_name!r}")
            continue

        desc = _load_schema_desc(db_root, collection_name)
        subdirs = sorted(p.name for p in part_root.iterdir() if p.is_dir())
        has_pk = (desc is not None and _schema_desc_has_partition_key(desc)) or (
            desc is None and subdirs == [PARTITION_KEY_EXPORT_DIR]
        )
        if desc is None and subdirs == [PARTITION_KEY_EXPORT_DIR]:
            click.echo(
                f"[{collection_name}] No __schema.json; inferring partition key from "
                f"partition/{PARTITION_KEY_EXPORT_DIR}/ only"
            )

        if has_pk:
            pk_path = part_root / PARTITION_KEY_EXPORT_DIR
            if not pk_path.is_dir():
                click.echo(
                    f"WARN: Partition key collection {collection_name!r} missing "
                    f"partition/{PARTITION_KEY_EXPORT_DIR}/ under {col_root}, skipping"
                )
                continue
            for uuid_dir in sorted(d for d in pk_path.iterdir() if d.is_dir()):
                if not UUID_DIR_RE.match(uuid_dir.name):
                    continue
                for fpath in sorted(uuid_dir.iterdir()):
                    if not fpath.is_file():
                        continue
                    suf = fpath.suffix.lower()
                    if suf == ".json":
                        rows = _read_json_or_ndjson_file(fpath)
                    elif suf == ".parquet":
                        rows = _read_parquet_rows(fpath)
                    else:
                        continue
                    _insert_batches(
                        client,
                        collection_name,
                        rows,
                        PARTITION_KEY_EXPORT_DIR,
                        omit_partition_name=True,
                    )
        else:
            for part_path in sorted(p for p in part_root.iterdir() if p.is_dir()):
                if part_path.name == PARTITION_KEY_EXPORT_DIR:
                    continue
                partition_name = part_path.name
                if partition_name != "_default":
                    if not client.has_partition(collection_name, partition_name):
                        client.create_partition(
                            collection_name=collection_name,
                            partition_name=partition_name,
                        )
                        click.echo(
                            f"Created partition {partition_name!r} on collection {collection_name!r}"
                        )
                for uuid_dir in sorted(d for d in part_path.iterdir() if d.is_dir()):
                    if not UUID_DIR_RE.match(uuid_dir.name):
                        continue
                    for fpath in sorted(uuid_dir.iterdir()):
                        if not fpath.is_file():
                            continue
                        suf = fpath.suffix.lower()
                        if suf == ".json":
                            rows = _read_json_or_ndjson_file(fpath)
                        elif suf == ".parquet":
                            rows = _read_parquet_rows(fpath)
                        else:
                            continue
                        _insert_batches(
                            client,
                            collection_name,
                            rows,
                            partition_name,
                            omit_partition_name=False,
                        )
        click.echo(f"Finished file restore for collection {collection_name!r}")


def _list_s3_keys_for_restore(
    bucket: str, prefix: str, csp: dict[str, str]
) -> list[str]:
    from minio import Minio

    raw_ep = (csp["endpoint"] or "").strip()
    if "://" in raw_ep:
        p = urlparse(raw_ep)
        ep_host = p.hostname or p.netloc.split("@")[-1]
        secure = p.scheme == "https"
        if p.port:
            ep_host = f"{ep_host}:{p.port}"
    else:
        low = raw_ep.lower()
        if "localhost" in low or low.startswith("127."):
            p = urlparse("http://" + raw_ep)
        else:
            p = urlparse("https://" + raw_ep)
        ep_host = p.hostname or ""
        if p.port:
            ep_host = f"{ep_host}:{p.port}"
        secure = p.scheme == "https"
    mc = Minio(
        ep_host,
        access_key=csp["access_key"],
        secret_key=csp["secret_key"],
        secure=secure,
    )
    pfx = prefix.rstrip("/") + "/"
    keys: list[str] = []
    for obj in mc.list_objects(bucket, prefix=pfx, recursive=True):
        if obj.object_name and not obj.is_dir:
            keys.append(obj.object_name)
    return keys


def _list_gs_keys_for_restore(bucket: str, prefix: str) -> list[str]:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as e:
        raise click.ClickException(
            "gs:// requires google-cloud-storage: pip install google-cloud-storage"
        ) from e
    client = storage.Client()
    pfx = prefix.rstrip("/") + "/"
    return [
        b.name
        for b in client.list_blobs(bucket, prefix=pfx)
        if b.name and not b.name.endswith("/")
    ]


def _filter_keys_latest_export(keys: list[str], prefix: str) -> list[str]:
    """Keep only keys under the newest col_<collection>__<ts> prefix per collection."""
    prefix_norm = prefix.rstrip("/") + "/"
    top_re = re.compile(r"^(col_.+__(\d+(?:\.\d+)?))/")
    best: dict[str, tuple[float, str]] = {}
    for key in keys:
        if not key.startswith(prefix_norm):
            continue
        rel = key[len(prefix_norm) :]
        m = top_re.match(rel)
        if not m:
            continue
        seg, ts_s = m.group(1), m.group(2)
        parsed = _parse_col_dir_name(seg)
        if parsed is None:
            continue
        coll, ts = parsed[0], float(ts_s)
        prev = best.get(coll)
        if prev is None or ts >= prev[0]:
            best[coll] = (ts, seg)
    keep = [prefix_norm + seg + "/" for (_ts, seg) in best.values()]
    return [k for k in keys if any(k.startswith(p) for p in keep)]


def _group_remote_import_files(
    keys: list[str], prefix: str
) -> dict[tuple[str, str], list[str]]:
    """
    Group object keys by (collection, partition).
    Expected layout: .../col_<name>__<ts>/partition/<part>/<uuid>/<file>
    """
    prefix_norm = prefix.rstrip("/") + "/"
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    row_re = re.compile(
        r"^col_(.+)__(\d+(?:\.\d+)?)/partition/([^/]+)/([^/]+)/([^/]+)$"
    )
    for key in keys:
        if not key.startswith(prefix_norm):
            continue
        rel = key[len(prefix_norm) :]
        m = row_re.match(rel)
        if not m:
            continue
        coll, _ts, part, uuid_part, fname = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        if not UUID_DIR_RE.match(uuid_part):
            continue
        if fname.lower().endswith((".parquet", ".json")):
            groups[(coll, part)].append(key)
    return groups


def _poll_import_job(
    import_url: str,
    job_id: str,
    api_key: str,
    cluster_id: str,
    timeout_s: float = 3600.0,
) -> None:
    from pymilvus.bulk_writer import get_import_progress

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = get_import_progress(import_url, job_id, cluster_id, api_key)
        body = resp.json()
        data = body.get("data") or {}
        state = (
            data.get("state")
            or data.get("status")
            or data.get("importState")
            or ""
        )
        state_s = str(state).lower()
        click.echo(f"Import job {job_id!r} state: {state!r}")
        if state_s in ("completed", "success", "finished"):
            return
        if state_s in ("failed", "error", "cancelled", "canceled"):
            raise click.ClickException(
                f"Import job {job_id} failed: {json.dumps(data, default=str)}"
            )
        time.sleep(3.0)
    raise click.ClickException(f"Timed out waiting for import job {job_id}")


def _restore_collection_data_remote(
    scheme: str,
    bucket: str,
    prefix: str,
    db_name: str,
    connect_yaml: Path,
    milvus_endpoint: str,
    api_key: str,
) -> None:
    from pymilvus.bulk_writer import bulk_import

    data = _load_connection_yaml(connect_yaml)
    csp = _cloud_storage_params_from_yaml(data)
    if not csp["access_key"] or not csp["secret_key"]:
        raise click.BadParameter(
            "cloud_storage_params (access_key/secret_key) or AWS_* env vars required for remote import"
        )
    import_url = _bulk_import_api_url(milvus_endpoint, data)
    cluster_id = _bulk_import_cluster_id(milvus_endpoint, data)
    click.echo(f"bulk_import REST base: {import_url!r} cluster_id={cluster_id!r}")

    if scheme == "s3":
        keys = _list_s3_keys_for_restore(bucket, prefix, csp)
    else:
        keys = _list_gs_keys_for_restore(bucket, prefix)

    keys = _filter_keys_latest_export(keys, prefix)

    groups = _group_remote_import_files(keys, prefix)
    if not groups:
        raise click.ClickException(
            f"No importable parquet/json files found under {scheme}://{bucket}/{prefix}"
        )
    for (coll, part), key_list in sorted(groups.items()):
        if not key_list:
            continue
        scheme_prefix = f"{scheme}://{bucket}/"
        object_urls = [[scheme_prefix + k] for k in sorted(key_list)]
        omit_pn = part == PARTITION_KEY_EXPORT_DIR
        if omit_pn:
            click.echo(
                f"bulk_import collection={coll!r} (partition key export, no partition_name) "
                f"files={len(object_urls)}"
            )
        else:
            click.echo(
                f"bulk_import collection={coll!r} partition={part!r} files={len(object_urls)}"
            )
        bi_kwargs: dict[str, Any] = dict(
            url=import_url,
            api_key=api_key,
            cluster_id=cluster_id,
            db_name=db_name,
            collection_name=coll,
            object_urls=object_urls,
            access_key=csp["access_key"],
            secret_key=csp["secret_key"],
        )
        if not omit_pn:
            bi_kwargs["partition_name"] = part
        resp = bulk_import(**bi_kwargs)
        body = resp.json()
        job_id = (
            (body.get("data") or {}).get("jobId")
            or (body.get("data") or {}).get("id")
            or body.get("jobId")
        )
        if job_id:
            _poll_import_job(import_url, str(job_id), api_key, cluster_id)
        else:
            click.echo(f"bulk_import response (no jobId to poll): {json.dumps(body, default=str)}")


def _restore_collection_data(
    database_dir: str, connect_config_file: Path, api_key: str
) -> None:
    scheme, db_name, details = _parse_database_dir_uri(database_dir)
    milvus_endpoint = _load_connect_config(connect_config_file)
    client = MilvusClient(uri=milvus_endpoint, token=api_key)
    _ensure_database(client, db_name)

    if scheme == "file":
        _restore_collection_data_file(client, details["local_path"], db_name)
        return

    _restore_collection_data_remote(
        scheme,
        details["bucket"],
        details["prefix"],
        db_name,
        connect_config_file,
        milvus_endpoint,
        api_key,
    )


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


def _schema_desc_has_partition_key(desc: dict[str, Any]) -> bool:
    """True if dumped __schema.json has a field with is_partition_key (Milvus partition key)."""
    for f in desc.get("fields") or []:
        if not isinstance(f, dict):
            continue
        if f.get("is_partition_key") or f.get("isPartitionKey"):
            return True
    return False


def _load_schema_desc(database_dir: Path, collection_name: str) -> dict[str, Any] | None:
    path = database_dir / f"{collection_name}__schema.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def _indexes_path_for_collection(database_dir: Path, collection: str) -> Path | None:
    for suffix in ["__indexes.json", "_indexes.json"]:
        p = database_dir / f"{collection}{suffix}"
        print(f"indexes_path_for_collection: {p}")
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


def _append_index_entry_to_params(
    index_params: Any, entry: dict[str, Any], *, preserve_index_type: bool
) -> None:
    field_name = entry["field_name"]
    if preserve_index_type:
        index_type = entry.get("index_type")
    else:
        index_type = "AUTOINDEX"
    index_name = entry.get("index_name") or field_name + "_index"
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
    client: MilvusClient,
    collection: str,
    indexes_path: Path,
    *,
    preserve_index_type: bool,
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
        _append_index_entry_to_params(
            index_params, item, preserve_index_type=preserve_index_type
        )
        to_add = True
    if to_add:
        client.create_index(
            collection_name=collection, index_params=index_params
        )
        click.echo(f"Created indexes on collection {collection!r} from {indexes_path.name}")


def _restore_collections(
    client: MilvusClient,
    database_dir: Path,
    db_name: str,
    ignore_default_partition: bool,
    preserve_index_type: bool,
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

    pk_by_coll: dict[str, bool] = {}
    for path in schema_paths:
        cn = _collection_from_schema_filename(path.name)
        assert cn is not None
        desc = json.loads(path.read_text())
        pk_by_coll[cn] = (
            _schema_desc_has_partition_key(desc) if isinstance(desc, dict) else False
        )

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
        if pk_by_coll.get(coll_name):
            click.echo(
                f"WARN: Ignoring {path.name!r} — collection {coll_name!r} has a partition key "
                "(__part.json not expected); skip create_partition from this file"
            )
            continue
        if ignore_default_partition and partition_name == "_default":
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
        _restore_collection_indexes(
            client,
            coll_name,
            idx_path,
            preserve_index_type=preserve_index_type,
        )


@click.command()
@click.option(
    "--database-dir",
    "database_dir",
    type=str,
    required=True,
    help="For --restore-collections-meta: local directory path. "
    "For --restore-collection-data: file://, s3://, or gs:// URI; last path segment is the database name.",
)
@click.option(
    "-i",
    "connect_config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Milvus YAML: connect.endpoint; for remote data import also cloud_storage_params (see docs).",
)
@click.option(
    "--restore-collections-meta",
    "restore_collections_meta",
    is_flag=True,
    help="Create database, collections, partitions, and indexes from JSON in --database-dir.",
)
@click.option(
    "--restore-collection-data",
    "restore_collection_data",
    is_flag=True,
    help="Restore rows from bulk export under --database-dir (file://, s3://, or gs://).",
)
@click.option(
    "--ignore-default-partition",
    is_flag=True,
    help="Skip restoring the _default partition (Milvus already has an implicit default).",
)
@click.option(
    "--preserve-index-type",
    is_flag=True,
    help="Use index_type from dumped JSON; otherwise use AUTOINDEX when creating indexes.",
)
def main(
    database_dir: str,
    connect_config_file: Path,
    restore_collections_meta: bool,
    restore_collection_data: bool,
    ignore_default_partition: bool,
    preserve_index_type: bool,
) -> None:
    """Restore Milvus metadata or collection data."""
    mode_flags = int(restore_collections_meta) + int(restore_collection_data)
    if mode_flags != 1:
        raise click.UsageError(
            "Specify exactly one of --restore-collections-meta or --restore-collection-data."
        )

    load_dotenv()
    token = os.getenv("TARGET_API_TOKEN", "root:Milvus")

    if restore_collections_meta:
        uri = _load_connect_config(connect_config_file)
        client = MilvusClient(uri=uri, token=token)
        database_dir_path = Path(database_dir).expanduser()
        if not database_dir_path.is_dir():
            raise click.BadParameter(
                f"--database-dir must be an existing directory for metadata restore: {database_dir_path}"
            )
        _restore_collections(
            client,
            database_dir_path,
            database_dir_path.name,
            ignore_default_partition=ignore_default_partition,
            preserve_index_type=preserve_index_type,
        )
        return

    _restore_collection_data(database_dir, connect_config_file, token)

if __name__ == "__main__":
    main()
