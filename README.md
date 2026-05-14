# milvus-extract

Tools for **extracting** Milvus collection data to disk and **restoring** metadata and bulk-exported rows.

- **`extract.py`** — Pulls data from a Milvus endpoint via `query_iterator` and writes to disk (JSON or Parquet) at `file://` or `s3://` locations. Can also list databases/collections and dump schemas and indexes to JSON.
- **`restore.py`** — Restores metadata from JSON dumps (`--restore-collections-meta`) or restores rows from bulk export paths (`--restore-collection-data`).

## Environment variables (`.env`)

Place a `.env` file in the working directory (or ensure variables are exported); `extract.py` and `restore.py` load it via `python-dotenv` where applicable.

| Variable | Description |
| -------- | ----------- |
| `SOURCE_API_TOKEN` | Milvus API key / token used for **data export** (`extract.py -f`). Falls back to `root:Milvus` if unset. |
| `API_TOKEN` | Milvus API key / token used for **extract.py** CLI actions with `-i` only (e.g. `--list-databases`, `--dump-schema`). Falls back to `root:Milvus` if unset. |
| `TARGET_API_TOKEN` | Milvus API key / token used for **restore.py** (metadata and collection-data restore). Falls back to `root:Milvus` if unset. |
| `CLOUD_STORAGE_ACCESS_KEY` | Object storage access key for **S3 bulk export** (`extract.py` when `output_location` is `s3://`) and **remote bulk import** (`restore.py --restore-collection-data` on `s3://` / `gs://`). |
| `CLOUD_STORAGE_SECRET_KEY` | Object storage secret key; pair with `CLOUD_STORAGE_ACCESS_KEY`. |
| `AWS_ACCESS_KEY_ID` | Alternative to `CLOUD_STORAGE_ACCESS_KEY` if the dedicated names are not set. |
| `AWS_SECRET_ACCESS_KEY` | Alternative to `CLOUD_STORAGE_SECRET_KEY` if the dedicated names are not set. |

Object-storage keys are read from the environment only (not from extract YAML). Endpoint and bucket for S3 still come from your YAML `cloud_storage_params` where required.

## Setup

```bash
pip install -r requirements.txt
```

## `extract.py --help`

```
Usage: extract.py [OPTIONS]

  Pull data from a Milvus endpoint via query_iterator and write to disk.

Options:
  -d, --database TEXT    Database name to extract from (default: None).
  -c, --collection TEXT  Collection name (default: None).
  -f PATH                Path to YAML config file (required for extract).
  -i PATH                Path to YAML Milvus connection credentials file (for
                         actions other than data extract)
  --list-databases       Print all database names.
  --list-collections     Print all collection names for the given database.
  --dump-schema          Write given collection schema as JSON in
                         schema_dir/dbname/ Use with -d/--database and
                         -c/--collection
  --dump-schema-all      Write all collection schemas as JSON in
                         schema_dir/dbname/ Use with -d/--database
  --dump-indexes         Write given collection's indexes as JSON in
                         schema_dir/dbname/ Use with -d/--database and
                         -c/--collection
  --schema-dir PATH      Directory for schema JSON files (default: current
                         directory).
  --help                 Show this message and exit.
```

## `restore.py --help`

```
Usage: restore.py [OPTIONS]

  Restore Milvus metadata or collection data.

Options:
  --database-dir TEXT         For --restore-collections-meta: local directory
                              path. For --restore-collection-data: file://,
                              s3://, or gs:// URI; last path segment is the
                              database name.  [required]
  -i PATH                     Milvus YAML: connect.endpoint; for remote data
                              import also cloud_storage_params (see docs).
                              [required]
  --restore-collections-meta  Create database, collections, partitions, and
                              indexes from JSON in --database-dir.
  --restore-collection-data   Restore rows from bulk export under --database-
                              dir (file://, s3://, or gs://).
  --ignore-default-partition  Skip restoring the _default partition (Milvus
                              already has an implicit default).
  --preserve-index-type       Use index_type from dumped JSON; otherwise use
                              AUTOINDEX when creating indexes.
  --help                      Show this message and exit.
```
