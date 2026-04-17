# milvus-extract

Tools for **extracting** Milvus collection data to disk and **restoring** database metadata from JSON dumps produced by the extract tool.

- **`extract.py`** — Pulls data from a Milvus endpoint via `query_iterator` and writes to disk (JSON or Parquet) at `file://` or `s3://` locations. Can also list databases/collections and dump schemas and indexes to JSON.
- **`restore.py`** — Recreates database, collections, partitions, and indexes from JSON layout: `db_dir/<collection>__schema.json`, `<collection>__<partition>__part.json`, and `<collection>__indexes.json`.

## Setup

```bash
pip install -r requirements.txt
```

## `extract.py --help`

```
Usage: extract.py [OPTIONS]

  Pull data from a Milvus endpoint via query_iterator and write to disk.

Options:
  -d, --database TEXT    Database name (default: None).
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

  Restore Milvus metadata from dumped JSON (use --restore-collections-meta).

Options:
  --database-dir DIRECTORY    Directory named like the target Milvus database
                              (basename = db name).  [required]
  -i PATH                     Milvus connection YAML (root key 'connect':
                              endpoint, api_key).  [required]
  --restore-collections-meta  Create database, collections, partitions, and
                              indexes from JSON in --database-dir.
  --ignore-default-partition  Skip restoring the _default partition (Milvus
                              already has an implicit default).
  --preserve-index-type       Use index_type from dumped JSON; otherwise use
                              AUTOINDEX when creating indexes.
  --help                      Show this message and exit.
```
