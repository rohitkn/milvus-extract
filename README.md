# milvus-extract

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

  Restore Milvus metadata from dumped JSON (use --restore-collections).

Options:
  --database-dir DIRECTORY  Directory named like the target Milvus database
                            (basename = db name).  [required]
  -i PATH                   Milvus connection YAML (root key 'connect':
                            endpoint, api_key).  [required]
  --restore-collections     Create database, collections, partitions, and
                            indexes from JSON in --database-dir.
  --help                    Show this message and exit.
```
