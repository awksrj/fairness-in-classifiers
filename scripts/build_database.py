#!/usr/bin/env python
"""Build a local DuckDB database from the project's active CSV files.

This script is intentionally manual. It only runs when invoked directly, for
example:

    python scripts/build_database.py

CSV files inside the top-level unused/ directory are skipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import duckdb
except ImportError:  # pragma: no cover - depends on local environment
    print(
        "Missing dependency: duckdb\n"
        "Install it with: python -m pip install duckdb",
        file=sys.stderr,
    )
    raise SystemExit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "thought_experiment.duckdb"


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def normalize_name(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return normalized or "dataset"


def table_name_for(path: Path, used_names: set[str]) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    parent = "__".join(normalize_name(part) for part in rel.parent.parts)
    stem = normalize_name(path.stem)
    base = f"{parent}__{stem}" if parent else stem

    name = base
    counter = 2
    while name in used_names:
        name = f"{base}_{counter}"
        counter += 1
    used_names.add(name)
    return name


def is_active_csv(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT)
    return path.suffix.lower() == ".csv" and "unused" not in rel.parts


def discover_csv_files() -> list[Path]:
    return sorted(path for path in PROJECT_ROOT.rglob("*.csv") if is_active_csv(path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return len(next(reader, []))


def table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    return [row[1] for row in rows]


def create_catalog(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE dataset_catalog (
            table_name TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            experiment_group TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            row_count BIGINT NOT NULL,
            column_count BIGINT NOT NULL,
            file_hash TEXT NOT NULL,
            imported_at TIMESTAMP NOT NULL
        )
        """
    )


def import_csv(conn: duckdb.DuckDBPyConnection, path: Path, table_name: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")
    conn.execute(
        f"""
        CREATE TABLE {quote_identifier(table_name)} AS
        SELECT * FROM read_csv_auto(
            {quote_literal(path.as_posix())},
            header = true,
            union_by_name = true
        )
        """
    )


def insert_catalog_row(
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    table_name: str,
    imported_at: datetime,
) -> None:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    row_count = conn.execute(
        f"SELECT COUNT(*) FROM {quote_identifier(table_name)}"
    ).fetchone()[0]
    column_count = len(table_columns(conn, table_name)) or header_count(path)
    experiment_group = path.relative_to(PROJECT_ROOT).parts[0]

    conn.execute(
        """
        INSERT INTO dataset_catalog
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            table_name,
            rel,
            experiment_group,
            path.stem,
            row_count,
            column_count,
            file_sha256(path),
            imported_at,
        ],
    )


CANONICAL_COLUMNS = [
    ("ID", "id"),
    ("Gender", "gender"),
    ("SAT", "sat"),
    ("Hobby", "hobby"),
    ("Admission", "admission"),
    ("Qualification", "qualification"),
    ("Department", "department"),
    ("Admission_Probability", "admission_probability"),
    ("FTA_Probability", "fta_probability"),
    ("FTA_Prediction", "fta_prediction"),
    ("Predicted_Admission", "predicted_admission"),
    ("LFR_Score", "lfr_score"),
    ("Prototype", "prototype"),
    ("Admission_Score", "admission_score"),
    ("Alpha", "alpha"),
    ("Operation", "operation"),
    ("SAT_BIN", "sat_bin"),
]


def canonical_select(table_name: str, columns: set[str]) -> str:
    selected = [
        f"{quote_literal(table_name)} AS source_table",
        "catalog.source_path",
        "catalog.experiment_group",
        "catalog.dataset_name",
    ]

    for original, alias in CANONICAL_COLUMNS:
        if original in columns:
            selected.append(f"CAST(src.{quote_identifier(original)} AS VARCHAR) AS {alias}")
        else:
            selected.append(f"CAST(NULL AS VARCHAR) AS {alias}")

    return (
        "SELECT "
        + ", ".join(selected)
        + f" FROM {quote_identifier(table_name)} AS src "
        + "JOIN dataset_catalog AS catalog "
        + f"ON catalog.table_name = {quote_literal(table_name)}"
    )


def create_union_view(
    conn: duckdb.DuckDBPyConnection,
    view_name: str,
    tables: list[tuple[str, set[str]]],
) -> None:
    conn.execute(f"DROP VIEW IF EXISTS {quote_identifier(view_name)}")
    if not tables:
        conn.execute(
            f"""
            CREATE VIEW {quote_identifier(view_name)} AS
            SELECT *
            FROM dataset_catalog
            WHERE FALSE
            """
        )
        return

    sql = "\nUNION ALL\n".join(
        canonical_select(table_name, columns) for table_name, columns in tables
    )
    conn.execute(f"CREATE VIEW {quote_identifier(view_name)} AS {sql}")


def create_views(conn: duckdb.DuckDBPyConnection, imported_tables: list[str]) -> None:
    table_metadata = []
    for table_name in imported_tables:
        catalog_row = conn.execute(
            """
            SELECT source_path, dataset_name
            FROM dataset_catalog
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()
        source_path, dataset_name = catalog_row
        columns = set(table_columns(conn, table_name))
        table_metadata.append(
            {
                "table_name": table_name,
                "source_path": source_path,
                "dataset_name": dataset_name,
                "columns": columns,
            }
        )

    def pairs(predicate):
        return [
            (item["table_name"], item["columns"])
            for item in table_metadata
            if predicate(item)
        ]

    create_union_view(
        conn,
        "v_all_training_sets",
        pairs(
            lambda item: "training" in item["dataset_name"].lower()
            and "Admission" in item["columns"]
        ),
    )
    create_union_view(
        conn,
        "v_all_evaluation_sets",
        pairs(
            lambda item: "evaluation" in item["dataset_name"].lower()
            and "result" not in item["dataset_name"].lower()
        ),
    )
    create_union_view(
        conn,
        "v_all_prediction_results",
        pairs(
            lambda item: "result" in item["dataset_name"].lower()
            or "predicted" in item["dataset_name"].lower()
            or {"Admission_Probability", "FTA_Probability", "Predicted_Admission", "FTA_Prediction"}
            & item["columns"]
        ),
    )
    create_union_view(
        conn,
        "v_lfr_representations",
        pairs(lambda item: "representations" in item["dataset_name"].lower()),
    )
    create_union_view(
        conn,
        "v_lfr_prototypes",
        pairs(lambda item: "Prototype" in item["columns"]),
    )
    create_union_view(
        conn,
        "v_repair_operations",
        pairs(lambda item: "Operation" in item["columns"]),
    )


def build_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    csv_files = discover_csv_files()
    if not csv_files:
        raise SystemExit("No active CSV files found.")

    imported_at = datetime.now(timezone.utc)
    used_names: set[str] = set()

    conn = duckdb.connect(str(db_path))
    try:
        create_catalog(conn)

        imported_tables = []
        for path in csv_files:
            table_name = table_name_for(path, used_names)
            import_csv(conn, path, table_name)
            insert_catalog_row(conn, path, table_name, imported_at)
            imported_tables.append(table_name)

        create_views(conn, imported_tables)
    finally:
        conn.close()

    print(f"Built {db_path.relative_to(PROJECT_ROOT)}")
    print(f"Imported {len(csv_files)} CSV files")
    print("Skipped CSV files under unused/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually build the local DuckDB database from active CSV files."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Output database path. Default: {DEFAULT_DB_PATH.relative_to(PROJECT_ROOT)}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = args.db
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    build_database(db_path)


if __name__ == "__main__":
    main()
