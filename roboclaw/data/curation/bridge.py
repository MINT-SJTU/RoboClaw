from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    table = pq.read_table(path)
    return table.to_pylist()


def write_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table({})
    else:
        table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return {
        "path": str(path),
        "row_count": len(rows),
    }
