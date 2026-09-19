"""MaleCNS v1.0 raw data loading + provenance.

DEVELOPMENT.md section 8 step 1 requires the raw download itself to be recorded
(source, date, size, SHA256, version). `read_release` does the reading;
`release_provenance` produces the audit block.

The public release lives in the Google Cloud Storage bucket `flyem-male-cns`:

    https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/

The two files this project needs are the segment-to-segment connection strengths and the
curated per-neuron annotations. Column names differ between release versions, so every
column is resolved through an explicit candidate list rather than being hard-coded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..manifest import describe_file

RELEASE_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

DEFAULT_FILES = {
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body_stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
}

# Candidate column names, in priority order.
_PRE_CANDIDATES = ("bodyId_pre", "bodyid_pre", "pre", "pre_bodyId", "body_pre", "source")
_POST_CANDIDATES = ("bodyId_post", "bodyid_post", "post", "post_bodyId", "body_post", "target")
_WEIGHT_CANDIDATES = ("weight", "synapse_count", "syn_count", "count", "n_syn")
_ID_CANDIDATES = ("bodyId", "bodyid", "body_id", "id", "segment_id")


def _resolve_column(columns, candidates, what: str, path: Path) -> str:
    lookup = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    raise KeyError(
        f"Could not find a {what} column in {path}. "
        f"Looked for {list(candidates)}; available columns: {[str(c) for c in columns]}"
    )


def resolve_weight_columns(df: pd.DataFrame, path: Path) -> tuple[str, str, str]:
    pre = _resolve_column(df.columns, _PRE_CANDIDATES, "pre-synaptic body id", path)
    post = _resolve_column(df.columns, _POST_CANDIDATES, "post-synaptic body id", path)
    weight = _resolve_column(df.columns, _WEIGHT_CANDIDATES, "connection weight", path)
    return pre, post, weight


def resolve_id_column(df: pd.DataFrame, path: Path) -> str:
    return _resolve_column(df.columns, _ID_CANDIDATES, "body id", path)


def read_edges(path: str | Path) -> pd.DataFrame:
    """Read the segment-to-segment connection table with canonical column names.

    Returns a frame with columns `pre`, `post`, `weight` (int64 where possible).
    """
    path = Path(path)
    # Resolve names from the schema first, then read only the three columns we need: the full
    # release table is ~1 GB and carries extra per-edge metadata we never use.
    pre, post, weight = resolve_weight_columns(_read_feather_schema(path), path)
    df = pd.read_feather(path, columns=[pre, post, weight])
    out = pd.DataFrame(
        {
            "pre": df[pre].astype("int64"),
            "post": df[post].astype("int64"),
            "weight": df[weight].astype("float64"),
        }
    )
    return out


def _read_feather_schema(path: Path) -> pd.DataFrame:
    """Column names of a Feather file without loading its data."""
    try:
        import pyarrow.feather as feather

        reader = feather.FeatherReader(str(path))
        names = [field.name for field in reader.schema]
        return pd.DataFrame(columns=names)
    except Exception:
        # Fallback: read one row. Only used if the Feather reader API changes.
        return pd.read_feather(path, columns=None).head(1)


def read_annotations(path: str | Path) -> pd.DataFrame:
    """Read per-neuron annotations. Column names are left untouched for the caller to map."""
    path = Path(path)
    df = pd.read_feather(path)
    id_col = resolve_id_column(df, path)
    if id_col != "bodyId":
        df = df.rename(columns={id_col: "bodyId"})
    df["bodyId"] = df["bodyId"].astype("int64")
    return df


def find_column(df: pd.DataFrame, candidates, default=None):
    lookup = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return default


def describe_annotations(df: pd.DataFrame) -> dict:
    """Small schema summary used in reports so the reader can see what was available."""
    info: dict = {"n_rows": int(len(df)), "columns": [str(c) for c in df.columns]}
    for key, cands in {
        "super_class": ("superClass", "super_class", "superclass"),
        "cell_class": ("class", "cell_class", "cellClass"),
        "cell_type": ("cellType", "cell_type", "celltype"),
        "side": ("side", "hemisphere"),
        "region": ("region", "somaNeuromere", "primaryRoi", "roi", "neuropil"),
        "status": ("status", "review_status", "proofread"),
    }.items():
        col = find_column(df, cands)
        if col is None:
            info[key] = None
            continue
        vc = df[col].astype(str).value_counts().head(15)
        info[key] = {"column": str(col), "n_unique": int(df[col].nunique(dropna=True)),
                     "top_values": {str(k): int(v) for k, v in vc.items()}}
    return info


def release_provenance(paths: dict[str, str | Path], version: str = "v1.0",
                      download_date: str | None = None) -> dict:
    """Provenance block for the raw release files (DEVELOPMENT.md section 8 step 1)."""
    files = {}
    for key, path in paths.items():
        entry = describe_file(path)
        entry["source_url"] = f"{RELEASE_BASE}/{Path(path).name}"
        files[key] = entry
    return {
        "dataset": "MaleCNS",
        "version": version,
        "bucket": "gs://flyem-male-cns",
        "release_base": RELEASE_BASE,
        "download_date_utc": download_date or datetime.now(timezone.utc).date().isoformat(),
        "files": files,
    }
