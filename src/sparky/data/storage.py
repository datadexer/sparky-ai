"""Parquet storage layer with metadata and data manifest.

Handles saving/loading DataFrames to Parquet with metadata,
incremental fetch support, and SHA-256 hash manifest generation
for reproducibility.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = Path("data/data_manifest.json")


class DataStore:
    """Parquet-based data storage with metadata and manifest tracking.

    Usage:
        store = DataStore()
        store.save(df, "data/raw/btc/ohlcv.parquet", metadata={"source": "binance"})
        df, meta = store.load("data/raw/btc/ohlcv.parquet")
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH

    def save(
        self,
        df: pd.DataFrame,
        path: str | Path,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save DataFrame to Parquet with metadata.

        Args:
            df: DataFrame to save.
            path: File path for the Parquet file.
            metadata: Optional metadata dict (source, asset, date_range, etc.).
                      Stored in the Parquet file's schema metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build metadata
        meta = metadata or {}
        meta["saved_at"] = datetime.now(timezone.utc).isoformat()
        meta["row_count"] = len(df)
        if not df.empty:
            meta["date_range_start"] = str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else str(df.index[0])
            meta["date_range_end"] = str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else str(df.index[-1])

        # Convert metadata to bytes for Parquet schema metadata
        table = pa.Table.from_pandas(df)
        existing_meta = table.schema.metadata or {}
        existing_meta[b"sparky_metadata"] = json.dumps(meta).encode()
        table = table.replace_schema_metadata(existing_meta)

        pq.write_table(table, str(path))
        logger.info(f"[DATA] Saved {len(df)} rows to {path}")

        # Update manifest
        self._update_manifest(path)

    def load(self, path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load DataFrame and metadata from Parquet.

        Args:
            path: Path to the Parquet file.

        Returns:
            Tuple of (DataFrame, metadata dict).
        """
        path = Path(path)
        table = pq.read_table(str(path))
        df = table.to_pandas()

        # Extract metadata
        meta = {}
        schema_meta = table.schema.metadata
        if schema_meta and b"sparky_metadata" in schema_meta:
            meta = json.loads(schema_meta[b"sparky_metadata"].decode())

        logger.info(f"[DATA] Loaded {len(df)} rows from {path}")
        return df, meta

    def get_last_timestamp(self, path: str | Path) -> Optional[datetime]:
        """Get the last timestamp from an existing Parquet file.

        Used for incremental fetching — fetch only data after this point.

        Args:
            path: Path to the Parquet file.

        Returns:
            Last timestamp as datetime, or None if file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            return None

        df, _ = self.load(path)
        if df.empty:
            return None

        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.max().to_pydatetime()

        # Try the last row's first datetime column
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return df[col].max().to_pydatetime()

        return None

    def append(
        self,
        new_df: pd.DataFrame,
        path: str | Path,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append new data to an existing Parquet file.

        If the file doesn't exist, creates it. Deduplicates by index.

        Args:
            new_df: New data to append.
            path: Path to the Parquet file.
            metadata: Optional metadata to update.
        """
        path = Path(path)
        if path.exists():
            existing_df, existing_meta = self.load(path)
            combined = pd.concat([existing_df, new_df])
            # Deduplicate by index
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            meta = {**existing_meta, **(metadata or {})}
        else:
            combined = new_df
            meta = metadata or {}

        self.save(combined, path, metadata=meta)

    def _update_manifest(self, path: Path) -> None:
        """Update data_manifest.json with SHA-256 hash of the file."""
        try:
            sha256 = self._compute_sha256(path)

            manifest = {}
            if self.manifest_path.exists():
                with open(self.manifest_path) as f:
                    manifest = json.load(f)

            manifest[str(path)] = {
                "sha256": sha256,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "size_bytes": path.stat().st_size,
            }

            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        except OSError as e:
            logger.warning(f"[DATA] Failed to update manifest: {e}")

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
