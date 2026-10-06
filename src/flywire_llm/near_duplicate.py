from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .quality import hamming_distance64, simhash_lsh_bands


class NearDuplicateIndexError(ValueError):
    pass


@dataclass(frozen=True)
class NearDuplicateMatch:
    record_id: str
    simhash64_hex: str
    hamming_distance: int


class SQLiteSimhashIndex:
    """Disk-backed candidate index for deterministic near-deduplication.

    Records must be presented in a frozen deterministic order. The first
    accepted record becomes canonical; later records at or below the configured
    Hamming threshold can be dropped or linked to that canonical record.
    """

    def __init__(self, path: str | Path, *, bands: int = 4) -> None:
        if bands < 1 or 64 % bands != 0:
            raise NearDuplicateIndexError(
                "bands must be a positive divisor of 64"
            )
        self.path = Path(path)
        self.bands = bands
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                simhash_hex TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS bands (
                band_key TEXT NOT NULL,
                record_id TEXT NOT NULL,
                PRIMARY KEY (band_key, record_id),
                FOREIGN KEY (record_id) REFERENCES records(record_id)
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_bands_key ON bands(band_key)"
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "SQLiteSimhashIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _keys(self, fingerprint: int) -> tuple[str, ...]:
        width = 64 // self.bands
        hex_width = (width + 3) // 4
        return tuple(
            f"{band}:{value:0{hex_width}x}"
            for band, value in simhash_lsh_bands(
                fingerprint,
                bands=self.bands,
            )
        )

    @staticmethod
    def _hex(fingerprint: int) -> str:
        if not 0 <= fingerprint < 2**64:
            raise NearDuplicateIndexError(
                "fingerprint must be an unsigned 64-bit integer"
            )
        return f"{fingerprint:016x}"

    @staticmethod
    def _from_hex(value: str) -> int:
        return int(value, 16)

    def find_matches(
        self,
        fingerprint: int,
        *,
        max_hamming_distance: int = 3,
    ) -> tuple[NearDuplicateMatch, ...]:
        if not 0 <= max_hamming_distance <= 64:
            raise NearDuplicateIndexError(
                "max_hamming_distance must be between 0 and 64"
            )
        keys = self._keys(fingerprint)
        placeholders = ",".join("?" for _ in keys)
        query = f"""
            SELECT DISTINCT r.record_id, r.simhash_hex
            FROM bands b
            JOIN records r ON r.record_id = b.record_id
            WHERE b.band_key IN ({placeholders})
        """
        matches: list[NearDuplicateMatch] = []
        for record_id, simhash_hex in self._db.execute(query, keys):
            distance = hamming_distance64(
                fingerprint,
                self._from_hex(simhash_hex),
            )
            if distance <= max_hamming_distance:
                matches.append(
                    NearDuplicateMatch(
                        record_id=record_id,
                        simhash64_hex=simhash_hex,
                        hamming_distance=distance,
                    )
                )
        return tuple(
            sorted(
                matches,
                key=lambda item: (
                    item.hamming_distance,
                    item.record_id,
                ),
            )
        )

    def add(
        self,
        record_id: str,
        fingerprint: int,
        *,
        commit: bool = True,
    ) -> None:
        if not isinstance(record_id, str) or not record_id.strip():
            raise NearDuplicateIndexError(
                "record_id must be a non-empty string"
            )
        simhash_hex = self._hex(fingerprint)
        try:
            self._db.execute(
                "INSERT INTO records(record_id, simhash_hex) VALUES (?, ?)",
                (record_id, simhash_hex),
            )
            self._db.executemany(
                "INSERT INTO bands(band_key, record_id) VALUES (?, ?)",
                ((key, record_id) for key in self._keys(fingerprint)),
            )
            if commit:
                self._db.commit()
        except sqlite3.IntegrityError as exc:
            self._db.rollback()
            raise NearDuplicateIndexError(
                f"duplicate record_id {record_id!r}"
            ) from exc

    def add_if_unique(
        self,
        record_id: str,
        fingerprint: int,
        *,
        max_hamming_distance: int = 3,
        commit: bool = True,
    ) -> tuple[bool, tuple[NearDuplicateMatch, ...]]:
        matches = self.find_matches(
            fingerprint,
            max_hamming_distance=max_hamming_distance,
        )
        if matches:
            return False, matches
        self.add(record_id, fingerprint, commit=commit)
        return True, ()

    def commit(self) -> None:
        self._db.commit()

    @property
    def record_count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()
        assert row is not None
        return int(row[0])
