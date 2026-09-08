"""Parser for the DJI exposure-event ``information.MRK`` text format."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


GPS_EPOCH = datetime(1980, 1, 6, tzinfo=UTC)
DEFAULT_GPS_UTC_LEAP_SECONDS = 18

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_MRK_PATTERN = re.compile(
    rf"""
    ^\s*(?P<exposure>\d+)\s+
    (?P<tow>{_FLOAT})\s+
    \[\s*(?P<week>\d+)\s*\]\s+
    (?P<offset_n>{_FLOAT})\s*,\s*N\s+
    (?P<offset_e>{_FLOAT})\s*,\s*E\s+
    (?P<offset_v>{_FLOAT})\s*,\s*V\s+
    (?P<latitude>{_FLOAT})\s*,\s*Lat\s+
    (?P<longitude>{_FLOAT})\s*,\s*Lon\s+
    (?P<height>{_FLOAT})\s*,\s*Ellh\s+
    (?P<std_lat>{_FLOAT})\s*,\s*
    (?P<std_lon>{_FLOAT})\s*,\s*
    (?P<std_hgt>{_FLOAT})\s+
    (?P<quality>\d+)\s*,\s*Q\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class MrkRecord:
    """One DJI camera exposure event parsed from ``information.MRK``."""

    exposure_index: int
    gps_seconds_of_week: float
    gps_week: int
    offset_n: float
    offset_e: float
    offset_v: float
    latitude: float
    longitude: float
    ellipsoidal_height: float
    std_lat_m: float
    std_lon_m: float
    std_hgt_m: float
    quality: int
    line_number: int
    raw_line: str

    @property
    def gps_scale_datetime(self) -> datetime:
        """Return GPS epoch + week + seconds without applying UTC leap seconds."""

        return GPS_EPOCH + timedelta(weeks=self.gps_week, seconds=self.gps_seconds_of_week)

    def utc_datetime(self, leap_seconds: int = DEFAULT_GPS_UTC_LEAP_SECONDS) -> datetime:
        """Convert GPS time to UTC using an explicit leap-second offset."""

        return self.gps_scale_datetime - timedelta(seconds=leap_seconds)


@dataclass(frozen=True)
class MrkParseIssue:
    """A non-empty MRK line that did not match the detected DJI text grammar."""

    line_number: int
    text: str
    reason: str


@dataclass(frozen=True)
class MrkParseResult:
    """Parsed records plus rejected-line diagnostics."""

    records: tuple[MrkRecord, ...]
    issues: tuple[MrkParseIssue, ...]


def parse_mrk_line(line: str, line_number: int = 1) -> MrkRecord:
    """Parse one observed DJI MRK exposure line.

    The N/E/V values are retained verbatim because their physical meaning is
    DJI-format-specific and they are not needed for image-position extraction.
    Latitude/longitude are WGS84 degrees and ``Ellh`` is ellipsoidal height.
    """

    match = _MRK_PATTERN.match(line)
    if match is None:
        raise ValueError(f"Line {line_number} does not match the DJI MRK exposure format")
    values = match.groupdict()
    return MrkRecord(
        exposure_index=int(values["exposure"]),
        gps_seconds_of_week=float(values["tow"]),
        gps_week=int(values["week"]),
        offset_n=float(values["offset_n"]),
        offset_e=float(values["offset_e"]),
        offset_v=float(values["offset_v"]),
        latitude=float(values["latitude"]),
        longitude=float(values["longitude"]),
        ellipsoidal_height=float(values["height"]),
        std_lat_m=float(values["std_lat"]),
        std_lon_m=float(values["std_lon"]),
        std_hgt_m=float(values["std_hgt"]),
        quality=int(values["quality"]),
        line_number=line_number,
        raw_line=line.rstrip("\r\n"),
    )


def parse_mrk(path: str | Path) -> MrkParseResult:
    """Parse every non-empty MRK line without hiding malformed records."""

    mrk_path = Path(path)
    if not mrk_path.is_file():
        raise FileNotFoundError(f"MRK file does not exist: {mrk_path}")
    records: list[MrkRecord] = []
    issues: list[MrkParseIssue] = []
    with mrk_path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(parse_mrk_line(line, line_number))
            except ValueError as exc:
                issues.append(MrkParseIssue(line_number, line.rstrip("\r\n"), str(exc)))

    duplicates = _duplicates(record.exposure_index for record in records)
    for duplicate in duplicates:
        issues.append(
            MrkParseIssue(
                line_number=0,
                text=str(duplicate),
                reason=f"Duplicate MRK exposure index: {duplicate}",
            )
        )
    return MrkParseResult(tuple(records), tuple(issues))


def _duplicates(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
