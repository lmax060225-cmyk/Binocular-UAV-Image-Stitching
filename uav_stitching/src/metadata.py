"""JPG EXIF/XMP inspection and explicit image-to-MRK correspondence."""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from .mrk_parser import MrkParseResult, MrkRecord, parse_mrk
from .utils import find_jpeg_images, write_csv_atomic


LOGGER = logging.getLogger(__name__)
_CAPTURE_INDEX_RE = re.compile(r"_(\d{4,})_[^_]+$", re.IGNORECASE)
_XMP_ATTRIBUTE_RE = re.compile(r"(?P<name>[\w-]+:[\w-]+)\s*=\s*[\"'](?P<value>.*?)[\"']")

METADATA_CSV_FIELDS = (
    "image_id",
    "capture_index",
    "filename",
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
    "gps_source",
    "mrk_index",
    "correspondence_method",
    "camera_make",
    "camera_model",
    "focal_length_mm",
    "width",
    "height",
    "container_format",
    "datetime_original",
    "subsec_time_original",
    "gps_date_stamp",
    "gps_time_stamp",
    "xmp_create_date",
    "xmp_utc_at_exposure",
    "xmp_gps_status",
    "xmp_rtk_flag",
    "xmp_latitude",
    "xmp_longitude",
    "xmp_absolute_altitude",
    "mrk_gps_week",
    "mrk_gps_seconds_of_week",
    "mrk_gps_scale_datetime",
    "mrk_utc_datetime",
    "mrk_quality",
    "mrk_std_lat_m",
    "mrk_std_lon_m",
    "mrk_std_hgt_m",
    "position_crosscheck_m",
    "altitude_crosscheck_m",
)


@dataclass(frozen=True)
class RawImageMetadata:
    """Metadata decoded from one image before MRK association."""

    path: Path
    capture_index: int | None
    width: int
    height: int
    container_format: str
    camera_make: str | None
    camera_model: str | None
    focal_length_mm: float | None
    datetime_original: str | None
    subsec_time_original: str | None
    gps_date_stamp: str | None
    gps_time_stamp: str | None
    exif_latitude: float | None
    exif_longitude: float | None
    exif_altitude: float | None
    xmp_create_date: str | None
    xmp_utc_at_exposure: str | None
    xmp_gps_status: str | None
    xmp_rtk_flag: str | None
    xmp_latitude: float | None
    xmp_longitude: float | None
    xmp_absolute_altitude: float | None


@dataclass(frozen=True)
class ImageMetadata:
    """One output CSV row with the chosen position and its provenance."""

    image_id: int
    capture_index: int | None
    filename: str
    timestamp: str | None
    latitude: float | None
    longitude: float | None
    altitude: float | None
    gps_source: str | None
    mrk_index: int | None
    correspondence_method: str
    camera_make: str | None
    camera_model: str | None
    focal_length_mm: float | None
    width: int
    height: int
    container_format: str
    datetime_original: str | None
    subsec_time_original: str | None
    gps_date_stamp: str | None
    gps_time_stamp: str | None
    xmp_create_date: str | None
    xmp_utc_at_exposure: str | None
    xmp_gps_status: str | None
    xmp_rtk_flag: str | None
    xmp_latitude: float | None
    xmp_longitude: float | None
    xmp_absolute_altitude: float | None
    mrk_gps_week: int | None
    mrk_gps_seconds_of_week: float | None
    mrk_gps_scale_datetime: str | None
    mrk_utc_datetime: str | None
    mrk_quality: int | None
    mrk_std_lat_m: float | None
    mrk_std_lon_m: float | None
    mrk_std_hgt_m: float | None
    position_crosscheck_m: float | None
    altitude_crosscheck_m: float | None


@dataclass(frozen=True)
class MappingDiagnostic:
    """An explicit mapping warning/error that must not be silently guessed."""

    severity: str
    filename: str
    message: str


@dataclass(frozen=True)
class MetadataBuildResult:
    """Complete Phase-2 result and its mapping diagnostics."""

    records: tuple[ImageMetadata, ...]
    raw_images: tuple[RawImageMetadata, ...]
    mrk_result: MrkParseResult
    diagnostics: tuple[MappingDiagnostic, ...]

    @property
    def matched_mrk_count(self) -> int:
        return sum(record.mrk_index is not None for record in self.records)


def extract_capture_index(filename: str) -> int | None:
    """Extract the final DJI capture number, such as ``0001`` from ``..._0001_V``."""

    match = _CAPTURE_INDEX_RE.search(Path(filename).stem)
    return int(match.group(1)) if match else None


def inspect_image(path: str | Path) -> RawImageMetadata:
    """Read EXIF, GPS IFD, and DJI XMP fields from one JPG/JPEG.

    Pillow reports the supplied DJI files as MPO containers even though their
    extension is JPG. The first frame is the visible survey image; no frame
    ordering assumption is used for GPS association.
    """

    image_path = Path(path)
    with Image.open(image_path) as image:
        width, height = image.size
        container_format = image.format or "unknown"
        exif = image.getexif()
        root_tags = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif.items()}
        exif_ifd = _safe_ifd(exif, "Exif")
        gps_ifd = _safe_ifd(exif, "GPSInfo")
        xmp = _decode_xmp(image.info.get("xmp"))

    exif_tags = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_ifd.items()}
    gps_tags = {ExifTags.GPSTAGS.get(tag, tag): value for tag, value in gps_ifd.items()}
    xmp_attributes = {match.group("name"): match.group("value") for match in _XMP_ATTRIBUTE_RE.finditer(xmp)}

    exif_latitude = _dms_to_degrees(gps_tags.get("GPSLatitude"), gps_tags.get("GPSLatitudeRef"))
    exif_longitude = _dms_to_degrees(gps_tags.get("GPSLongitude"), gps_tags.get("GPSLongitudeRef"))
    exif_altitude = _gps_altitude(gps_tags.get("GPSAltitude"), gps_tags.get("GPSAltitudeRef"))
    gps_time_stamp = _format_gps_time(gps_tags.get("GPSTimeStamp"))

    return RawImageMetadata(
        path=image_path.resolve(),
        capture_index=extract_capture_index(image_path.name),
        width=int(width),
        height=int(height),
        container_format=container_format,
        camera_make=_string_or_none(root_tags.get("Make")),
        camera_model=_string_or_none(root_tags.get("Model")),
        focal_length_mm=_float_or_none(exif_tags.get("FocalLength", root_tags.get("FocalLength"))),
        datetime_original=_string_or_none(exif_tags.get("DateTimeOriginal")),
        subsec_time_original=_string_or_none(exif_tags.get("SubSecTimeOriginal")),
        gps_date_stamp=_string_or_none(gps_tags.get("GPSDateStamp")),
        gps_time_stamp=gps_time_stamp,
        exif_latitude=exif_latitude,
        exif_longitude=exif_longitude,
        exif_altitude=exif_altitude,
        xmp_create_date=_string_or_none(xmp_attributes.get("xmp:CreateDate")),
        xmp_utc_at_exposure=_string_or_none(xmp_attributes.get("drone-dji:UTCAtExposure")),
        xmp_gps_status=_string_or_none(xmp_attributes.get("drone-dji:GpsStatus")),
        xmp_rtk_flag=_string_or_none(xmp_attributes.get("drone-dji:RtkFlag")),
        xmp_latitude=_float_or_none(xmp_attributes.get("drone-dji:GpsLatitude")),
        xmp_longitude=_float_or_none(xmp_attributes.get("drone-dji:GpsLongitude")),
        xmp_absolute_altitude=_float_or_none(xmp_attributes.get("drone-dji:AbsoluteAltitude")),
    )


def build_image_metadata(
    images_dir: str | Path,
    mrk_file: str | Path,
    *,
    local_timezone_hours: float = 8.0,
    xmp_time_tolerance_s: float = 0.1,
    exif_time_tolerance_s: float = 3.0,
) -> MetadataBuildResult:
    """Inspect all images and associate MRK records without line-order matching.

    Association priority is: unique filename capture index, unique DJI XMP GPS
    scale timestamp, then unique EXIF local timestamp. A time-based association
    is accepted only when exactly one MRK record is inside the configured
    tolerance. Ambiguous candidates remain unmatched and are diagnosed.
    """

    image_paths = find_jpeg_images(Path(images_dir))
    if not image_paths:
        raise ValueError(f"No JPG/JPEG images found in: {Path(images_dir).resolve()}")
    raw_images = tuple(inspect_image(path) for path in image_paths)
    mrk_result = parse_mrk(mrk_file)
    mrk_records = mrk_result.records

    diagnostics: list[MappingDiagnostic] = []
    by_index: dict[int, list[MrkRecord]] = {}
    for record in mrk_records:
        by_index.setdefault(record.exposure_index, []).append(record)

    used_mrk_indices: set[int] = set()
    output: list[ImageMetadata] = []
    for image_id, raw in enumerate(raw_images):
        match, method, candidate_diagnostic = _match_mrk_record(
            raw,
            mrk_records,
            by_index,
            used_mrk_indices,
            local_timezone_hours=local_timezone_hours,
            xmp_time_tolerance_s=xmp_time_tolerance_s,
            exif_time_tolerance_s=exif_time_tolerance_s,
        )
        if candidate_diagnostic:
            diagnostics.append(candidate_diagnostic)
        if match is not None:
            used_mrk_indices.add(match.exposure_index)
        output.append(_merge_image_and_mrk(image_id, raw, match, method, diagnostics))

    unmatched_mrk = sorted(set(by_index) - used_mrk_indices)
    for exposure_index in unmatched_mrk:
        diagnostics.append(
            MappingDiagnostic("WARNING", "", f"MRK exposure {exposure_index} has no associated image")
        )
    return MetadataBuildResult(tuple(output), raw_images, mrk_result, tuple(diagnostics))


def write_metadata_csv(path: str | Path, records: tuple[ImageMetadata, ...] | list[ImageMetadata]) -> None:
    """Persist image metadata using an atomic cache update."""

    rows = [asdict(record) for record in records]
    write_csv_atomic(Path(path), rows, METADATA_CSV_FIELDS)


def read_metadata_csv(path: str | Path) -> list[dict[str, str]]:
    """Read the Phase-2 metadata cache as string dictionaries."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata cache does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def inspect_auxiliary_file(path: str | Path | None) -> dict[str, Any]:
    """Detect RINEX text versus opaque DJI binary without attempting PPK solving."""

    if path is None:
        return {"path": None, "exists": False, "format": "not_configured"}
    file_path = Path(path).resolve()
    if not file_path.is_file():
        return {"path": str(file_path), "exists": False, "format": "missing"}
    payload = file_path.read_bytes()[:4096]
    first_line = payload.splitlines()[0].decode("ascii", errors="replace") if payload else ""
    result: dict[str, Any] = {
        "path": str(file_path),
        "exists": True,
        "size_bytes": file_path.stat().st_size,
        "first_line": first_line,
    }
    if b"RINEX VERSION / TYPE" in payload:
        version = first_line[:9].strip()
        if "OBSERVATION DATA" in first_line:
            kind = "observation"
        elif "NAV DATA" in first_line:
            kind = "navigation"
        else:
            kind = "unknown"
        result.update(format="rinex", rinex_version=version, rinex_kind=kind)
        return result
    printable = sum(byte in b"\t\n\r" or 32 <= byte <= 126 for byte in payload)
    printable_ratio = printable / max(1, len(payload))
    if b"\x00" in payload or printable_ratio < 0.80:
        result.update(
            format="binary",
            details="Opaque DJI RTK/PPK raw stream; retained but not decoded for GPS neighbor construction",
            magic_hex=payload[:16].hex(" "),
        )
    else:
        result.update(format="text", details="Text file with no recognized RINEX header")
    return result


def _match_mrk_record(
    raw: RawImageMetadata,
    mrk_records: tuple[MrkRecord, ...],
    by_index: dict[int, list[MrkRecord]],
    used_indices: set[int],
    *,
    local_timezone_hours: float,
    xmp_time_tolerance_s: float,
    exif_time_tolerance_s: float,
) -> tuple[MrkRecord | None, str, MappingDiagnostic | None]:
    if raw.capture_index is not None:
        candidates = by_index.get(raw.capture_index, [])
        if len(candidates) == 1 and candidates[0].exposure_index not in used_indices:
            return candidates[0], "capture_index", None
        if len(candidates) > 1:
            return None, "ambiguous", MappingDiagnostic(
                "ERROR", raw.path.name, f"Capture index {raw.capture_index} maps to multiple MRK records"
            )

    xmp_time = _parse_iso_datetime(raw.xmp_utc_at_exposure, default_tz=UTC)
    if xmp_time is not None:
        candidates = [
            record
            for record in mrk_records
            if record.exposure_index not in used_indices
            and abs((record.gps_scale_datetime - xmp_time.astimezone(UTC)).total_seconds()) <= xmp_time_tolerance_s
        ]
        if len(candidates) == 1:
            return candidates[0], "xmp_gps_scale_time", None
        if len(candidates) > 1:
            return None, "ambiguous", MappingDiagnostic(
                "ERROR", raw.path.name, f"XMP exposure time matches {len(candidates)} MRK records"
            )

    exif_time = _parse_exif_datetime(raw.datetime_original, raw.subsec_time_original, local_timezone_hours)
    if exif_time is not None:
        candidates = [
            record
            for record in mrk_records
            if record.exposure_index not in used_indices
            and abs((record.utc_datetime() - exif_time.astimezone(UTC)).total_seconds()) <= exif_time_tolerance_s
        ]
        if len(candidates) == 1:
            return candidates[0], "exif_time", None
        if len(candidates) > 1:
            return None, "ambiguous", MappingDiagnostic(
                "ERROR", raw.path.name, f"EXIF time matches {len(candidates)} MRK records"
            )

    return None, "unmatched", MappingDiagnostic(
        "ERROR",
        raw.path.name,
        "No unique MRK match by capture index, XMP GPS-scale time, or EXIF time",
    )


def _merge_image_and_mrk(
    image_id: int,
    raw: RawImageMetadata,
    mrk: MrkRecord | None,
    method: str,
    diagnostics: list[MappingDiagnostic],
) -> ImageMetadata:
    xmp_position = (raw.xmp_latitude, raw.xmp_longitude, raw.xmp_absolute_altitude)
    exif_position = (raw.exif_latitude, raw.exif_longitude, raw.exif_altitude)
    crosscheck_m: float | None = None
    altitude_crosscheck_m: float | None = None
    if mrk is not None:
        latitude, longitude, altitude = mrk.latitude, mrk.longitude, mrk.ellipsoidal_height
        gps_source = "MRK"
        comparison = xmp_position if None not in xmp_position[:2] else exif_position
        if comparison[0] is not None and comparison[1] is not None:
            crosscheck_m = _horizontal_distance_m(
                mrk.latitude,
                mrk.longitude,
                float(comparison[0]),
                float(comparison[1]),
            )
            if crosscheck_m > 1.0:
                diagnostics.append(
                    MappingDiagnostic(
                        "WARNING",
                        raw.path.name,
                        f"MRK and embedded GPS differ by {crosscheck_m:.3f} m",
                    )
                )
        if comparison[2] is not None:
            altitude_crosscheck_m = abs(mrk.ellipsoidal_height - float(comparison[2]))
            if altitude_crosscheck_m > 1.0:
                diagnostics.append(
                    MappingDiagnostic(
                        "WARNING",
                        raw.path.name,
                        f"MRK and embedded altitude differ by {altitude_crosscheck_m:.3f} m",
                    )
                )
    elif raw.xmp_latitude is not None and raw.xmp_longitude is not None:
        latitude, longitude, altitude = xmp_position
        gps_source = "DJI_XMP"
    else:
        latitude, longitude, altitude = exif_position
        gps_source = "EXIF" if latitude is not None and longitude is not None else None

    timestamp = raw.xmp_create_date or _exif_iso_string(raw.datetime_original, raw.subsec_time_original)
    return ImageMetadata(
        image_id=image_id,
        capture_index=raw.capture_index,
        filename=raw.path.name,
        timestamp=timestamp,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        gps_source=gps_source,
        mrk_index=mrk.exposure_index if mrk else None,
        correspondence_method=method,
        camera_make=raw.camera_make,
        camera_model=raw.camera_model,
        focal_length_mm=raw.focal_length_mm,
        width=raw.width,
        height=raw.height,
        container_format=raw.container_format,
        datetime_original=raw.datetime_original,
        subsec_time_original=raw.subsec_time_original,
        gps_date_stamp=raw.gps_date_stamp,
        gps_time_stamp=raw.gps_time_stamp,
        xmp_create_date=raw.xmp_create_date,
        xmp_utc_at_exposure=raw.xmp_utc_at_exposure,
        xmp_gps_status=raw.xmp_gps_status,
        xmp_rtk_flag=raw.xmp_rtk_flag,
        xmp_latitude=raw.xmp_latitude,
        xmp_longitude=raw.xmp_longitude,
        xmp_absolute_altitude=raw.xmp_absolute_altitude,
        mrk_gps_week=mrk.gps_week if mrk else None,
        mrk_gps_seconds_of_week=mrk.gps_seconds_of_week if mrk else None,
        mrk_gps_scale_datetime=mrk.gps_scale_datetime.isoformat() if mrk else None,
        mrk_utc_datetime=mrk.utc_datetime().isoformat() if mrk else None,
        mrk_quality=mrk.quality if mrk else None,
        mrk_std_lat_m=mrk.std_lat_m if mrk else None,
        mrk_std_lon_m=mrk.std_lon_m if mrk else None,
        mrk_std_hgt_m=mrk.std_hgt_m if mrk else None,
        position_crosscheck_m=crosscheck_m,
        altitude_crosscheck_m=altitude_crosscheck_m,
    )


def _safe_ifd(exif: Any, ifd_name: str) -> dict[int, Any]:
    try:
        ifd_id = getattr(ExifTags.IFD, ifd_name)
        return dict(exif.get_ifd(ifd_id))
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}


def _decode_xmp(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8-sig", errors="replace")
    return value if isinstance(value, str) else ""


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip("\x00 ")
    return text or None


def _dms_to_degrees(value: Any, reference: Any) -> float | None:
    if value is None:
        return None
    try:
        degrees, minutes, seconds = (float(component) for component in value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if str(reference).upper() in {"S", "W"}:
        decimal = -decimal
    return decimal


def _gps_altitude(value: Any, reference: Any) -> float | None:
    altitude = _float_or_none(value)
    if altitude is None:
        return None
    below_sea_level = reference in {1, b"\x01", "1"}
    return -altitude if below_sea_level else altitude


def _format_gps_time(value: Any) -> str | None:
    if value is None:
        return None
    try:
        hours, minutes, seconds = (float(component) for component in value)
    except (TypeError, ValueError, ZeroDivisionError):
        return _string_or_none(value)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:09.6f}"


def _parse_iso_datetime(value: str | None, default_tz: timezone) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=default_tz) if parsed.tzinfo is None else parsed


def _parse_exif_datetime(value: str | None, subsec: str | None, timezone_hours: float) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    if subsec and subsec.isdigit():
        parsed = parsed.replace(microsecond=int((subsec + "000000")[:6]))
    return parsed.replace(tzinfo=timezone(timedelta(hours=timezone_hours)))


def _exif_iso_string(value: str | None, subsec: str | None) -> str | None:
    if not value:
        return None
    suffix = f".{subsec}" if subsec else ""
    return value.replace(":", "-", 2).replace(" ", "T") + suffix


def _horizontal_distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    mean_latitude = math.radians((lat_a + lat_b) / 2.0)
    north = math.radians(lat_b - lat_a) * 6_378_137.0
    east = math.radians(lon_b - lon_a) * 6_378_137.0 * math.cos(mean_latitude)
    return math.hypot(east, north)
