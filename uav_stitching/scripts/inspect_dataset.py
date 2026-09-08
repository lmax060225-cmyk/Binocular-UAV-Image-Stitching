#!/usr/bin/env python
"""Phase 1-2: inspect real UAV image metadata and DJI positioning files."""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metadata import (  # noqa: E402
    MetadataBuildResult,
    build_image_metadata,
    inspect_auxiliary_file,
    write_metadata_csv,
)
from src.utils import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Directory containing JPG/JPEG images")
    parser.add_argument("--mrk", required=True, type=Path, help="DJI information.MRK path")
    parser.add_argument("--rtk", type=Path, help="Optional information.RTK path; defaults beside MRK")
    parser.add_argument("--nav", type=Path, help="Optional information.NAV path; defaults beside MRK")
    parser.add_argument("--obs", type=Path, help="Optional information.OBS path; defaults beside MRK")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "cache" / "image_metadata.csv",
        help="Metadata CSV cache path",
    )
    parser.add_argument("--preview-records", type=int, default=10, help="Number of MRK records to print")
    parser.add_argument("--timezone-offset-hours", type=float, default=8.0, help="EXIF local UTC offset")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _default_sibling(explicit: Path | None, mrk_path: Path, filename: str) -> Path:
    return explicit if explicit is not None else mrk_path.resolve().parent / filename


def print_runtime() -> None:
    import cv2

    print("[Environment]")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"cv2.SIFT_create: {hasattr(cv2, 'SIFT_create')}")
    print(f"cv2.detail: {hasattr(cv2, 'detail')}")
    print(f"cv2.detail_GraphCutSeamFinder: {hasattr(cv2, 'detail_GraphCutSeamFinder')}")


def print_auxiliary_files(args: argparse.Namespace) -> None:
    print("\n[Auxiliary positioning files]")
    paths = {
        "RTK": _default_sibling(args.rtk, args.mrk, "information.RTK"),
        "NAV": _default_sibling(args.nav, args.mrk, "information.NAV"),
        "OBS": _default_sibling(args.obs, args.mrk, "information.OBS"),
        "PBK": args.mrk.resolve().parent / "ppk_file_name.pbk",
    }
    for label, path in paths.items():
        report = inspect_auxiliary_file(path)
        details = ", ".join(
            f"{key}={value}"
            for key, value in report.items()
            if key not in {"path", "first_line", "details"}
        )
        print(f"{label}: {report.get('path')} | {details}")
        if report.get("details"):
            print(f"  {report['details']}")
        if label == "PBK" and report.get("first_line"):
            print(f"  referenced RTK filename: {report['first_line']}")


def print_mrk_preview(result: MetadataBuildResult, count: int) -> None:
    print("\n[MRK parser]")
    print(f"Valid records: {len(result.mrk_result.records)}")
    print(f"Rejected/duplicate diagnostics: {len(result.mrk_result.issues)}")
    for issue in result.mrk_result.issues:
        print(f"  ERROR line={issue.line_number}: {issue.reason} | {issue.text}")
    print(f"First {min(count, len(result.mrk_result.records))} valid records:")
    for record in result.mrk_result.records[:count]:
        print(
            "  "
            f"exposure={record.exposure_index:4d} "
            f"gps_week={record.gps_week} tow={record.gps_seconds_of_week:.6f} "
            f"lat={record.latitude:.8f} lon={record.longitude:.8f} "
            f"ellh={record.ellipsoidal_height:.3f}m "
            f"std=({record.std_lat_m:.6f},{record.std_lon_m:.6f},{record.std_hgt_m:.6f})m "
            f"quality={record.quality}"
        )


def print_image_preview(result: MetadataBuildResult) -> None:
    print("\n[Three-image EXIF/XMP preview]")
    total = len(result.raw_images)
    sample_indices = sorted({0, total // 2, total - 1})
    by_filename = {record.filename: record for record in result.records}
    for index in sample_indices:
        raw = result.raw_images[index]
        merged = by_filename[raw.path.name]
        print(f"  {raw.path.name}")
        print(
            f"    size={raw.width}x{raw.height} container={raw.container_format} "
            f"camera={raw.camera_make} {raw.camera_model} focal={raw.focal_length_mm}mm"
        )
        print(
            f"    DateTimeOriginal={raw.datetime_original!r} SubSec={raw.subsec_time_original!r} "
            f"GPSDate={raw.gps_date_stamp!r} GPSTime={raw.gps_time_stamp!r}"
        )
        print(
            f"    EXIF GPS=({raw.exif_latitude:.9f}, {raw.exif_longitude:.9f}, {raw.exif_altitude:.3f})"
            if raw.exif_latitude is not None and raw.exif_longitude is not None and raw.exif_altitude is not None
            else "    EXIF GPS=incomplete"
        )
        print(
            f"    DJI XMP GPS=({raw.xmp_latitude}, {raw.xmp_longitude}, {raw.xmp_absolute_altitude}) "
            f"status={raw.xmp_gps_status} rtk_flag={raw.xmp_rtk_flag}"
        )
        print(
            f"    XMP CreateDate={raw.xmp_create_date!r} UTCAtExposure={raw.xmp_utc_at_exposure!r} "
            f"MRK={merged.mrk_index} via {merged.correspondence_method}"
        )


def print_mapping_summary(result: MetadataBuildResult) -> None:
    image_indices = {raw.capture_index for raw in result.raw_images if raw.capture_index is not None}
    mrk_indices = {record.exposure_index for record in result.mrk_result.records}
    methods = Counter(record.correspondence_method for record in result.records)
    print("\n[Image/MRK correspondence]")
    print(f"Images: {len(result.records)}")
    print(f"MRK records: {len(result.mrk_result.records)}")
    print(f"Matched image/MRK records: {result.matched_mrk_count}")
    print(f"Methods: {dict(methods)}")
    print(f"Image capture indices without MRK: {sorted(image_indices - mrk_indices)}")
    print(f"MRK exposure indices without image: {sorted(mrk_indices - image_indices)}")
    all_expected = set(range(min(image_indices | mrk_indices), max(image_indices | mrk_indices) + 1))
    print(f"Missing indices in both image and MRK sequences: {sorted(all_expected - image_indices - mrk_indices)}")

    horizontal = [record.position_crosscheck_m for record in result.records if record.position_crosscheck_m is not None]
    vertical = [record.altitude_crosscheck_m for record in result.records if record.altitude_crosscheck_m is not None]
    if horizontal:
        print(
            f"MRK vs embedded GPS horizontal delta: median={statistics.median(horizontal):.6f}m "
            f"max={max(horizontal):.6f}m"
        )
    if vertical:
        print(
            f"MRK vs embedded altitude delta: median={statistics.median(vertical):.6f}m "
            f"max={max(vertical):.6f}m"
        )

    raw_time_deltas: list[float] = []
    utc_time_deltas: list[float] = []
    mrk_by_index = {record.exposure_index: record for record in result.mrk_result.records}
    for raw in result.raw_images:
        if raw.capture_index not in mrk_by_index or not raw.xmp_utc_at_exposure:
            continue
        try:
            xmp_time = __import__("datetime").datetime.fromisoformat(raw.xmp_utc_at_exposure).replace(tzinfo=__import__("datetime").UTC)
        except ValueError:
            continue
        mrk = mrk_by_index[raw.capture_index]
        raw_time_deltas.append(abs((xmp_time - mrk.gps_scale_datetime).total_seconds()))
        utc_time_deltas.append(abs((xmp_time - mrk.utc_datetime()).total_seconds()))
    if raw_time_deltas:
        print(
            "DJI XMP UTCAtExposure diagnostic: "
            f"median delta to raw GPS scale={statistics.median(raw_time_deltas):.6f}s; "
            f"to leap-corrected UTC={statistics.median(utc_time_deltas):.6f}s"
        )
        print("  Interpretation: this dataset's tag value is GPS-scale time despite its UTCAtExposure name.")

    if result.diagnostics:
        print("Diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"  {diagnostic.severity}: {diagnostic.filename or '[MRK]'}: {diagnostic.message}")
    else:
        print("Diagnostics: none")


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    print_runtime()
    print_auxiliary_files(args)
    result = build_image_metadata(
        args.images,
        args.mrk,
        local_timezone_hours=args.timezone_offset_hours,
    )
    print_mrk_preview(result, max(0, args.preview_records))
    print_image_preview(result)
    print_mapping_summary(result)
    write_metadata_csv(args.output.resolve(), result.records)
    print(f"\n[Cache]\nMetadata CSV: {args.output.resolve()}")
    has_error = bool(result.mrk_result.issues) or any(item.severity == "ERROR" for item in result.diagnostics)
    return 2 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
