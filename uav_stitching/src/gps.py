"""WGS84 ellipsoidal coordinates to a local East-North-Up frame."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .utils import write_csv_atomic


WGS84_SEMI_MAJOR_AXIS_M = 6_378_137.0
WGS84_FLATTENING = 1.0 / 298.257_223_563
WGS84_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)


POSITION_CSV_FIELDS = (
    "image_id",
    "capture_index",
    "filename",
    "latitude",
    "longitude",
    "altitude",
    "east",
    "north",
    "up",
    "origin_latitude",
    "origin_longitude",
    "origin_altitude",
)


@dataclass(frozen=True)
class EnuOrigin:
    """Geodetic origin of a local ENU frame in WGS84 degrees/meters."""

    latitude: float
    longitude: float
    altitude: float


@dataclass(frozen=True)
class ImagePosition:
    """One image position in both WGS84 and local ENU coordinates."""

    image_id: int
    capture_index: int | None
    filename: str
    latitude: float
    longitude: float
    altitude: float
    east: float
    north: float
    up: float
    origin_latitude: float
    origin_longitude: float
    origin_altitude: float


def choose_enu_origin(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    altitudes: np.ndarray,
) -> EnuOrigin:
    """Choose the survey-center origin as the component-wise geodetic mean."""

    if latitudes.size == 0:
        raise ValueError("At least one WGS84 position is required")
    if not (
        np.all(np.isfinite(latitudes))
        and np.all(np.isfinite(longitudes))
        and np.all(np.isfinite(altitudes))
    ):
        raise ValueError("WGS84 coordinates must all be finite")
    return EnuOrigin(
        latitude=float(np.mean(latitudes)),
        longitude=float(np.mean(longitudes)),
        altitude=float(np.mean(altitudes)),
    )


def wgs84_to_enu(
    latitudes: np.ndarray | list[float],
    longitudes: np.ndarray | list[float],
    altitudes: np.ndarray | list[float],
    origin: EnuOrigin | None = None,
) -> tuple[np.ndarray, EnuOrigin]:
    """Convert WGS84 geodetic positions to local ENU meters.

    Input ordering is latitude, longitude, ellipsoidal altitude. Internally,
    The standard WGS84 ellipsoid equations convert geodetic coordinates to
    Earth-centered Earth-fixed (ECEF), after which the ECEF-to-ENU rotation at
    *origin* is applied. Geographic degrees are therefore never treated as
    Cartesian distances. This direct formulation avoids a native ``pyproj``
    DLL conflict observed in the supplied Windows Conda environment.

    Returns an ``(N, 3)`` array ordered as east, north, up and the origin used.
    """

    lat = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    lon = np.asarray(longitudes, dtype=np.float64).reshape(-1)
    alt = np.asarray(altitudes, dtype=np.float64).reshape(-1)
    if not (lat.shape == lon.shape == alt.shape):
        raise ValueError("Latitude, longitude, and altitude arrays must have identical shapes")
    if origin is None:
        origin = choose_enu_origin(lat, lon, alt)
    if not np.all(np.isfinite(np.column_stack((lat, lon, alt)))):
        raise ValueError("WGS84 coordinates must all be finite")

    x, y, z = _geodetic_to_ecef(lat, lon, alt)
    origin_xyz = _geodetic_to_ecef(
        np.array([origin.latitude], dtype=np.float64),
        np.array([origin.longitude], dtype=np.float64),
        np.array([origin.altitude], dtype=np.float64),
    )
    x0, y0, z0 = (component[0] for component in origin_xyz)
    delta = np.column_stack((x - x0, y - y0, z - z0))

    phi = math.radians(origin.latitude)
    lam = math.radians(origin.longitude)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    # Expanded component form intentionally avoids NumPy's BLAS-backed matrix
    # multiply. The supplied Windows environment has a native DLL conflict in
    # ``delta @ rotation.T`` even though element-wise NumPy operations work.
    dx, dy, dz = delta[:, 0], delta[:, 1], delta[:, 2]
    east = -sin_lam * dx + cos_lam * dy
    north = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    up = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz
    enu = np.column_stack((east, north, up))
    return enu, origin


def _geodetic_to_ecef(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    altitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert WGS84 geodetic degrees/meters to ECEF meters."""

    phi = np.deg2rad(latitudes)
    lam = np.deg2rad(longitudes)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    radius = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(1.0 - WGS84_ECCENTRICITY_SQUARED * sin_phi**2)
    x = (radius + altitudes) * cos_phi * np.cos(lam)
    y = (radius + altitudes) * cos_phi * np.sin(lam)
    z = (radius * (1.0 - WGS84_ECCENTRICITY_SQUARED) + altitudes) * sin_phi
    return x, y, z


def metadata_rows_to_positions(rows: list[dict[str, str]]) -> tuple[list[ImagePosition], EnuOrigin]:
    """Convert Phase-2 CSV rows with complete GPS fields into ENU positions."""

    if not rows:
        raise ValueError("Metadata CSV contains no image rows")
    incomplete = [
        row.get("filename", f"image_id={row.get('image_id', '?')}")
        for row in rows
        if not row.get("latitude") or not row.get("longitude") or not row.get("altitude")
    ]
    if incomplete:
        preview = ", ".join(incomplete[:8])
        raise ValueError(f"GPS is incomplete for {len(incomplete)} images: {preview}")

    latitudes = np.array([float(row["latitude"]) for row in rows], dtype=np.float64)
    longitudes = np.array([float(row["longitude"]) for row in rows], dtype=np.float64)
    altitudes = np.array([float(row["altitude"]) for row in rows], dtype=np.float64)
    enu, origin = wgs84_to_enu(latitudes, longitudes, altitudes)

    positions: list[ImagePosition] = []
    for row, coordinate in zip(rows, enu, strict=True):
        capture_text = row.get("capture_index", "")
        positions.append(
            ImagePosition(
                image_id=int(row["image_id"]),
                capture_index=int(capture_text) if capture_text else None,
                filename=row["filename"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                altitude=float(row["altitude"]),
                east=float(coordinate[0]),
                north=float(coordinate[1]),
                up=float(coordinate[2]),
                origin_latitude=origin.latitude,
                origin_longitude=origin.longitude,
                origin_altitude=origin.altitude,
            )
        )
    return positions, origin


def write_positions_csv(path: str | Path, positions: list[ImagePosition]) -> None:
    """Persist ENU coordinates using an atomic cache update."""

    write_csv_atomic(Path(path), [asdict(position) for position in positions], POSITION_CSV_FIELDS)


def read_positions_csv(path: str | Path) -> list[ImagePosition]:
    """Read cached ENU coordinates."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Position cache does not exist: {csv_path}")
    positions: list[ImagePosition] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            capture_text = row.get("capture_index", "")
            positions.append(
                ImagePosition(
                    image_id=int(row["image_id"]),
                    capture_index=int(capture_text) if capture_text else None,
                    filename=row["filename"],
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    altitude=float(row["altitude"]),
                    east=float(row["east"]),
                    north=float(row["north"]),
                    up=float(row["up"]),
                    origin_latitude=float(row["origin_latitude"]),
                    origin_longitude=float(row["origin_longitude"]),
                    origin_altitude=float(row["origin_altitude"]),
                )
            )
    return positions


def choose_center_reference(positions: list[ImagePosition]) -> int:
    """Return the image ID closest to the ENU origin in the horizontal plane."""

    if not positions:
        raise ValueError("At least one position is required")
    return min(positions, key=lambda position: (math.hypot(position.east, position.north), position.image_id)).image_id
