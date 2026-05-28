#!/usr/bin/env python

# Copyright (c) 2026 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Runtime compatibility shim between CARLA 0.9.x (UE4 / PhysX) and CARLA 0.10.0+
(UE5 / Chaos). Exposes a version probe, an `IS_UE5` engine flag, and
version-aware lookup tables that translate legacy blueprint ids and
per-category fallbacks to the ids that actually ship on the running server.

Version mapping:
    0.9.x  -> UE4 / PhysX     (IS_UE5 == False)
    0.10.x -> UE5 / Chaos     (IS_UE5 == True)

Detection order (highest priority first):
    1. `SR_CARLA_VERSION` environment variable, if set. Useful when the
       installed wheel version disagrees with the running server, or for
       smoke tests that need to force a path. Value is a version string
       (e.g. `0.9.16`, `0.10.0`).
    2. The `carla` python package version reported by importlib.metadata.

Design notes:
- 0.10.0 renamed and trimmed the vehicle/walker catalogue. Several 0.9.x ids
  (tesla.model3, volkswagen.t2, kawasaki.ninja, diamondback.century, ...) have
  no direct UE5 equivalent. For those, the alias table points at the closest
  documented substitute from the 0.10.0 catalogue rather than failing the
  spawn outright. See .claude/reports/carla_ue5_0.10.0_compatibility_gap_analysis.md
  section G1 for the full inventory.
- The shim never invents an id: every UE5 target is taken from the 0.10.0
  vehicle/walker listing observed in the running server.
- On 0.9.x the lookup is a passthrough, so legacy behaviour is preserved.
"""

from __future__ import print_function

import os

try:
    from importlib.metadata import version as _carla_version, PackageNotFoundError

    def _read_carla_version_str():
        return _carla_version("carla")
except ImportError:
    import pkg_resources

    PackageNotFoundError = pkg_resources.DistributionNotFound  # type: ignore[misc,assignment]

    def _read_carla_version_str():
        return pkg_resources.get_distribution("carla").version

from packaging.version import Version


UE5_THRESHOLD = Version("0.10.0")
"""Version at which CARLA switches from UE4 / PhysX to UE5 / Chaos."""

_VERSION_OVERRIDE_ENV = "SR_CARLA_VERSION"


def _detect_carla_version():
    override = os.environ.get(_VERSION_OVERRIDE_ENV)
    if override:
        return Version(override)
    try:
        return Version(_read_carla_version_str())
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "scenario_runner could not determine the CARLA version: the `carla` "
            "Python package is not installed. Install the CARLA wheel matching "
            "your server, or set the {} environment variable "
            "(e.g. SR_CARLA_VERSION=0.10.0).".format(_VERSION_OVERRIDE_ENV)
        ) from exc


CARLA_VERSION = _detect_carla_version()
"""Resolved CARLA version (env override `SR_CARLA_VERSION` or installed wheel)."""

IS_UE5 = CARLA_VERSION >= UE5_THRESHOLD
"""True when running against CARLA 0.10.0+ (UE5 / Chaos)."""

IS_UE4 = not IS_UE5
"""True when running against CARLA 0.9.x (UE4 / PhysX). Convenience alias."""


def carla_version():
    """Return the resolved CARLA version as a packaging.Version."""
    return CARLA_VERSION


def is_ue5():
    """Function form of IS_UE5 (kept for callers that prefer call-style)."""
    return IS_UE5


# ---------------------------------------------------------------------------
# Blueprint id aliases — 0.9.x identifier -> 0.10.0 identifier
#
# Targets are the actual ids advertised by a live CARLA 0.10.0 server's
# blueprint library. When a 0.9.x model has no direct UE5 sibling (tesla,
# kawasaki, diamondback, etc.) we fall through to the closest documented
# substitute so spawns succeed; that substitution is recorded here, not
# fabricated at call time.
# ---------------------------------------------------------------------------

_UE5_VEHICLE_ALIASES = {
    # Direct renames
    'vehicle.lincoln.mkz_2017': 'vehicle.lincoln.mkz',
    'vehicle.lincoln.mkz2017': 'vehicle.lincoln.mkz',
    'vehicle.audi.tt': 'vehicle.ue4.audi.tt',
    'vehicle.dodge.charger_police_2020': 'vehicle.dodgecop.charger',
    'vehicle.carlamotors.carlacola': 'vehicle.carlacola.actors',

    # Removed in 0.10.0 — substituted to closest catalogue entry
    'vehicle.tesla.model3': 'vehicle.lincoln.mkz',
    'vehicle.volkswagen.t2': 'vehicle.sprinter.mercedes',
    'vehicle.nissan.micra': 'vehicle.mini.cooper',
    'vehicle.kawasaki.ninja': 'vehicle.mini.cooper',
    'vehicle.diamondback.century': 'vehicle.mini.cooper',
    'vehicle.bh.crossbike': 'vehicle.mini.cooper',
    'vehicle.gazelle.omafiets': 'vehicle.mini.cooper',
}

_UE5_WALKER_ALIASES = {
    # 0.10.0 pedestrians start at .0015 (no .0001..0014).
    'walker.pedestrian.0001': 'walker.pedestrian.0015',
}


# ---------------------------------------------------------------------------
# Per-category fallback used by CarlaDataProvider.create_blueprint when a
# requested model id is missing from the library. Values must resolve on the
# active server.
# ---------------------------------------------------------------------------

_LEGACY_CATEGORY_FALLBACKS = {
    'car': 'vehicle.tesla.model3',
    'van': 'vehicle.volkswagen.t2',
    'truck': 'vehicle.carlamotors.carlacola',
    'trailer': '',
    'semitrailer': '',
    'bus': 'vehicle.volkswagen.t2',
    'motorbike': 'vehicle.kawasaki.ninja',
    'bicycle': 'vehicle.diamondback.century',
    'train': '',
    'tram': '',
    'pedestrian': 'walker.pedestrian.0001',
    'misc': 'static.prop.streetbarrier',
}

_UE5_CATEGORY_FALLBACKS = {
    'car': 'vehicle.lincoln.mkz',
    'van': 'vehicle.sprinter.mercedes',
    'truck': 'vehicle.carlacola.actors',
    'trailer': '',
    'semitrailer': '',
    'bus': 'vehicle.sprinter.mercedes',
    'motorbike': 'vehicle.mini.cooper',
    'bicycle': 'vehicle.mini.cooper',
    'train': '',
    'tram': '',
    'pedestrian': 'walker.pedestrian.0015',
    'misc': 'static.prop.streetbarrier',
}


def resolve_blueprint_id(model):
    """Translate a possibly-legacy blueprint id to one valid on the running server.

    On 0.9.x this is a passthrough. On 0.10.0+ it applies the documented
    vehicle / walker rename tables. Unknown ids pass through unchanged so the
    existing wildcard `filter()` path still has a chance.
    """
    if not is_ue5() or not isinstance(model, str):
        return model
    if model in _UE5_VEHICLE_ALIASES:
        return _UE5_VEHICLE_ALIASES[model]
    if model in _UE5_WALKER_ALIASES:
        return _UE5_WALKER_ALIASES[model]
    return model


def actor_blueprint_categories():
    """Return the per-category fallback dict appropriate for the running server."""
    return _UE5_CATEGORY_FALLBACKS if is_ue5() else _LEGACY_CATEGORY_FALLBACKS
