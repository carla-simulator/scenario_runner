"""Backend-independent trajectory sampling plans."""

from __future__ import absolute_import

from srunner.openscenario.validation.diagnostics import Diagnostic, ValidationError


class TrajectorySamplingError(ValidationError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super(TrajectorySamplingError, self).__init__([diagnostic])


def _literal_float(value):
    return float(value)


def polyline_samples(trajectory, resolve=_literal_float):
    """Return polyline vertices with times resolved by ``resolve`` (None when a vertex has no time)."""
    if trajectory.shape != "Polyline":
        raise TrajectorySamplingError(Diagnostic(
            "unsupported_feature", "Only Polyline trajectories have a generic sampler",
            source=trajectory.source))
    samples = []
    for point in trajectory.control_points:
        sample = dict(point)
        sample["time"] = resolve(point["time"]) if point.get("time") is not None else None
        samples.append(sample)
    times = [sample["time"] for sample in samples if sample["time"] is not None]
    if any(times[index] > times[index + 1] for index in range(len(times) - 1)):
        raise TrajectorySamplingError(Diagnostic(
            "semantic_error", "Polyline vertex times must be non-decreasing",
            source=trajectory.source))
    return samples


def timed_samples(trajectory, time_reference, resolve=_literal_float):
    """Return ``(samples, times)`` where ``times`` applies the TimeReference.

    ``times`` is None for ``TimeReference/None``; otherwise each vertex time ``t`` becomes
    ``t * scale + offset``. Every vertex needs a time when Timing is used.
    """
    samples = polyline_samples(trajectory, resolve)
    if time_reference is None or not time_reference.uses_timing:
        return samples, None
    if any(sample["time"] is None for sample in samples):
        raise TrajectorySamplingError(Diagnostic(
            "semantic_error", "Every Polyline vertex of trajectory {!r} needs a time when Timing is used".format(
                trajectory.name), source=trajectory.source))
    scale = resolve(time_reference.scale)
    offset = resolve(time_reference.offset)
    return samples, [sample["time"] * scale + offset for sample in samples]


def backend_sampling_required(trajectory):
    """Describe shapes that require a backend sampling strategy."""
    return trajectory.shape in ("Clothoid", "ClothoidSpline", "Nurbs")
