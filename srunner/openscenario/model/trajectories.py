"""Backend-independent trajectory sampling plans."""

from __future__ import absolute_import

from srunner.openscenario.validation.diagnostics import Diagnostic


class TrajectorySamplingError(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super(TrajectorySamplingError, self).__init__(str(diagnostic))


def polyline_samples(trajectory):
    """Return semantic polyline samples without converting positions to CARLA."""
    if trajectory.shape != "Polyline":
        raise TrajectorySamplingError(Diagnostic(
            "unsupported_feature", "Only Polyline trajectories have a generic sampler",
            source=trajectory.source))
    samples = list(trajectory.control_points)
    if any(samples[index]["time"] > samples[index + 1]["time"]
           for index in range(len(samples) - 1)):
        raise TrajectorySamplingError(Diagnostic(
            "semantic_error", "Polyline vertex times must be non-decreasing",
            source=trajectory.source))
    return samples


def backend_sampling_required(trajectory):
    """Describe shapes that require a backend sampling strategy."""
    if trajectory.shape not in ("Clothoid", "Nurbs"):
        return False
    return True
