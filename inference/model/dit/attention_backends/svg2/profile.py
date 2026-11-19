"""Optional stage scopes used by SVG2 microbenchmarks."""

from contextlib import nullcontext


def stage(profile, name: str):
    return nullcontext() if profile is None else profile(name)
