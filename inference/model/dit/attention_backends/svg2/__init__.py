"""SVG2 sparse attention for MagiHuman inference.

The public interface stays at ``attention_backends.svg2`` while algorithm,
executor and profiling code live in separate modules under this package.
"""

from .core import (
    SVG2Config,
    SVG2Session,
    _dense_gqa,
    _kmeans,
    _top_p_map,
    observe_svg2_density,
    observe_svg2_inputs,
    scheduled_svg2_attention,
    svg2_attention,
)

__all__ = [
    "SVG2Config",
    "SVG2Session",
    "observe_svg2_density",
    "observe_svg2_inputs",
    "scheduled_svg2_attention",
    "svg2_attention",
]
