"""Froxel-based volumetric fog math.

The helpers in this module are the CPU reference for a froxel renderer and
encode the fixes for the usual artefacts seen in a valley scene:

* :func:`froxel_depth` distributes slices logarithmically in depth, so thin
  near mist and dense far fog no longer share a single slice.
* :func:`integrate` accumulates single scattering along the view ray instead
  of taking one sample per light.
* :func:`reproject` resamples the previous frame using the per-pixel velocity
  instead of reusing the previous froxel at the same grid cell.
* :func:`shadow_slice` maps a fog slice onto the matching (linear,
  orthographic) directional-shadow depth slice and can snap to the shadow
  map's own slice centres, so filter taps never straddle a boundary.
* :func:`composite` blends fog with the scene in linear radiance space; HDR
  values stay unclamped and are expected to be tonemapped afterwards.
* :func:`composite_transparent` inserts transparents between the fog behind
  them and the fog in front of them, instead of folding them into the opaque
  pass where the whole fog volume ends up covering them.

All depths handed to and returned by this module are normalised to the
fog/shadow depth range unless documented otherwise.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence, Union

__all__ = [
    "froxel_depth",
    "integrate",
    "reproject",
    "shadow_slice",
    "composite",
    "composite_transparent",
]

Number = Union[int, float]
Inscatter = Union[Number, Callable[[float], float]]


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the closed unit interval."""
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def froxel_depth(
    index: int,
    near: float,
    far: float,
    num_slices: int = None,
    slice_index: int = None,
) -> float:
    """Return the view-space depth in metres of a froxel slice.

    Slices are distributed logarithmically between ``near`` and ``far``::

        depth(i) = near * (far / near) ** (i / (num_slices - 1))

    A logarithmic distribution keeps relative depth precision roughly
    constant, so close-range mist and far-range fog do not collapse into the
    same slice.  Slice ``0`` is exactly ``near`` and the last slice is exactly
    ``far``; the endpoints are snapped rather than exponentiated so callers
    can rely on them bit-for-bit.

    The slice index may be supplied in the leading position (``index``) or in
    the trailing position (``slice_index``); the trailing value wins when both
    are given.  This keeps the historical five-argument call site working.

    Parameters
    ----------
    index:
        Slice index (0 .. ``num_slices - 1``).
    near, far:
        Near and far plane of the froxel volume in metres.  ``near`` must be
        positive because the depth axis is geometric.
    num_slices:
        Number of depth slices in the froxel grid.
    slice_index:
        Optional trailing slice index; overrides ``index``.

    Raises
    ------
    ValueError
        If the depth range is degenerate or fewer than one slice is requested.
    """
    if num_slices is None:
        raise ValueError("num_slices is required")
    num_slices = int(num_slices)
    if num_slices < 1:
        raise ValueError("num_slices must be >= 1")
    if not near > 0.0:
        raise ValueError("near must be > 0 for a logarithmic distribution")
    if not far > near:
        raise ValueError("far must be > near")

    i = slice_index if slice_index is not None else index
    i = int(i)

    if num_slices == 1:
        return float(near)
    if i <= 0:
        return float(near)
    if i >= num_slices - 1:
        return float(far)

    # Geometric interpolation.  Computed in log space so the ratio's rounding
    # error does not compound at large slice counts.
    t = i / (num_slices - 1)
    return float(near * math.exp(math.log(far / near) * t))


def integrate(
    density: float,
    step_size: float,
    steps: int,
    *,
    inscatter: Inscatter = 1.0,
    extinction: float = 0.0,
    jitter: float = 0.0,
) -> float:
    """Integrate in-scattering along a view ray through the fog.

    The ray is marched in ``steps`` segments of ``step_size`` metres.  Each
    segment contributes the radiance arriving at its midpoint, so a point
    light's distance falloff inside the volume is integrated rather than
    point-sampled::

        sum_i  inscatter(x_i) * density * step_size * T(x_i)

    ``x_i`` is the distance from the ray origin to the sample midpoint.  Pass
    a callable as ``inscatter`` to evaluate the light's contribution at that
    distance (e.g. inverse-square falloff plus a phase function); a constant
    models a uniform ambient in-scatter.

    View-ray extinction is *off by default* (``extinction=0``) because the
    fog's transmittance is carried by its alpha and applied by
    :func:`composite`; enabling it here would attenuate the radiance twice.
    Set ``extinction`` to the medium's extinction coefficient when a
    self-attenuated radiance estimate is wanted.

    Parameters
    ----------
    density:
        Scattering coefficient of the medium (per metre).
    step_size:
        Length of one march segment in metres.
    steps:
        Number of segments; at least one.
    inscatter:
        Constant radiance per unit scattering, or a callable taking the
        sample's distance along the ray.
    extinction:
        Optional view-ray extinction coefficient applied per sample.
    jitter:
        Offset of the sample points in step units, in ``[-0.5, 0.5]``.  A
        per-pixel random offset breaks up banding from the fixed march.

    Raises
    ------
    ValueError
        If ``density``/``extinction``/``step_size`` are negative or ``steps``
        is less than one.
    """
    density = float(density)
    step_size = float(step_size)
    extinction = float(extinction)
    jitter = float(jitter)
    count = int(steps)

    if density < 0.0:
        raise ValueError("density must be >= 0")
    if extinction < 0.0:
        raise ValueError("extinction must be >= 0")
    if step_size < 0.0:
        raise ValueError("step_size must be >= 0")
    if count < 1:
        raise ValueError("steps must be >= 1")
    if not -0.5 <= jitter <= 0.5:
        raise ValueError("jitter must be in [-0.5, 0.5]")

    if density == 0.0 or step_size == 0.0:
        return 0.0

    use_callable = callable(inscatter)
    constant = 0.0 if use_callable else float(inscatter)

    total = 0.0
    for step in range(count):
        distance = (step + 0.5 + jitter) * step_size
        radiance = inscatter(distance) if use_callable else constant
        transmittance = math.exp(-extinction * distance) if extinction else 1.0
        total += float(radiance) * density * step_size * transmittance
    return total


def _sample_history(history: Sequence[float], x: float) -> float:
    """Linearly sample ``history`` at fractional index ``x`` (edge-clamped)."""
    last = len(history) - 1
    if x <= 0.0:
        return float(history[0])
    if x >= last:
        return float(history[last])
    i0 = int(math.floor(x))
    frac = x - i0
    return float(history[i0]) * (1.0 - frac) + float(history[i0 + 1]) * frac


def reproject(
    current: Sequence[float],
    previous: Sequence[float],
    velocity: Sequence[float],
) -> list:
    """Reproject the previous frame's froxel values with per-pixel velocity.

    Each current slice ``i`` looks up the previous frame at ``i - v`` where
    ``v`` is the velocity expressed in fractional slices, and resamples with
    bilinear filtering.  Neighbouring history slices are blended because a
    fractional shift lands between slices; reintroducing the previous frame's
    value at the *same* index is what makes fog walls follow the camera.

    A sample that lands outside the history with no usable neighbourhood
    (a disocclusion) falls back to the current frame's value so stale fog is
    not dragged across newly revealed geometry.

    ``current`` and ``previous`` are 1-D arrays of froxel values.  For a
    three-component velocity the depth component (``velocity[2]``) is used,
    since screen-space x/y do not index a 1-D history; two-component
    velocities use their first component.

    Parameters
    ----------
    current:
        Values produced by the current frame's lighting pass.
    previous:
        Values from the previous frame's froxel volume.
    velocity:
        Motion in fractional slices; the sign is the displacement from the
        current position back to the previous one.

    Returns
    -------
    list of float
        Reprojected history, one value per entry in ``current``.
    """
    current = list(current)
    previous = list(previous)
    if len(current) != len(previous):
        raise ValueError("current and previous must have the same length")
    if not current:
        return []

    try:
        components = len(velocity)
    except TypeError:  # scalar velocity
        shift = float(velocity)
    else:
        if components >= 3:
            shift = float(velocity[2])
        elif components >= 1:
            shift = float(velocity[0])
        else:
            shift = 0.0

    count = len(previous)
    result = []
    for i in range(count):
        source = i - shift
        # A shift larger than half a slice at the volume's edge cannot be
        # trusted: the history has no coverage there (disocclusion).
        if source < -0.5 or source > count - 0.5:
            result.append(float(current[i]))
            continue
        result.append(_sample_history(previous, source))
    return result


def _snap_depth(depth: float, num_slices: int) -> float:
    """Snap a normalised depth to the centre of its depth slice."""
    if num_slices < 1:
        raise ValueError("num_slices must be >= 1")
    # Keep the outer boundaries exact; they are slice edges, not centres.
    if depth <= 0.0:
        return 0.0
    if depth >= 1.0:
        return 1.0
    index = int(math.floor(depth * num_slices))
    if index >= num_slices:
        index = num_slices - 1
    return (index + 0.5) / num_slices


def shadow_slice(
    depth: float,
    near: float = 0.0,
    far: float = 1.0,
    num_slices: int = None,
    *,
    shadow_near: float = None,
    shadow_far: float = None,
    bias: float = 0.0,
) -> float:
    """Map a fog depth onto the matching directional-shadow depth slice.

    Directional shadow maps are orthographic, so shadow depth is linear while
    the fog slices are logarithmic.  Sampling the shadow map at a fog slice's
    *index* therefore misses the slice's actual depth and produces the hard,
    misaligned tree-shadow edges.  This helper converts the fog depth into the
    shadow map's normalised depth; pass ``num_slices`` to snap to the shadow
    map's own slice centres so PCF/filter taps stay inside one slice.

    Parameters
    ----------
    depth:
        Normalised fog depth (0 = ``near``, 1 = ``far``).  With the defaults
        this is already a normalised shadow depth and the mapping is the
        identity.
    near, far:
        Fog depth range that ``depth`` is normalised against.
    num_slices:
        Optional number of shadow depth slices to snap to.
    shadow_near, shadow_far:
        Shadow camera depth range, if it differs from the fog range.
    bias:
        Constant offset in normalised shadow depth; a negative bias
        compensates shadow-map depth precision at slice boundaries.

    Returns
    -------
    float
        Normalised shadow depth in ``[0, 1]``.
    """
    near = float(near)
    far = float(far)
    if not far > near:
        raise ValueError("far must be > near")

    t = (float(depth) - near) / (far - near)

    if shadow_near is not None or shadow_far is not None:
        s_near = near if shadow_near is None else float(shadow_near)
        s_far = far if shadow_far is None else float(shadow_far)
        if not s_far > s_near:
            raise ValueError("shadow_far must be > shadow_near")
        world_depth = near + t * (far - near)
        t = (world_depth - s_near) / (s_far - s_near)

    t = _clamp01(t)
    if bias:
        t = _clamp01(t + float(bias))
    if num_slices is not None:
        t = _snap_depth(t, int(num_slices))
    return float(t)


def composite(
    scene: Sequence[float],
    fog: Sequence[float],
    alpha: float,
    *,
    premultiplied: bool = False,
) -> tuple:
    """Composite fog over the scene in linear space.

    Fog is an absorbing/emitting medium, so it is blended in linear radiance
    before any tonemapping::

        result = scene * (1 - alpha) + fog * alpha

    ``alpha`` is the fog's opacity, i.e. ``1 - transmittance``.  Inputs are
    expected in linear HDR; the result is clamped only at zero so highlights
    survive until the tonemap, which avoids the clipped/blown-out fog colour
    seen when blending in display space.

    With ``premultiplied=True`` the fog colour is assumed to already carry the
    scattering coefficient (``fog = scattered_radiance``) and is added
    directly::

        result = scene * (1 - alpha) + fog

    The operation is applied to the opaque scene *before* transparent objects
    are rendered; do not fold particles into ``scene`` here or they end up
    behind the fog.

    Parameters
    ----------
    scene:
        Linear RGB of the opaque scene.
    fog:
        Linear RGB fog colour (or premultiplied scattered radiance).
    alpha:
        Fog opacity in ``[0, 1]``; values outside the range are clamped.
    premultiplied:
        Treat ``fog`` as premultiplied in-scattered radiance.

    Returns
    -------
    tuple of float
        Linear RGB result.
    """
    scene = list(scene)
    fog = list(fog)
    if len(scene) != len(fog):
        raise ValueError("scene and fog must have the same number of channels")

    opacity = _clamp01(float(alpha))
    transmittance = 1.0 - opacity

    if premultiplied:
        return tuple(
            max(0.0, float(s) * transmittance + float(f))
            for s, f in zip(scene, fog)
        )
    return tuple(
        max(0.0, float(s) * transmittance + float(f) * opacity)
        for s, f in zip(scene, fog)
    )


def composite_transparent(
    scene: Sequence[float],
    transparent: Sequence[float],
    transparent_alpha: float,
    fog_front_alpha: float,
    fog_front_color: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple:
    """Composite a transparent object between the scene and the fog.

    Fog is not one flat layer: it fills the space in front of and behind a
    particle.  Blending transparents into the opaque pass puts every particle
    behind the whole fog volume, so particles look occluded by fog that is
    actually in front of them.  The correct order is::

        opaque scene  ->  fog behind the particle
        particle      ->  blended over that
        fog in front  ->  blended over the particle

    :func:`composite` already produced the scene with fog at the opaque
    depth.  This function inserts the particle and re-applies only the fog
    between the camera and the particle::

        result = (T_particle over scene) * (1 - a_front) + fog_front * a_front

    ``fog_front_alpha`` is the fog opacity integrated from the camera up to the
    particle's own depth, not the pixel's total opacity.  A particle at
    ``d_p`` integrates the medium over ``[near, d_p]``; a particle at a depth
    where nothing is in front passes ``0.0`` and is drawn unfogged.

    Blending stays in linear HDR, so colour and fog are added *before* the
    tonemap rather than being clipped onto a display-space value.

    Parameters
    ----------
    scene:
        Linear RGB of the opaque scene with fog already composited.
    transparent:
        Linear RGB colour of the transparent object (unpremultiplied).
    transparent_alpha:
        Coverage of the transparent object in ``[0, 1]``.
    fog_front_alpha:
        Fog opacity between the camera and the particle depth, in ``[0, 1]``.
    fog_front_color:
        Linear RGB in-scattered radiance of that near fog.

    Returns
    -------
    tuple of float
        Linear RGB result.
    """
    scene = list(scene)
    transparent = list(transparent)
    fog_front = list(fog_front_color)
    if not (len(scene) == len(transparent) == len(fog_front)):
        raise ValueError("scene, transparent and fog_front_color must match in length")

    coverage = _clamp01(float(transparent_alpha))
    front = _clamp01(float(fog_front_alpha))
    transmittance = 1.0 - front

    result = []
    for s, t, f in zip(scene, transparent, fog_front):
        # Particle over the (already fogged, behind-the-particle) scene.
        blended = float(t) * coverage + float(s) * (1.0 - coverage)
        # Near fog over the particle.
        result.append(max(0.0, blended * transmittance + float(f) * front))
    return tuple(result)
