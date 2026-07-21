"""Tests for the bipolar north-fold padding.

The global grids these serve are *tripolar* (South Pole + two Arctic poles); the
north fold itself is *bipolar* -- its seam is the line joining the two northern
poles. A fold padding is requested as a per-axis ``padding`` value on the
(single tile) fold axis -- the meridional "Y" axis -- e.g.
``padding={"X": "periodic", "Y": {"fold": "corner"}}``. The northern edge of
the grid folds onto itself: the seam (zonal "X") axis is mirrored about the pole
and vector components reverse sign. See ``xgcm/padding.py`` for the pivot/offset
conventions.
"""

import numpy as np
import pytest
import xarray as xr

from xgcm import Grid
from xgcm.padding import _resolve_pivot, pad

Nx, Ny = 8, 5


def _make_ds():
    """Build a small dataset with one field per Arakawa staggering.

    The four fields ``c``/``u``/``v``/``q`` cover every combination of the seam
    (zonal "X") and fold (meridional "Y") axes being staggered at a cell center
    vs a cell edge, so a single fixture exercises all four fold sub-cases. Each
    field is filled with ``arange`` values so a mirrored halo is easy to predict.
    """
    ds = xr.Dataset(
        coords={
            "xh": np.arange(Nx),
            "xl": np.arange(Nx),
            "yh": np.arange(Ny),
            "yl": np.arange(Ny),
        }
    )

    def fld(dy, dx):
        """Return a 2-D field on dims ``(dy, dx)`` filled with ``arange`` values."""
        n = ds.sizes[dy] * ds.sizes[dx]
        return xr.DataArray(
            np.arange(n).reshape(ds.sizes[dy], ds.sizes[dx]).astype(float),
            dims=[dy, dx],
        )

    ds["c"] = fld("yh", "xh")  # tracer  (seam=center, fold=center)
    ds["u"] = fld("yh", "xl")  # u-point (seam=edge,   fold=center)
    ds["v"] = fld("yl", "xh")  # v-point (seam=center, fold=edge)
    ds["q"] = fld("yl", "xl")  # corner  (seam=edge,   fold=edge)
    return ds


def _grid(ds, pivot):
    """Build a ``Grid`` with a periodic X seam and a north fold on Y at ``pivot``.

    ``pivot`` is passed straight through as the fold's per-axis padding value, so
    tests can vary the pivot alias (e.g. ``"corner"``, ``"U"``) or an explicit
    ``{axis: position}`` mapping without repeating the boilerplate coords.
    """
    return Grid(
        ds,
        coords={
            "X": {"center": "xh", "left": "xl"},
            "Y": {"center": "yh", "left": "yl"},
        },
        padding={"X": "periodic", "Y": {"fold": pivot}},
        autoparse_metadata=False,
    )


# ---------------------------------------------------------------------------
# Pivot parsing / aliases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "alias, seam, fold",
    [
        ("center", "center", "center"),
        ("T", "center", "center"),
        ("corner", "edge", "edge"),
        ("F", "edge", "edge"),
        ("U", "edge", "center"),
        ("V", "center", "edge"),
    ],
)
def test_pivot_aliases(alias, seam, fold):
    """Each pivot alias resolves to the expected seam/fold cell-position roles.

    Covers the descriptive names (``center``/``corner``) and the Arakawa point
    labels (``T``/``F``/``U``/``V``), asserting the seam and fold axes land on
    ``center`` (cell center) or ``edge`` (cell face) as documented.
    """
    roles = _resolve_pivot(alias, fold_axis="Y", seam_axis="X")
    assert roles == {"seam": seam, "fold": fold}


def test_explicit_pivot_pair_and_left_right_equivalence():
    """An explicit ``{axis: position}`` pivot resolves by role, and left==right.

    Verifies the alternative to a named alias: giving each axis a concrete
    position. It also pins down that ``left`` and ``right`` name the same edge
    sublattice, so they must yield an identical pivot.
    """
    # explicit {axis: position} mapping resolves by role ...
    assert _resolve_pivot({"X": "right", "Y": "center"}, "Y", "X") == {
        "seam": "edge",
        "fold": "center",
    }
    # ... and left vs right name the same (edge) sublattice -> identical pivot
    left = _resolve_pivot({"X": "left", "Y": "left"}, "Y", "X")
    right = _resolve_pivot({"X": "right", "Y": "right"}, "Y", "X")
    assert left == right == {"seam": "edge", "fold": "edge"}


# ---------------------------------------------------------------------------
# Grid-level validation
# ---------------------------------------------------------------------------
def test_fold_requires_periodic_seam():
    """A fold with no periodic companion axis to serve as the seam must raise.

    The fold mirror wraps around the pole along the seam, which only makes sense
    if the seam axis is periodic; here X is ``"fill"``, so construction fails.
    """
    ds = _make_ds()
    with pytest.raises(ValueError, match="periodic seam axis"):
        Grid(
            ds,
            coords={"X": {"center": "xh"}, "Y": {"center": "yh"}},
            padding={"X": "fill", "Y": {"fold": "corner"}},
            autoparse_metadata=False,
        )


def test_fold_seam_axis_inferred():
    """The seam axis is inferred from the sole periodic axis when unspecified.

    With only X periodic, the fold on Y should record ``seam_axis == "X"``
    without the user naming it explicitly.
    """
    ds = _make_ds()
    grid = _grid(ds, "corner")
    assert grid._folds["Y"]["seam_axis"] == "X"


def test_fold_seam_ignores_unspecified_vertical():
    """Seam inference ignores axes that are merely unspecified (not periodic).

    Adds a Z axis with no padding declared; because only an explicitly periodic
    axis is a seam candidate, X remains the unambiguous seam and Z is excluded.
    """
    # A vertical axis left unspecified takes ``padding=None`` (no boundary) and
    # must NOT be mistaken for the seam: only an *explicitly* periodic axis (X)
    # is a candidate. The grid must construct and infer X as the seam.
    ds = _make_ds()
    ds = ds.assign_coords(zh=np.arange(3))
    grid = Grid(
        ds,
        coords={
            "X": {"center": "xh", "left": "xl"},
            "Y": {"center": "yh", "left": "yl"},
            "Z": {"center": "zh"},  # padding unspecified -> None (no boundary)
        },
        padding={"X": "periodic", "Y": {"fold": "corner"}},
        autoparse_metadata=False,
    )
    assert grid._folds["Y"]["seam_axis"] == "X"
    # the unspecified vertical is not periodic, hence not a seam candidate
    assert "Z" not in grid._explicitly_periodic_axes


def test_fold_seam_ambiguous_when_two_explicit_periodic():
    """Two explicitly periodic non-fold axes make the seam ambiguous -> raise.

    When both X and Z are periodic, the fold cannot decide which is its seam, so
    construction must fail rather than pick one silently.
    """
    # two *explicitly* periodic non-fold axes is genuinely ambiguous -> raise.
    ds = _make_ds()
    ds = ds.assign_coords(zh=np.arange(3))
    with pytest.raises(ValueError, match="ambiguous"):
        Grid(
            ds,
            coords={
                "X": {"center": "xh", "left": "xl"},
                "Y": {"center": "yh", "left": "yl"},
                "Z": {"center": "zh"},
            },
            padding={"X": "periodic", "Z": "periodic", "Y": {"fold": "corner"}},
            autoparse_metadata=False,
        )


def test_bad_pivot_raises():
    """An unrecognized pivot alias raises a clear ``Unknown fold pivot`` error."""
    ds = _make_ds()
    with pytest.raises(ValueError, match="Unknown fold pivot"):
        _grid(ds, "banana")


def test_explicit_pivot_invalid_position_raises():
    """A bad position in an explicit pivot mapping raises ``Invalid position``.

    Guards against silently coercing an unknown position (a typo like ``centre``
    or nonsense like ``banana``) into an edge pivot, which would produce a wrong
    halo instead of an error.
    """
    # An invalid position in an explicit {axis: position} mapping must raise,
    # not be silently coerced to an edge pivot (which would give a wrong halo).
    ds = _make_ds()
    with pytest.raises(ValueError, match="Invalid position"):
        _grid(ds, {"X": "centre", "Y": "center"})  # British spelling typo
    with pytest.raises(ValueError, match="Invalid position"):
        _grid(ds, {"X": "banana"})


def test_fold_rejects_face_connections():
    """Declaring both a fold and ``face_connections`` fails at construction.

    The two padding mechanisms are mutually exclusive; the error must surface
    immediately (``NotImplementedError``) rather than cryptically during ``pad``.
    """
    # the fold and face-connection padding paths are mutually exclusive; declaring
    # both must fail clearly at construction, not cryptically at pad time.
    ds = xr.Dataset(
        coords={
            "face": [0, 1],
            "xh": np.arange(Nx),
            "xl": np.arange(Nx),
            "yh": np.arange(Ny),
            "yl": np.arange(Ny),
        }
    )
    fc = {
        "face": {0: {"X": (None, (1, "X", False))}, 1: {"X": ((0, "X", False), None)}}
    }
    with pytest.raises(NotImplementedError, match="face_connections"):
        Grid(
            ds,
            coords={
                "X": {"center": "xh", "left": "xl"},
                "Y": {"center": "yh", "left": "yl"},
            },
            padding={"X": "periodic", "Y": {"fold": "corner"}},
            face_connections=fc,
            autoparse_metadata=False,
        )


@pytest.mark.parametrize("pivot", ["center", "V"])
def test_inner_seam_position_center_pivot_raises(pivot):
    """An ``inner`` seam position under a center-type pivot raises at pad time.

    The ``inner`` staggering has no mirror partner about a cell-center pole, so a
    center/``V`` pivot must give a clear ``NotImplementedError`` instead of an
    opaque ``IndexError``.
    """
    # an `inner` seam position has no mirror partner about a cell-center pole, so a
    # center-type pivot must raise a clear error rather than an opaque IndexError.
    ds = xr.Dataset(
        coords={
            "xh": np.arange(Nx),
            "xi": np.arange(Nx - 1),
            "yh": np.arange(Ny),
            "yl": np.arange(Ny),
        }
    )
    ds["f"] = (("yl", "xi"), np.zeros((Ny, Nx - 1)))
    grid = Grid(
        ds,
        coords={
            "X": {"center": "xh", "inner": "xi"},
            "Y": {"center": "yh", "left": "yl"},
        },
        padding={"X": "periodic", "Y": {"fold": pivot}},
        autoparse_metadata=False,
    )
    with pytest.raises(NotImplementedError, match="inner.*incompatible|incompatible"):
        pad(ds.f, grid, padding_width={"Y": (0, 1)})


# ---------------------------------------------------------------------------
# Per-position halo (explicit expected, the Oceananigans-kernel conventions)
# ---------------------------------------------------------------------------
def test_corner_pivot_all_positions():
    """Corner (F-point) pivot produces the expected halo for all four positions.

    With ``seam=edge, fold=edge`` the mirror combines a reverse with a roll of 1
    on edge-staggered seam dims and skips the redundant top row on edge-staggered
    fold dims. Each of ``c``/``u``/``v``/``q`` is checked against the explicit
    Oceananigans-kernel convention, including the vector sign flip.
    """
    ds = _make_ds()
    grid = _grid(ds, "corner")  # seam=edge, fold=edge

    # scalar c: seam=center -> no roll; fold=center, pivot fold=edge -> no skip
    out = pad(ds.c, grid, padding_width={"Y": (0, 1)})
    np.testing.assert_allclose(out.isel(yh=-1).values, ds.c.isel(yh=-1).values[::-1])

    # u (vector): seam=edge -> reverse+roll(1); fold=center skip 0; sign flip
    out = pad(
        {"X": ds.u}, grid, padding_width={"Y": (0, 1)}, other_component={"Y": ds.v}
    )
    np.testing.assert_allclose(
        out.isel(yh=-1).values, -np.roll(ds.u.isel(yh=-1).values[::-1], 1)
    )

    # v (vector): seam=center -> no roll; fold=edge skip 1; sign flip
    out = pad(
        {"Y": ds.v}, grid, padding_width={"Y": (0, 1)}, other_component={"X": ds.u}
    )
    np.testing.assert_allclose(out.isel(yl=-1).values, -ds.v.isel(yl=-2).values[::-1])

    # q (scalar corner): seam=edge roll(1); fold=edge skip 1
    out = pad(ds.q, grid, padding_width={"Y": (0, 1)})
    np.testing.assert_allclose(
        out.isel(yl=-1).values, np.roll(ds.q.isel(yl=-2).values[::-1], 1)
    )


def test_u_pivot_redundant_row():
    """U-point pivot skips the duplicated seam row only for center-fold fields.

    Under a ``fold=center`` pivot, a center-fold field (``c``) shares its top row
    with the pole and must skip it (sources ``yh=-2``), whereas an edge-fold field
    (``v``) does not coincide with the pole and sources its own top row.
    """
    ds = _make_ds()
    grid = _grid(ds, "U")  # seam=edge, fold=center

    # tracer fold=center == pivot fold -> skip the duplicated top row
    out = pad(ds.c, grid, padding_width={"Y": (0, 1)})
    np.testing.assert_allclose(out.isel(yh=-1).values, ds.c.isel(yh=-2).values[::-1])
    # v fold=edge != pivot fold -> no skip (sources the top row)
    out = pad(ds.v, grid, padding_width={"Y": (0, 1)})
    np.testing.assert_allclose(out.isel(yl=-1).values, ds.v.isel(yl=-1).values[::-1])


# ---------------------------------------------------------------------------
# Independent geometric consistency: tracer (center) and u (edge) must mirror
# about the SAME physical pole line. We sample a generic field of physical x and
# check the fold halo equals that field evaluated at the mirrored location.
# ---------------------------------------------------------------------------
def test_center_and_edge_mirror_same_pole():
    """Center- and edge-staggered seam fields mirror about the same pole line.

    Rather than assert index gymnastics, this samples a smooth physical profile
    ``F(x)`` at cell centers (``i+0.5``) and cell edges (``i``) and checks each
    field's fold halo equals ``F`` evaluated at the reflected physical location
    ``-x mod Nx``. Agreement proves both staggerings fold about one shared pole.
    """
    ds = xr.Dataset(
        coords={"xh": np.arange(Nx), "xl": np.arange(Nx), "yh": np.arange(Ny)}
    )
    # physical seam coordinate: center i at i+0.5, left(edge) i at i
    F = lambda x: np.sin(2 * np.pi * x / Nx) + 0.3 * np.cos(6 * np.pi * x / Nx)
    xc = np.arange(Nx) + 0.5
    xe = np.arange(Nx).astype(float)
    # constant in y, so the top row carries the F profile
    ds["c"] = (("yh", "xh"), np.tile(F(xc), (Ny, 1)))
    ds["u"] = (("yh", "xl"), np.tile(F(xe), (Ny, 1)))
    grid = Grid(
        ds,
        coords={"X": {"center": "xh", "left": "xl"}, "Y": {"center": "yh"}},
        padding={"X": "periodic", "Y": {"fold": "corner"}},
        autoparse_metadata=False,
    )
    # corner pivot -> pole at x=0 -> mirror(x) = -x (periodic on [0, Nx))
    halo_c = pad(ds.c, grid, padding_width={"Y": (0, 1)}).isel(yh=-1).values
    halo_u = pad(ds.u, grid, padding_width={"Y": (0, 1)}).isel(yh=-1).values
    np.testing.assert_allclose(halo_c, F((-xc) % Nx), atol=1e-12)
    np.testing.assert_allclose(halo_u, F((-xe) % Nx), atol=1e-12)


def test_outer_symmetric_memory():
    """Fold mirrors ``outer`` (length N+1) dims correctly despite the dup endpoint.

    MOM6 "symmetric" memory stores cell-edge dims at the ``outer`` position, which
    duplicates the periodic endpoint. Using a smooth ``F(x, y)`` this checks the
    ``v`` and ``q`` fold halos still land on the pole-mirrored physical locations.
    """
    # MOM6 "symmetric" memory uses the `outer` position (length N+1) for the
    # cell-edge dims xq/yq. Fold must mirror those about the pole correctly
    # despite the duplicated periodic endpoint.
    ds = xr.Dataset(
        coords={
            "xh": np.arange(Nx),
            "xq": np.arange(Nx + 1),
            "yh": np.arange(Ny),
            "yq": np.arange(Ny + 1),
        }
    )
    # corner pivot -> X pole at x=0 (mirror -x mod Nx), Y pole at top edge y=Ny
    F = lambda x, y: np.sin(2 * np.pi * x / Nx) + 0.5 * y
    xq, yq = np.arange(Nx + 1), np.arange(Ny + 1)
    ds["q"] = (("yq", "xq"), F(xq[None, :], yq[:, None]))  # both outer
    ds["v"] = (("yq", "xh"), F((np.arange(Nx) + 0.5)[None, :], yq[:, None]))
    grid = Grid(
        ds,
        coords={
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
        },
        padding={"X": "periodic", "Y": {"fold": "corner"}},
        autoparse_metadata=False,
    )
    xc = np.arange(Nx) + 0.5
    halo_v = pad(ds.v, grid, padding_width={"Y": (0, 1)}).isel(yq=-1).values
    np.testing.assert_allclose(halo_v, F((-xc) % Nx, Ny - 1), atol=1e-12)
    halo_q = pad(ds.q, grid, padding_width={"Y": (0, 1)}).isel(yq=-1).values
    np.testing.assert_allclose(
        halo_q, np.array([F((-j) % Nx, Ny - 1) for j in range(Nx + 1)]), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Vector sign: scalar does not flip, vector does
# ---------------------------------------------------------------------------
def test_vector_flips_scalar_does_not():
    """The same array folds with a sign flip as a vector but not as a scalar.

    Padding ``v`` once as a plain scalar and once as the Y component of a vector
    must differ only by the 180-degree sign reversal the fold applies to vectors.
    """
    ds = _make_ds()
    grid = _grid(ds, "corner")
    # same array, once as scalar, once as a vector component
    scal = pad(ds.v, grid, padding_width={"Y": (0, 1)}).isel(yl=-1).values
    vec = (
        pad({"Y": ds.v}, grid, padding_width={"Y": (0, 1)}, other_component={"X": ds.u})
        .isel(yl=-1)
        .values
    )
    np.testing.assert_allclose(vec, -scal)


# ---------------------------------------------------------------------------
# End-to-end operators + dask
# ---------------------------------------------------------------------------
def test_diff_across_seam_runs():
    """A ``diff`` that pads the north edge runs and yields finite results.

    A smoke test: a left->center diff in Y forces a fold pad, so this confirms the
    operator wires through the fold path end-to-end with the expected output dims
    and no NaNs/infs.
    """
    ds = _make_ds()
    grid = _grid(ds, "corner")
    # left->center diff pads the north edge -> exercises the fold
    d = grid.diff(ds.q, "Y")  # q lives on yl -> shifts to yh (center), pads north
    assert d.dims == ("yh", "xl")
    assert np.isfinite(d.values).all()


def test_interp_diff_across_seam_known_answer():
    """`interp` and `diff` across the north fold must equal the same operation on
    a hand-folded (pure-numpy) halo -- a known-answer check that the *operators*,
    not just `pad`, are correct across the seam.

    A left->center shift in Y pads the north edge, so the top output row straddles
    the last interior row and the folded halo row. We build that halo by hand
    (never calling ``pad``) for several pivot/position combinations and finite-
    difference/average against it. The cases deliberately span the distinct fold
    branches: seam=center (mirror is a plain reverse), seam=edge (mirror is a
    non-trivial *roll* -- the case a reverse-only test would miss), skip=1 (the
    top row is the redundant seam row) vs skip=0, and scalar vs vector.
    """
    ds = _make_ds()
    v, q = ds.v.values, ds.q.values  # v on (yl, xh); q on (yl, xl)

    def straddle(field, halo):
        """Return the hand-built interp and diff of ``field`` against ``halo``.

        Appends ``halo`` as the extra top row and computes the center average and
        the difference straddling each adjacent row pair -- the reference answer a
        left->center Y operator must reproduce.
        """
        fp = np.vstack([field, halo[None, :]])  # append halo as the top yl row
        return 0.5 * (fp[:-1] + fp[1:]), fp[1:] - fp[:-1]  # interp(yh), diff(yh)

    # corner pivot: v -> seam=center (reverse), fold=edge -> skip 1 (sources yl=-2)
    grid = _grid(ds, "corner")
    exp_i, exp_d = straddle(v, v[-2][::-1])  # scalar: no sign change
    np.testing.assert_allclose(grid.interp(ds.v, "Y").values, exp_i)
    np.testing.assert_allclose(grid.diff(ds.v, "Y").values, exp_d)
    exp_i, exp_d = straddle(v, -v[-2][::-1])  # vector: + the 180deg sign flip
    oc = {"X": ds.u}
    np.testing.assert_allclose(
        grid.interp({"Y": ds.v}, "Y", other_component=oc).values, exp_i
    )
    np.testing.assert_allclose(
        grid.diff({"Y": ds.v}, "Y", other_component=oc).values, exp_d
    )

    # corner pivot: q -> seam=EDGE, so the mirror is a roll, not a plain reverse
    # (reflect edge index about the pole at x=0: k -> -k mod Nx); fold=edge skip 1.
    halo_q = np.roll(q[-2][::-1], 1)
    exp_i, exp_d = straddle(q, halo_q)  # corner scalar
    np.testing.assert_allclose(grid.interp(ds.q, "Y").values, exp_i)
    np.testing.assert_allclose(grid.diff(ds.q, "Y").values, exp_d)

    # U pivot: v -> fold=center != edge, so skip 0 (sources the top row, yl=-1)
    gridU = _grid(ds, "U")
    exp_i, exp_d = straddle(v, v[-1][::-1])
    np.testing.assert_allclose(gridU.interp(ds.v, "Y").values, exp_i)
    np.testing.assert_allclose(gridU.diff(ds.v, "Y").values, exp_d)


def test_fold_south_edge_respects_per_call_padding():
    """A per-call ``padding`` overrides the south edge while the north still folds.

    The north fold is topological and fixed, but the south is an ordinary
    boundary. Padding both edges shows a per-call ``"extend"`` overrides the
    construction-time south mode, the north halo stays the folded mirror, and the
    default south (no override) remains ``"fill"`` zeros.
    """
    # the north always folds (topology), but the south edge is an ordinary
    # boundary: a per-call `padding` must override the construction-time `south`
    # mode (default "fill"), while the north halo stays the folded mirror.
    ds = _make_ds()
    grid = _grid(ds, "corner")  # default south mode is "fill"
    # pad both edges so we can check south (override) and north (fold) at once
    out = pad(
        ds.c,
        grid,
        padding_width={"Y": (1, 1)},
        padding={"Y": "extend"},
    )
    # south edge: per-call "extend" -> repeats the southern interior row
    np.testing.assert_allclose(out.isel(yh=0).values, ds.c.isel(yh=0).values)
    # north edge: still the folded mirror, unaffected by the per-call boundary
    np.testing.assert_allclose(out.isel(yh=-1).values, ds.c.isel(yh=-1).values[::-1])
    # default (no override) south stays "fill" (zeros), confirming the override
    # above actually changed something
    default = pad(ds.c, grid, padding_width={"Y": (1, 0)})
    np.testing.assert_allclose(default.isel(yh=0).values, 0.0)


def test_multi_row_halo():
    """A multi-row north halo mirrors consecutive interior rows top-down.

    Requesting two halo rows for a center field must source the last two interior
    rows in order (``yh=-1`` then ``yh=-2``), each reversed about the pole.
    """
    ds = _make_ds()
    grid = _grid(ds, "corner")  # center field: skip 0, no roll
    out = pad(ds.c, grid, padding_width={"Y": (0, 2)})
    # consecutive halo rows source consecutive interior rows from the top down
    np.testing.assert_allclose(out.isel(yh=-2).values, ds.c.isel(yh=-1).values[::-1])
    np.testing.assert_allclose(out.isel(yh=-1).values, ds.c.isel(yh=-2).values[::-1])


def test_north_halo_wider_than_interior_raises():
    """Requesting more north halo rows than interior rows raises clearly.

    The fold can mirror at most ``Ny`` interior rows: exactly ``Ny`` still works,
    but ``Ny+1`` must raise an error naming the axis, request, and max available
    rather than silently clamping ``isel`` and returning a too-short array.
    """
    # the fold mirrors interior rows, so it can supply at most as many halo rows
    # as there are interior rows. Requesting more must fail loudly rather than
    # silently clamp `isel` and return a too-short array.
    ds = _make_ds()
    grid = _grid(ds, "corner")  # center field: skip 0 -> Ny interior rows
    # in-range widths still work (boundary, exactly Ny)
    out = pad(ds.c, grid, padding_width={"Y": (0, Ny)})
    assert out.sizes["yh"] == 2 * Ny
    # one row too many -> clear error naming axis, request, and max available
    with pytest.raises(ValueError, match="exceeds the .* interior row"):
        pad(ds.c, grid, padding_width={"Y": (0, Ny + 1)})


def test_fold_with_simultaneous_seam_padding():
    """Padding the seam and folding the north in one call compose correctly.

    When X and Y are padded together, the folded north row must itself be wrapped
    periodically along the seam, so its halo equals the mirrored row bracketed by
    its periodic neighbors.
    """
    ds = _make_ds()
    grid = _grid(ds, "corner")
    out = pad(ds.c, grid, padding_width={"X": (1, 1), "Y": (0, 1)})
    assert out.shape == (Ny + 1, Nx + 2)
    # the folded north row is itself wrapped periodically along the seam axis
    base = ds.c.isel(yh=-1).values[::-1]
    expected = np.concatenate([[base[-1]], base, [base[0]]])
    np.testing.assert_allclose(out.isel(yh=-1).values, expected)


@pytest.mark.parametrize("chunks", [{"xh": Nx, "xl": Nx}, {"xh": 3, "xl": 3}])
def test_fold_dask_matches_numpy(chunks):
    """The fold pad on dask-backed data stays lazy and matches the numpy result.

    Runs the same corner fold on eager and chunked datasets (including chunks that
    split the seam axis) and asserts the output is a dask collection whose
    computed values equal the numpy reference.
    """
    ds = _make_ds()
    grid_np = _grid(ds, "corner")
    expected = pad(ds.c, grid_np, padding_width={"Y": (0, 1)})

    dsc = ds.chunk(chunks)
    grid_da = _grid(dsc, "corner")
    out = pad(dsc.c, grid_da, padding_width={"Y": (0, 1)})
    import dask

    assert dask.is_dask_collection(out.data)
    np.testing.assert_allclose(out.compute().values, expected.values)
