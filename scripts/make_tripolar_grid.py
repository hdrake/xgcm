"""Generate the tripolar-grid hero figure for ``docs/grid_topology.md``.

A single publication-quality figure is written to ``docs/images``:

* ``tripolar_grid.png`` -- an *idealized* tripolar grid rendered on a shaded
  3-D globe (hand-rolled orthographic projection, pure numpy + matplotlib).
  South of a join latitude the mesh is an ordinary spherical lat-lon grid;
  north of it a **bipolar Arctic cap** carries the grid's two displaced
  northern poles, joined by the **bipolar seam**. Together with the ordinary
  South Pole those are the three singularities that make the grid *tripolar*.

The cap is a faithful port of the Murray (1996) / MOM6 bipolar projection
(``ocean_grid_generator.py``, Niki Zadeh, NOAA-GFDL). A *logical* regular
spherical cap -- columns at the very same longitudes as the regular meridians,
rows running from the join latitude ``PHI_J`` up to the logical North Pole --
is carried conformally onto the sphere so that the map is the **identity on
the join parallel** (a point at ``PHI_J`` maps to itself). Consequently every
cap meridian continues its southern meridian with no gap, the join circle is a
single shared grid line, and the single logical North-Pole point opens into the
**bipolar seam** running between the two displaced poles. No map data, cartopy
or basemap is used -- the globe is shaded and projected by hand so the script
has only numpy + matplotlib as dependencies.

    python scripts/make_tripolar_grid.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

# ---- palette -------------------------------------------------------------
INK = "#22303f"  # near-black for outlines / limb
FACE = "#c9d4de"  # cell-face gridlines
ACCENT = "#0b6e4f"  # join-latitude circle (cap vs. regular divide)
POLE = "#f4a300"  # the two Arctic poles (stars)
SEAM = "#b3402a"  # the bipolar seam joining the two poles
OCEAN_DEEP = np.array([0.09, 0.21, 0.35])  # shaded (limb) ocean
OCEAN_LITE = np.array([0.56, 0.74, 0.86])  # sunlit ocean
GRAT = "#eef4f9"  # muted regular graticule on the sphere
CAP = "#12212e"  # emphasised bipolar-cap gridlines

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.linewidth": 0.0,
        "savefig.dpi": 200,
        "figure.dpi": 120,
    }
)

# ---- grid geometry parameters -------------------------------------------
PHI_J = 65.0  # join latitude (deg N): regular grid below, bipolar cap above
LON_BP = 0.0  # bipolar longitude: seam / the two displaced poles lie here & +180
D_MERID = 30.0  # regular meridian spacing (deg)
D_PARAL = 15.0  # regular parallel spacing (deg)

# In the Murray/MOM6 bipolar projection the displaced poles sit *on* the join
# parallel, diametrically opposite in longitude at LON_BP and LON_BP + 180.
POLE_LONS = (LON_BP, LON_BP + 180.0)

# ---- viewpoint (oblique orthographic) -----------------------------------
CAM_LAT = 58.0  # camera latitude (deg N): high enough to look down onto the cap
CAM_LON = -90.0  # camera longitude: puts the two poles side-by-side on screen
LIGHT_DIR = np.array([-0.45, 0.35, 0.82])  # sun direction (upper-left, high)


# =========================================================================
# projection helpers
# =========================================================================
def _unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def sphere_xyz(lon, lat):
    """(lon, lat) in degrees -> unit-sphere Cartesian, stacked on last axis."""
    la, ph = np.radians(lon), np.radians(lat)
    return np.stack(
        [np.cos(ph) * np.cos(la), np.cos(ph) * np.sin(la), np.sin(ph)], axis=-1
    )


# camera basis: view vector v, plus screen right/up orthonormal to it.
V = _unit(sphere_xyz(CAM_LON, CAM_LAT))
RIGHT = _unit(np.cross([0, 0, 1.0], V))
UP = np.cross(V, RIGHT)
LIGHT = _unit(LIGHT_DIR)


def project(lon, lat):
    """Orthographic screen coords; back-hemisphere points become NaN so that
    polylines are split at the horizon and hidden lines drop out."""
    p = sphere_xyz(np.asarray(lon, float), np.asarray(lat, float))
    sx = p @ RIGHT
    sy = p @ UP
    back = (p @ V) <= 0.0
    sx = np.where(back, np.nan, sx)
    sy = np.where(back, np.nan, sy)
    return sx, sy


# =========================================================================
# bipolar cap construction -- Murray (1996) / MOM6 conformal projection
# =========================================================================
# Port of ``bipolar_projection`` from NOAA-GFDL ``ocean_grid_generator.py``
# (Niki Zadeh). A regular spherical cap (logical coordinates ``lamg``, ``phig``)
# is mapped conformally onto the sphere. With
#     rp = tan((90 - PHI_J) / 2)
# the map is the *identity* on the join parallel phig = PHI_J and folds the
# single logical North Pole into the bipolar seam between the two displaced
# poles at (LON_BP, PHI_J) and (LON_BP + 180, PHI_J).
PI_180 = np.pi / 180.0
HUGE = 1.0e30
RP = np.tan(0.5 * (90.0 - PHI_J) * PI_180)  # displaced-pole / join stereographic radius


def _mdist(x1, x2):
    """Positive angular distance modulo 360 deg."""
    return np.minimum(np.mod(x1 - x2, 360.0), np.mod(x2 - x1, 360.0))


def bipolar_projection(lamg, phig, lon_bp=LON_BP, rp=RP):
    """Map logical cap coords (lamg, phig) [deg] -> geographic (lon, lat) [deg].

    ``lamg`` spans a full 360 deg starting at ``lon_bp``; ``phig`` runs from the
    join latitude ``PHI_J`` up to 90. On the join parallel the map is the
    identity, so cap grid lines join the regular grid with no discontinuity.
    """
    lamg = np.asarray(lamg, float)
    phig = np.asarray(phig, float)
    # symmetry-meridian resolution reparametrisation (maps PHI_J -> 0, 90 -> 90)
    phig = 90 - 2 * np.arctan(np.tan(0.5 * (90 - phig) * PI_180) / rp) / PI_180
    tmp = _mdist(lamg, lon_bp) * PI_180
    sinla = np.sin(tmp)
    sphig = np.sin(phig * PI_180)
    alpha2 = (np.cos(tmp)) ** 2
    beta2_inv = (np.tan(phig * PI_180)) ** 2
    rden = 1.0 / (1.0 + alpha2 * beta2_inv)

    B = sinla * np.sqrt(rden)
    B = np.where(np.abs(beta2_inv) > HUGE, 0.0, B)
    lamc = np.arcsin(np.clip(B, -1.0, 1.0)) / PI_180
    # pick the correct root of arcsin(B) so lamc is continuous around the cap
    dl = lamg - lon_bp
    lamc = np.where((dl > 90) & (dl <= 180), 180 - lamc, lamc)
    lamc = np.where((dl > 180) & (dl <= 270), 180 + lamc, lamc)
    lamc = np.where(dl > 270, 360 - lamc, lamc)
    lamc = np.where(dl == 90, 90, lamc)  # along symmetry meridian
    lamc = np.where(dl == 270, 270, lamc)
    lams = lamc + lon_bp

    A = sinla * sphig
    chic = np.arccos(np.clip(A, -1.0, 1.0))
    phis = 90 - 2 * np.arctan(rp * np.tan(chic / 2)) / PI_180
    return lams, phis


def cap_meridians(n_pts=400):
    """Cap pseudo-meridians: constant-``lamg`` curves from the join up to a
    pole. Uses the *same* longitudes as the regular meridians so each one
    continues its southern partner. The two pole longitudes are degenerate
    (they collapse to the displaced poles) and so are skipped."""
    phig = np.linspace(PHI_J, 90.0, n_pts)
    out = []
    for lon in np.arange(0.0, 360.0, D_MERID):
        if np.any(np.isclose(_mdist(lon, np.array(POLE_LONS)), 0.0)):
            continue
        lamg = np.full_like(phig, lon)
        out.append(bipolar_projection(lamg, phig))
    return out


def cap_parallels(phigs=(70.5, 76.0, 81.0, 85.0, 88.0), n_pts=800):
    """Cap pseudo-parallels: constant-``phig`` curves (the nested bipolar ovals
    around the two displaced poles). ``phig = PHI_J`` is the join circle itself
    and is drawn separately, so it is excluded here."""
    lamg = np.linspace(0.0, 360.0, n_pts)
    out = []
    for phig in phigs:
        out.append(bipolar_projection(lamg, np.full_like(lamg, phig)))
    return out


def seam_curve(n_pts=300):
    """The bipolar seam: the fold of the logical North-Pole row. It runs up the
    LON_BP meridian from PHI_J to the geographic North Pole and back down the
    LON_BP + 180 meridian to PHI_J."""
    up_lat = np.linspace(PHI_J, 90.0, n_pts)
    dn_lat = np.linspace(90.0, PHI_J, n_pts)
    lon = np.concatenate([np.full(n_pts, POLE_LONS[0]), np.full(n_pts, POLE_LONS[1])])
    lat = np.concatenate([up_lat, dn_lat])
    return lon, lat


# =========================================================================
# regular spherical lat-lon grid (south of the join latitude)
# =========================================================================
def regular_meridians(n_pts=400):
    out = []
    for lon in np.arange(0.0, 360.0, D_MERID):
        lat = np.linspace(-88.0, PHI_J, n_pts)
        out.append((np.full_like(lat, lon), lat))
    return out


def regular_parallels(n_pts=600):
    out = []
    lon = np.linspace(0.0, 360.0, n_pts)
    for lat in np.arange(-75.0, PHI_J, D_PARAL):
        out.append((lon, np.full_like(lon, lat)))
    return out


def join_circle(n_pts=600):
    lon = np.linspace(0.0, 360.0, n_pts)
    return lon, np.full_like(lon, PHI_J)


# =========================================================================
# globe shading
# =========================================================================
def shaded_globe_image(n=900):
    """Lambert-shaded ocean disk as an RGBA image on screen coords [-1, 1]."""
    g = np.linspace(-1.0, 1.0, n)
    SX, SY = np.meshgrid(g, g)
    r2 = SX**2 + SY**2
    inside = r2 <= 1.0
    zc = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    # front-hemisphere surface point for every pixel of the disk
    P = SX[..., None] * RIGHT + SY[..., None] * UP + zc[..., None] * V
    lam = np.clip(P @ LIGHT, 0.0, 1.0)
    shade = 0.30 + 0.70 * lam  # ambient + diffuse
    # subtle limb darkening for a rounder read
    shade *= 0.80 + 0.20 * zc
    rgb = OCEAN_DEEP + (OCEAN_LITE - OCEAN_DEEP) * shade[..., None]
    img = np.zeros((n, n, 4))
    img[..., :3] = np.clip(rgb, 0, 1)
    img[..., 3] = inside.astype(float)
    return img


# =========================================================================
# numeric acceptance test: cap grid must join the regular grid exactly
# =========================================================================
def verify_join_continuity():
    """Assert C0 continuity of every cap meridian with its southern partner at
    the join latitude, and that the outermost cap parallel is exactly PHI_J.
    Prints the worst mismatch (must be ~0)."""
    # every regular meridian, evaluated at the join, must map to itself
    lons = np.arange(0.0, 360.0, D_MERID)
    lams, phis = bipolar_projection(lons, np.full_like(lons, PHI_J))
    lon_err = np.max(_mdist(lams, lons))
    lat_err = np.max(np.abs(phis - PHI_J))
    # the cap-parallel system evaluated at phig = PHI_J is the join circle
    plon = np.linspace(0.0, 360.0, 720)
    _, pphi = bipolar_projection(plon, np.full_like(plon, PHI_J))
    ring_err = np.max(np.abs(pphi - PHI_J))
    worst = max(lon_err, lat_err, ring_err)
    print(  # noqa: T201
        f"join continuity: max meridian lon mismatch = {lon_err:.3e} deg, "
        f"lat mismatch = {lat_err:.3e} deg, join-ring lat error = {ring_err:.3e} deg"
    )
    print(f"  --> worst join mismatch = {worst:.3e} deg")  # noqa: T201
    assert worst < 1e-6, f"cap does not meet regular grid at the join: {worst:.3e}"
    return worst


# =========================================================================
# figure
# =========================================================================
def _draw_lines(ax, polylines, **kw):
    for lon, lat in polylines:
        sx, sy = project(lon, lat)
        ax.plot(sx, sy, **kw)


def fig_tripolar():
    fig, ax = plt.subplots(figsize=(9.4, 9.6))

    # shaded sphere body + crisp limb
    ax.imshow(
        shaded_globe_image(),
        extent=(-1.0, 1.0, -1.0, 1.0),
        origin="lower",
        interpolation="bilinear",
        zorder=0,
    )
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=INK, lw=2.2, zorder=6))

    # regular spherical grid (muted)
    _draw_lines(ax, regular_meridians(), color=GRAT, lw=0.9, alpha=0.85, zorder=2)
    _draw_lines(ax, regular_parallels(), color=GRAT, lw=0.9, alpha=0.85, zorder=2)

    # bipolar cap grid (emphasised) -- drawn below the join ring so the shared
    # join parallel reads as a single crisp line on top
    _draw_lines(ax, cap_parallels(), color=CAP, lw=1.15, alpha=0.9, zorder=3)
    _draw_lines(ax, cap_meridians(), color=CAP, lw=1.15, alpha=0.9, zorder=3)

    # join-latitude circle: the single shared cap / regular-grid divide
    jlon, jlat = join_circle()
    jsx, jsy = project(jlon, jlat)
    ax.plot(jsx, jsy, color=ACCENT, lw=2.4, zorder=4)

    # bipolar seam
    slon, slat = seam_curve()
    ssx, ssy = project(slon, slat)
    ax.plot(ssx, ssy, color=SEAM, lw=3.0, zorder=5, solid_capstyle="round")

    # the two displaced Arctic poles as stars (they sit on the join circle)
    pole_screen = []
    for pole_lon in POLE_LONS:
        px, py = project(pole_lon, PHI_J)
        pole_screen.append((float(px), float(py)))
        ax.scatter(
            px,
            py,
            s=360,
            marker="*",
            facecolor=POLE,
            edgecolor=INK,
            linewidth=1.3,
            zorder=8,
        )

    # ---- annotations -----------------------------------------------------
    # a soft white plate keeps label text legible where it grazes the globe
    def _plate(fc="white", alpha=0.78):
        return dict(boxstyle="round,pad=0.28", fc=fc, ec="none", alpha=alpha)

    # one tidy callout for both displaced poles, with a thin leader to each star.
    # the label sits in the white margin above the globe; the leaders angle out
    # to the two stars where they perch on the join circle.
    label_xy = (0.5, 0.905)
    for sx, sy in pole_screen:
        if np.isfinite(sx) and np.isfinite(sy):
            ax.annotate(
                "",
                (sx, sy),
                xytext=label_xy,
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-",
                    color=INK,
                    lw=0.8,
                    alpha=0.55,
                    shrinkA=14,
                    shrinkB=9,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=7,
            )
    ax.annotate(
        "the two displaced poles\nof the bipolar seam",
        xy=label_xy,
        xytext=label_xy,
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
        fontweight="bold",
        bbox=_plate(),
        zorder=9,
    )
    # seam label: point at a spot on the seam near the geographic North Pole
    spx = project(POLE_LONS[0], 84.0)
    ax.annotate(
        "bipolar seam",
        (float(spx[0]), float(spx[1])),
        xytext=(0.10, 0.86),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=10.5,
        color=SEAM,
        fontweight="bold",
        bbox=_plate(),
        arrowprops=dict(arrowstyle="-", color=SEAM, lw=1.2, shrinkA=2, shrinkB=6),
        zorder=9,
    )
    # join latitude label: short angled leader to the left side of the join circle
    jx, jy = project(CAM_LON - 62.0, PHI_J)
    ax.annotate(
        f"join latitude ~{PHI_J:.0f}°N\n(cap ↔ regular grid)",
        (float(jx), float(jy)),
        xytext=(0.045, 0.56),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=9.5,
        color=ACCENT,
        fontweight="bold",
        bbox=_plate(),
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.0, shrinkA=2, shrinkB=6),
        zorder=9,
    )
    ax.text(
        0.5,
        0.045,
        "south of the join latitude: an ordinary spherical lat–lon grid",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color="0.30",
        style="italic",
        zorder=9,
    )

    # title
    ax.set_title(
        "An idealized tripolar grid — bipolar Arctic cap",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    lim = 1.14
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    fig.tight_layout()
    out = OUT / "tripolar_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)  # noqa: T201


if __name__ == "__main__":
    verify_join_continuity()
    fig_tripolar()
