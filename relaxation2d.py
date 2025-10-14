import numpy as np
import matplotlib.pyplot as plt
import tqdm

def make_box(shape, bounds):
    """
    Return coordinates and outward normals for the rectangular box boundary only.

    Parameters
    - shape: tuple (ny, nx) grid shape
    - bounds: tuple (y0, y1, x0, x1) box indices with y in [y0, y1) and x in [x0, x1)

    Returns
    - coords: ndarray shape (N,2) of (y, x) integer grid coordinates on the boundary
    - normals: ndarray shape (N,2) of float unit normals (dy, dx) for each boundary coordinate.
    """
    ny, nx = shape
    y0, y1, x0, x1 = bounds

    # Clip bounds to grid and ensure valid box
    y0 = int(np.clip(y0, 0, ny))
    y1 = int(np.clip(y1, 0, ny))
    x0 = int(np.clip(x0, 0, nx))
    x1 = int(np.clip(x1, 0, nx))
    if y1 <= y0 or x1 <= x0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=float)

    normals_map = {}

    # top edge (outward dy = -1)
    for x in range(x0, x1):
        key = (y0, x)
        normals_map.setdefault(key, np.array([0.0, 0.0]))
        normals_map[key] += np.array([-1.0, 0.0])

    # bottom edge (outward dy = +1)
    for x in range(x0, x1):
        key = (y1 - 1, x)
        normals_map.setdefault(key, np.array([0.0, 0.0]))
        normals_map[key] += np.array([1.0, 0.0])

    # left edge (outward dx = -1)
    for y in range(y0, y1):
        key = (y, x0)
        normals_map.setdefault(key, np.array([0.0, 0.0]))
        normals_map[key] += np.array([0.0, -1.0])

    # right edge (outward dx = +1)
    for y in range(y0, y1):
        key = (y, x1 - 1)
        normals_map.setdefault(key, np.array([0.0, 0.0]))
        normals_map[key] += np.array([0.0, 1.0])

    # build arrays
    coords_list = []
    normals_list = []
    for (y, x), n in normals_map.items():
        coords_list.append((y, x))
        normals_list.append(n)

    coords = np.array(coords_list, dtype=int)
    normals = np.array(normals_list, dtype=float)

    # normalize normals to unit length (corners get diagonal unit vectors)
    lengths = np.linalg.norm(normals, axis=1)
    nz = lengths > 0
    normals[nz] /= lengths[nz][:, None]

    return coords, normals

def make_circle(shape, center, radius, thickness=0.5):
    """
    Return integer grid coordinates and outward unit normals for a circular boundary.

    Parameters
    - shape: (ny, nx)
    - center: (yc, xc) in grid coordinates (can be float)
    - radius: radius in grid units
    - thickness: half-width tolerance for selecting boundary pixels (default 0.5)

    Returns
    - coords: (N,2) array of (y,x) integer coordinates on the circle boundary
    - normals: (N,2) array of outward unit normals (dy, dx) for each coord
    """
    ny, nx = shape
    yc, xc = float(center[0]), float(center[1])
    ys = np.arange(ny)
    xs = np.arange(nx)
    Y, X = np.meshgrid(ys, xs, indexing='ij')  # Y.shape = (ny,nx)
    dist = np.sqrt((Y - yc)**2 + (X - xc)**2)
    mask = np.abs(dist - radius) <= thickness

    if not np.any(mask):
        return np.empty((0,2), dtype=int), np.empty((0,2), dtype=float)

    coords_idx = np.argwhere(mask)  # (y,x)
    dy = (coords_idx[:,0].astype(float) - yc)
    dx = (coords_idx[:,1].astype(float) - xc)
    lengths = np.hypot(dy, dx)
    # avoid division by zero (shouldn't happen because dist ~ radius > 0)
    nz = lengths > 0
    normals = np.zeros_like(coords_idx, dtype=float)
    normals[nz, 0] = dy[nz] / lengths[nz]
    normals[nz, 1] = dx[nz] / lengths[nz]

    return coords_idx.astype(int), normals

def relax_2d(f, iters=1000):
    f = f.copy()
    ny, nx = f.shape

    def _bilinear_interp(arr, yf, xf):
        # bilinear interpolation of arr at fractional coordinates (yf, xf)
        yf = np.asarray(yf)
        xf = np.asarray(xf)
        y0 = np.floor(yf).astype(int)
        x0 = np.floor(xf).astype(int)
        y1 = np.clip(y0 + 1, 0, ny - 1)
        x1 = np.clip(x0 + 1, 0, nx - 1)
        y0 = np.clip(y0, 0, ny - 1)
        x0 = np.clip(x0, 0, nx - 1)

        wy = yf - y0
        wx = xf - x0

        return ((1 - wy) * (1 - wx) * arr[y0, x0]
                + (1 - wy) * wx * arr[y0, x1]
                + wy * (1 - wx) * arr[y1, x0]
                + wy * wx * arr[y1, x1])

    # grid spacing (assumed uniform). If you use physical spacing set h accordingly.
    h = 1.0

    for it in tqdm.tqdm(range(iters)):
        # Update interior points using the average of neighbors
        f[1:-1, 1:-1] = 0.25 * (f[1:-1, :-2] + f[1:-1, 2:] +
                                f[:-2, 1:-1] + f[2:, 1:-1])

        # Reapply Dirichlet potential surfaces
        for (coords, _), V in potential_surfaces:
            f[coords[:, 0], coords[:, 1]] = V

        # Enforce charged-surface condition for arbitrary normals.
        # Use centered one-step finite differences on both sides so we don't assume
        # the outside normal derivative is negligible.
        # From Gauss: eps0 (E_n,out - E_n,in) = sigma, E_n = -dV/dn
        # => dV_in/dn - dV_out/dn = sigma/eps0
        # Finite differences (h spacing):
        #   (V_surface - V_in)/h - (V_out - V_surface)/h = sigma/eps0
        # => 2 V_surface = V_in + V_out + (h * sigma / eps0)
        # => V_surface = 0.5*(V_in + V_out) + (h * sigma) / (2 * eps0)
        for (coords, normals), sigma in charged_surfaces:
            if coords.size == 0:
                continue

            # Only use boundary points where the normal is non-zero.
            lengths = np.linalg.norm(normals, axis=1)
            boundary_mask = lengths > 1e-12
            if not np.any(boundary_mask):
                continue

            b_coords = coords[boundary_mask]
            b_normals = normals[boundary_mask]

            ys = b_coords[:, 0].astype(float)
            xs = b_coords[:, 1].astype(float)
            nys = b_normals[:, 0].astype(float)
            nxs = b_normals[:, 1].astype(float)

            # inward and outward sample positions (one grid spacing along normal)
            yin = ys - nys * h
            xin = xs - nxs * h
            yout = ys + nys * h
            xout = xs + nxs * h

            # Clip sampling positions to grid to avoid indexing outside
            yin = np.clip(yin, 0, ny - 1)
            xin = np.clip(xin, 0, nx - 1)
            yout = np.clip(yout, 0, ny - 1)
            xout = np.clip(xout, 0, nx - 1)

            Vin = _bilinear_interp(f, yin, xin)
            Vout = _bilinear_interp(f, yout, xout)

            # set boundary potential from averaged interior & exterior samples
            f[b_coords[:, 0], b_coords[:, 1]] = 0.5 * (Vin + Vout) + (h * sigma) / (2.0 * e0)

    return f

def graph_potential(potential, potential_surfaces=None, charged_surfaces=None):
    """
    Plot potential contours and overlay surfaces:
    - potential_surfaces: list of ((coords, normals), V) -> plotted in yellow (Dirichlet)
    - charged_surfaces: list of ((coords, normals), sigma) -> plotted in red with normals as quiver
    """
    plt.figure()
    plt.contour(potential, np.arange(-1, 11, 0.5), cmap='viridis')
    plt.colorbar(label='Potential (V)')
    plt.title('Relaxed Potential Contours')
    plt.xlabel('x')
    plt.ylabel('y')

    handles = []
    labels = []

    # Plot Dirichlet potential surfaces (filled or boundary points)
    if potential_surfaces:
        for (coords, _), V in potential_surfaces:
            if coords.size == 0:
                continue
            xs = coords[:, 1]
            ys = coords[:, 0]
            # use a square marker and semi-transparent yellow so contours remain visible
            sc = plt.scatter(xs, ys, c='yellow', s=8, edgecolors='k', linewidths=0.2, alpha=0.8)
            handles.append(sc)
            labels.append(f'V={V}')

    # Plot charged surfaces (boundary points) and normals
    if charged_surfaces:
        for (coords, normals), sigma in charged_surfaces:
            if coords.size == 0:
                continue
            xs = coords[:, 1]
            ys = coords[:, 0]
            sc = plt.scatter(xs, ys, c='red', s=10, alpha=0.9)
            handles.append(sc)
            labels.append(f'sigma={sigma}')

            # plot a subsampled quiver of normals for clarity
            if normals.size:
                # subsample to at most ~200 arrows
                N = coords.shape[0]
                step = max(1, N // 200)
                idx = np.arange(0, N, step)
                plt.quiver(xs[idx], ys[idx], normals[idx, 1], normals[idx, 0],
                           color='white', scale=20, width=0.004, headwidth=3, alpha=0.85)

    if handles:
        plt.legend(handles, labels, loc='upper right', fontsize='small')

    plt.show()


e0 = 1

potential = np.zeros((400, 400))

charged_surfaces = [
    (make_box(potential.shape, (50, 350, 50, 350)), 1),
]

potential_surfaces = [
    (make_box(potential.shape, (125, 275, 125, 275)), 1),
]

relaxed = relax_2d(potential, iters=2000)

graph_potential(relaxed, potential_surfaces, charged_surfaces)