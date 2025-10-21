import numpy as np
import matplotlib.pyplot as plt
import tqdm

def make_box3d(shape, bounds):
    """
    Return coordinates and outward normals for the rectangular box boundary only.

    shape: (nz, ny, nx)
    bounds: (z0, z1, y0, y1, x0, x1) with half-open intervals [z0,z1),...
    Returns coords as (N,3) with (z,y,x) and normals (N,3) outward unit vectors.
    """
    nz, ny, nx = shape
    z0, z1, y0, y1, x0, x1 = bounds
    z0 = int(np.clip(z0, 0, nz))
    z1 = int(np.clip(z1, 0, nz))
    y0 = int(np.clip(y0, 0, ny))
    y1 = int(np.clip(y1, 0, ny))
    x0 = int(np.clip(x0, 0, nx))
    x1 = int(np.clip(x1, 0, nx))
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return np.empty((0,3), int), np.empty((0,3), float)

    normals_map = {}

    # z-faces
    for y in range(y0, y1):
        for x in range(x0, x1):
            key = (z0, y, x)
            normals_map.setdefault(key, np.array([0.0,0.0,0.0]))
            normals_map[key] += np.array([-1.0,0.0,0.0])

            key = (z1-1, y, x)
            normals_map.setdefault(key, np.array([0.0,0.0,0.0]))
            normals_map[key] += np.array([1.0,0.0,0.0])
    # y-faces
    for z in range(z0, z1):
        for x in range(x0, x1):
            key = (z, y0, x)
            normals_map.setdefault(key, np.array([0.0, 0.0, 0.0]))
            normals_map[key] += np.array([0.0, -1.0, 0.0])

            key = (z, y1 - 1, x)
            normals_map.setdefault(key, np.array([0.0, 0.0, 0.0]))
            normals_map[key] += np.array([0.0, 1.0, 0.0])
    # x-faces
    for z in range(z0, z1):
        for y in range(y0, y1):
            key = (z, y, x0)
            normals_map.setdefault(key, np.array([0.0, 0.0, 0.0]))
            normals_map[key] += np.array([0.0, 0.0, -1.0])

            key = (z, y, x1 - 1)
            normals_map.setdefault(key, np.array([0.0, 0.0, 0.0]))
            normals_map[key] += np.array([0.0, 0.0, 1.0])

    coords_list = []
    normals_list = []
    for (z,y,x), n in normals_map.items():
        coords_list.append((z,y,x))
        normals_list.append(n)
    coords = np.array(coords_list, dtype=int)
    normals = np.array(normals_list, dtype=float)
    lengths = np.linalg.norm(normals, axis=1)
    nz_mask = lengths > 0
    normals[nz_mask] /= lengths[nz_mask][:,None]
    return coords, normals

def make_sphere(shape, center, radius, thickness=0.6):
    """
    Return coords (z,y,x) and outward normals for a spherical shell boundary.
    center: (zc,yc,xc) floats. radius in grid units.
    thickness: tolerance for pixel selection
    """
    nz, ny, nx = shape
    zc, yc, xc = float(center[0]), float(center[1]), float(center[2])

    Z = np.arange(nz)[:,None,None]
    Y = np.arange(ny)[None,:,None]
    X = np.arange(nx)[None,None,:]

    dist = np.sqrt((Z - zc)**2 + (Y - yc)**2 + (X - xc)**2)
    mask = np.abs(dist - radius) <= thickness
    if not np.any(mask):
        return np.empty((0,3), int), np.empty((0,3), float)

    coords = np.argwhere(mask)  # (z,y,x)

    dz = coords[:,0].astype(float) - zc
    dy = coords[:,1].astype(float) - yc
    dx = coords[:,2].astype(float) - xc

    lengths = np.hypot(np.hypot(dz, dy), dx)
    normals = np.zeros_like(coords, dtype=float)
    nz_mask = lengths > 0

    normals[nz_mask,0] = dz[nz_mask] / lengths[nz_mask]
    normals[nz_mask,1] = dy[nz_mask] / lengths[nz_mask]
    normals[nz_mask,2] = dx[nz_mask] / lengths[nz_mask]

    return coords.astype(int), normals

def interp3d(arr, zf, yf, xf):
    """
    Nearest-neighbor sampling of arr at positions zf,yf,xf.
    Less accurate than trilinear interpolation but simpler and faster.
    Accepts scalars or arrays for zf,yf,xf and returns values taken from
    the nearest integer grid points (clipped to array bounds).
    """
    zf = np.asarray(zf)
    yf = np.asarray(yf)
    xf = np.asarray(xf)
    nz, ny, nx = arr.shape

    # Round to nearest grid index and clip to valid range
    zi = np.rint(zf).astype(int)
    yi = np.rint(yf).astype(int)
    xi = np.rint(xf).astype(int)

    zi = np.clip(zi, 0, nz - 1)
    yi = np.clip(yi, 0, ny - 1)
    xi = np.clip(xi, 0, nx - 1)

    return arr[zi, yi, xi]

# add: numerical Laplacian (6-point stencil)
def compute_laplacian_3d(f, h=1.0):
    """
    Compute the 3D discrete Laplacian using a 6-point stencil.
    Returns array same shape as f.
    """
    lap = np.zeros_like(f)
    lap[1:-1,1:-1,1:-1] = (
        (f[2:  ,1:-1,1:-1] - 2*f[1:-1,1:-1,1:-1] + f[:-2,1:-1,1:-1]) +
        (f[1:-1,2:  ,1:-1] - 2*f[1:-1,1:-1,1:-1] + f[1:-1,:-2,1:-1]) +
        (f[1:-1,1:-1,2:  ] - 2*f[1:-1,1:-1,1:-1] + f[1:-1,1:-1,:-2])
    ) / (h*h)
    return lap

def relax_3d(f, potential_surfaces, charged_surfaces, iters=2000, e0=1.0, h=1.0):
    """
    3D relaxation (Laplace/Poisson) on grid f (array shape nz,ny,nx).
    potential_surfaces: list of ((coords, normals), V) where coords shape (N,3), V scalar
    charged_surfaces: list of ((coords, normals), sigma) where normals are outward unit vectors
    """
    f = f.copy()
    nz, ny, nx = f.shape

    for _ in tqdm.tqdm(range(iters)):
        # 6-point average for interior
        f[1:-1,1:-1,1:-1] = (f[1:-1,1:-1,:-2] + f[1:-1,1:-1,2:] +
                             f[1:-1,:-2,1:-1] + f[1:-1,2:,1:-1] +
                             f[:-2,1:-1,1:-1] + f[2:,1:-1,1:-1]) / 6.0

        # Dirichlet surfaces: set fixed potentials
        for (coords, _), V in (potential_surfaces or []):
            if coords.size == 0:
                continue

            zs = coords[:,0]
            ys = coords[:,1]
            xs = coords[:,2]

            f[zs, ys, xs] = V

        # Charged surfaces: enforce jump in normal derivative
        for (coords, normals), sigma in (charged_surfaces or []):
            if coords.size == 0:
                continue

            # Only use boundary points where the normal is non-zero.
            lengths = np.linalg.norm(normals, axis=1)
            mask = lengths > 1e-12
            if not np.any(mask):
                continue

            b_coords = coords[mask]
            b_normals = normals[mask]

            zs = b_coords[:,0].astype(float)
            ys = b_coords[:,1].astype(float)
            xs = b_coords[:,2].astype(float)

            nzs = b_normals[:,0].astype(float)
            nys = b_normals[:,1].astype(float)
            nxs = b_normals[:,2].astype(float)

            # Compute interior and exterior points one grid spacing away along normal
            # (may be non-integer, so use interpolation of some kind)
            zin = zs - nzs*h
            yin = ys - nys*h
            xin = xs - nxs*h

            zout = zs + nzs*h
            yout = ys + nys*h
            xout = xs + nxs*h

            # Assure sampling points are within grid bounds
            zin = np.clip(zin, 0, nz-1)
            yin = np.clip(yin, 0, ny-1)
            xin = np.clip(xin, 0, nx-1)

            zout = np.clip(zout, 0, nz-1)
            yout = np.clip(yout, 0, ny-1)
            xout = np.clip(xout, 0, nx-1)

            Vin = interp3d(f, zin, yin, xin)
            Vout = interp3d(f, zout, yout, xout)

            f[b_coords[:,0], b_coords[:,1], b_coords[:,2]] = 0.5*(Vin + Vout) + (h*sigma)/(2.0*e0)

    return f

def graph_slices(p3, potential_surfaces=None, charged_surfaces=None, slices=None, levels=None, cmap='viridis'):
    """
    Plot multiple z-slices of 3D potential as 2D contours.
    p3: (nz,ny,nx)
    slices: list of z indices (if None, chooses 3 slices: quarter, middle, 3/4)
    """
    nz, ny, nx = p3.shape
    if slices is None:
        slices = [nz//4, nz//2, 3*nz//4]
    n = len(slices)
    cols = min(3, n)
    rows = (n + cols - 1)//cols
    levels = levels if levels is not None else np.linspace(np.min(p3), np.max(p3), 21)

    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)
    for i, iz in enumerate(slices):
        ax = axs[i//cols, i%cols]
        cs = ax.contourf(p3[iz], levels=levels, cmap=cmap)
        ax.set_title(f'z = {iz}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cs, ax=ax)

        # overlay potential_surfaces and charged_surfaces intersections
        if potential_surfaces:
            for (coords, _), V in potential_surfaces:
                if coords.size == 0:
                    continue

                mask = coords[:,0] == iz

                if not np.any(mask):
                    continue

                ys = coords[mask, 1]
                xs = coords[mask, 2]

                ax.scatter(xs, ys, c='yellow', s=6, edgecolors='k', linewidths=0.2, alpha=0.8)

        if charged_surfaces:
            for (coords, normals), sigma in charged_surfaces:
                if coords.size == 0:
                    continue

                mask = coords[:,0] == iz

                if not np.any(mask):
                    continue

                ys = coords[mask, 1]
                xs = coords[mask, 2]

                ax.scatter(xs, ys, c='red', s=8, alpha=0.9)

    # hide any empty subplots
    for j in range(n, rows*cols):
        axs[j//cols, j%cols].axis('off')

    plt.tight_layout()
    plt.show()

def vector_field_plot(vector_field_x, vector_field_y, title, scale_label, subsample_step=5):
    magnitude = np.sqrt(vector_field_x**2 + vector_field_y**2)

    subsample_offset = subsample_step // 2
    subsampled_magnitude = magnitude[subsample_offset::subsample_step, subsample_offset::subsample_step]

    subsampled_magnitude[subsampled_magnitude == 0] = 1

    subsampled_x = vector_field_x[subsample_offset::subsample_step, subsample_offset::subsample_step] / subsampled_magnitude
    subsampled_y = vector_field_y[subsample_offset::subsample_step, subsample_offset::subsample_step] / subsampled_magnitude

    X,Y = np.meshgrid(np.arange(subsampled_x.shape[1]),
                    np.arange(subsampled_x.shape[0]))

    plt.quiver(X, Y, subsampled_x, subsampled_y, subsampled_magnitude, cmap='inferno')
    plt.colorbar(label=scale_label)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    # small test grid
    nz, ny, nx = 48, 96, 96
    p3 = np.zeros((nz, ny, nx))
    e0 = 1.0

    # center solid cube at fixed potential
    cz, cy, cx = nz//2, ny//2, nx//2
    half = 8
    z0, z1 = cz-half, cz+half
    y0, y1 = cy-half, cy+half
    x0, x1 = cx-half, cx+half
    Z, Y, X = np.mgrid[z0:z1, y0:y1, x0:x1]
    filled_coords = np.column_stack((Z.ravel(), Y.ravel(), X.ravel()))
    filled_normals = np.zeros((filled_coords.shape[0], 3), float)
    potential_surfaces = [((filled_coords, filled_normals), 10.0)]

    # outer charged box boundary
    outer = make_box3d(p3.shape, (6, nz-6, 6, ny-6, 6, nx-6))
    charged_surfaces = [(outer, 1.0)]

    # relax
    relaxed = relax_3d(p3, potential_surfaces, charged_surfaces, iters=800, e0=e0, h=1.0)

    # compute Laplacian everywhere
    lap = compute_laplacian_3d(relaxed, h=1.0)

    # graph potential slices (contours) with surfaces overlaid
    levels_p = np.linspace(np.min(relaxed), np.max(relaxed), 21)
    graph_slices(relaxed, potential_surfaces, charged_surfaces,
                 slices=[nz//4, nz//2, 3*nz//4], levels=levels_p, cmap='viridis')

    # graph Laplacian slices (seismic) symmetrically about zero
    vmax = np.max(np.abs(lap))
    levels_l = np.linspace(-vmax, vmax, 31) if vmax > 0 else None
    graph_slices(lap, potential_surfaces=None, charged_surfaces=None,
                 slices=[nz//4, nz//2, 3*nz//4], levels=levels_l, cmap='seismic')
    
    # find efield and charge density
    # For array shape (nz, ny, nx), np.gradient(relaxed) returns (dV/dz, dV/dy, dV/dx)
    dVz, dVy, dVx = np.gradient(relaxed, 1.0, 1.0, 1.0)

    # Electric field E = -grad(V)
    Ez = -dVz
    Ey = -dVy
    Ex = -dVx

    # choose a z slice to visualize (middle)
    zmid = relaxed.shape[0] // 2
    ex_slice = Ex[zmid]
    ey_slice = Ey[zmid]

    print('Ex/Ey slice shapes:', ex_slice.shape, ey_slice.shape)
    vector_field_plot(ex_slice, ey_slice,
                      title='Electric Field Vectors (x-y plane), z = H/2',
                      scale_label='Electric Field Magnitude (V/m)',
                      subsample_step=4)

    xmid = relaxed.shape[0] // 2
    ex_slice = Ey[zmid]
    ey_slice = Ez[zmid]

    print('Ex/Ey slice shapes:', ex_slice.shape, ey_slice.shape)
    vector_field_plot(ex_slice, ey_slice,
                      title='Electric Field Vectors (y-z plane), x = H/2',
                      scale_label='Electric Field Magnitude (V/m)',
                      subsample_step=4)