# Solution to activity

# Not runnable -- only additional pieces of code.
# See solution_complete.py for full code.

def make_sphere(shape, center, radius, thickness=0.6):
    """
    Return coords (z,y,x) and outward normals for a spherical shell boundary.
    center: (zc,yc,xc) floats. radius in grid units.
    thickness: tolerance for pixel selection
    """
    nz, ny, nx = shape

    # Unpack the center of the sphere
    zc, yc, xc = float(center[0]), float(center[1]), float(center[2])

    # Create a grid of indices for each axis
    Z = np.arange(nz)[:,None,None]
    Y = np.arange(ny)[None,:,None]
    X = np.arange(nx)[None,None,:]

    # Calculate the distance from the center for each grid point
    dist = np.sqrt((Z - zc)**2 + (Y - yc)**2 + (X - xc)**2)

    # Find points within the specified thickness of the sphere surface
    mask = np.abs(dist - radius) <= thickness

    # If no points are within the sphere, return an empty array. Either the sphere is too small
    # or the center is outside the grid.
    if not np.any(mask):
        return np.empty((0,3), int), np.empty((0,3), float)

    # Find the coordinates of these points
    coords = np.argwhere(mask)  # (z,y,x)

    # Find the displacement vectors from center to these points
    dz = coords[:,0].astype(float) - zc
    dy = coords[:,1].astype(float) - yc
    dx = coords[:,2].astype(float) - xc

    # Compute the lengths of the displacement vectors and make sure we don't consider any zero-length ones
    lengths = np.hypot(np.hypot(dz, dy), dx)
    nz_mask = lengths > 0
    
    # Normalize the displacement vectors to get outward normals
    normals = np.zeros_like(coords, dtype=float)

    normals[nz_mask,0] = dz[nz_mask] / lengths[nz_mask]
    normals[nz_mask,1] = dy[nz_mask] / lengths[nz_mask]
    normals[nz_mask,2] = dx[nz_mask] / lengths[nz_mask]

    # Return integer coordinates and corresponding normals
    return coords.astype(int), normals


# Run the simulation

# outer charged sphere boundary
outer = make_sphere(p3.shape, center=(cz, cy, cx), radius=40, thickness=0.6)
charged_surfaces = [(outer, 1.0)]

# relax
relaxed = relax_3d(p3, potential_surfaces, charged_surfaces, iters=800, e0=e0, h=1.0)


# compute Laplacian everywhere
lap = compute_laplacian_3d(relaxed, h=1.0)

charge_density = -e0 * lap

# graph potential slices with surfaces overlaid
levels_p = np.linspace(np.min(relaxed), np.max(relaxed), 21)
graph_slices(relaxed, potential_surfaces, charged_surfaces,
                slices=[nz//4, nz//2, 3*nz//4], levels=levels_p, cmap='viridis',axis=0)

graph_slices(relaxed, potential_surfaces, charged_surfaces,
                slices=[ny//4, ny//2, 3*ny//4], levels=levels_p, cmap='viridis',axis=1)

# graph Laplacian slices symmetrically about zero
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
                    title='Electric Field Vectors (x-y plane), midplane',
                    scale_label='Electric Field Magnitude (V/m)',
                    subsample_step=4)

# select a mid x-slice (y-z plane) -> axis index 2
xmid = relaxed.shape[2] // 2
ex_slice = Ey[:, :, xmid]
ey_slice = Ez[:, :, xmid]

print('Ey/Ez (y-z) slice shapes:', ex_slice.shape, ey_slice.shape)
vector_field_plot(ex_slice, ey_slice,
                    title='Electric Field Vectors (y-z plane), midplane',
                    scale_label='Electric Field Magnitude (V/m)',
                    subsample_step=4)
