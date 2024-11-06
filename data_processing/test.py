import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the LAS file
las = laspy.read(r"C:\Users\lukas\Desktop\pointcloud.las")

# Extract x, y, z coordinates
x = las.x
y = las.y
z = las.z

# Combine coordinates into a single array
points = np.vstack((x, y, z)).T

# Set up voxelization parameters
voxel_size_x = 2  # For illustration
voxel_size_y = 2
voxel_size_z = 20  # Maximum height of the point cloud
overlap_ratio = 0.5  # Overlap ratio

# Adjust voxel size to include overlap
adjusted_voxel_size_x = voxel_size_x * (1 - overlap_ratio)
adjusted_voxel_size_y = voxel_size_y * (1 - overlap_ratio)
adjusted_voxel_size_z = voxel_size_z * (1 - overlap_ratio)

# Determine voxel indices with overlap
x_indices = np.floor((x - np.min(x)) / adjusted_voxel_size_x).astype(int)
y_indices = np.floor((y - np.min(y)) / adjusted_voxel_size_y).astype(int)
z_indices = np.floor((z - np.min(z)) / adjusted_voxel_size_z).astype(int)

# Combine the indices to create unique voxel identifiers
voxel_indices = np.vstack((x_indices, y_indices, z_indices)).T

# Get unique voxels
unique_voxels = np.unique(voxel_indices, axis=0)

# Plot the point cloud with voxel boundaries
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the points
ax.scatter(x, y, z, s=0.1, c='b', alpha=0.5)

# Draw voxel boundaries as cubes
for voxel in unique_voxels:
    voxel_corner = np.array([
        np.min(x) + voxel[0] * adjusted_voxel_size_x,
        np.min(y) + voxel[1] * adjusted_voxel_size_y,
        np.min(z) + voxel[2] * adjusted_voxel_size_z
    ])

    # Define the vertices of the voxel cube
    vertices = [
        [voxel_corner[0], voxel_corner[1], voxel_corner[2]],
        [voxel_corner[0] + adjusted_voxel_size_x, voxel_corner[1], voxel_corner[2]],
        [voxel_corner[0] + adjusted_voxel_size_x, voxel_corner[1] + adjusted_voxel_size_y, voxel_corner[2]],
        [voxel_corner[0], voxel_corner[1] + adjusted_voxel_size_y, voxel_corner[2]],
        [voxel_corner[0], voxel_corner[1], voxel_corner[2] + adjusted_voxel_size_z],
        [voxel_corner[0] + adjusted_voxel_size_x, voxel_corner[1], voxel_corner[2] + adjusted_voxel_size_z],
        [voxel_corner[0] + adjusted_voxel_size_x, voxel_corner[1] + adjusted_voxel_size_y, voxel_corner[2] + adjusted_voxel_size_z],
        [voxel_corner[0], voxel_corner[1] + adjusted_voxel_size_y, voxel_corner[2] + adjusted_voxel_size_z]
    ]

    # Define the 6 faces of the voxel cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]

    # Create a 3D polygon collection for the voxel
    ax.add_collection3d(Poly3DCollection(faces, facecolors='r', linewidths=0.2, edgecolors='k', alpha=0.1))

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.title('Voxely s překryvy v bodovém mračnu')

plt.show()