# %%
import laspy
import numpy as np
import h5py

# Load the LAS file efficiently
with laspy.open(r"C:\Users\lukas\Desktop\pointcloud_velky.las") as las_file:
    las = las_file.read()
    points = np.vstack((las.x, las.y, las.z)).T

    # Adding additional attributes if they exist
    if hasattr(las, 'intensity'):
        intensity = las.intensity[:, np.newaxis]
        points = np.hstack((points, intensity))

    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        rgb = np.vstack((las.red, las.green, las.blue)).T / 65535.0  # Normalize to [0, 1]
        points = np.hstack((points, rgb))
        print(rgb)

    if hasattr(las, 'classification_trees'):
        classification = las.classification_trees[:, np.newaxis]
        points = np.hstack((points, classification))

# Calculate the total number of points
total_points = len(points)
print(f"total_points: {total_points}")

# Voxelization with overlap
voxel_size_x = float(input("Enter voxel size for x-axis: "))
voxel_size_y = float(input("Enter voxel size for y-axis: "))
voxel_size_z = np.max(points[:, 2]) - np.min(points[:, 2]) + 1

# Define voxel overlap ratio (e.g., 0.5 for 50% overlap)
voxel_overlap_ratio = float(input("Enter voxel overlap ratio (0 to 1): "))

# Generate voxel bounds and indices with overlap
x_min, y_min, z_min = np.min(points[:, :3], axis=0)
x_max, y_max, z_max = np.max(points[:, :3], axis=0)

# Calculate the adjusted voxel size to include overlap
adjusted_voxel_size_x = voxel_size_x * (1 - voxel_overlap_ratio)
adjusted_voxel_size_y = voxel_size_y * (1 - voxel_overlap_ratio)
adjusted_voxel_size_z = voxel_size_z * (1 - voxel_overlap_ratio)

# Generate overlapping voxel indices
x_indices = []
y_indices = []
z_indices = []
for i in range(int(np.ceil((x_max - x_min) / adjusted_voxel_size_x))):
    x_indices.append(x_min + i * adjusted_voxel_size_x)
for i in range(int(np.ceil((y_max - y_min) / adjusted_voxel_size_y))):
    y_indices.append(y_min + i * adjusted_voxel_size_y)
for i in range(int(np.ceil((z_max - z_min) / adjusted_voxel_size_z))):
    z_indices.append(z_min + i * adjusted_voxel_size_z)

voxel_indices = []
for x in x_indices:
    for y in y_indices:
        for z in z_indices:
            mask = (points[:, 0] >= x) & (points[:, 0] < x + voxel_size_x) & \
                   (points[:, 1] >= y) & (points[:, 1] < y + voxel_size_y) & \
                   (points[:, 2] >= z) & (points[:, 2] < z + voxel_size_z)
            voxel_points = points[mask]
            if len(voxel_points) > 0:
                voxel_indices.append((x, y, z, len(voxel_points)))

# Store points for each voxel
voxel_dict = {}
for x, y, z, count in voxel_indices:
    key = (x, y, z)
    mask = (points[:, 0] >= x) & (points[:, 0] < x + voxel_size_x) & \
           (points[:, 1] >= y) & (points[:, 1] < y + voxel_size_y) & \
           (points[:, 2] >= z) & (points[:, 2] < z + voxel_size_z)
    voxel_points = points[mask]
    voxel_dict[key] = voxel_points

# Divide each voxel into blocks with 1024 points (input to PointNet++) without overlap
block_size = 1024
blocks = []
block_metadata = []

for voxel_key in voxel_dict:
    voxel_points = voxel_dict[voxel_key]

    print(f"Voxel {voxel_key}: Number of points in voxel: {len(voxel_points)}")

    if len(voxel_points) < block_size:
        continue

    # Create blocks without overlap
    start_idx = 0
    voxel_block_count = 0
    while start_idx + block_size <= len(voxel_points):
        block = voxel_points[start_idx:start_idx + block_size]
        blocks.append(block)
        block_metadata.append({'voxel_key': voxel_key, 'block_index': voxel_block_count})
        voxel_block_count += 1
        start_idx += block_size  # Move forward by block size to create non-overlapping blocks

    print(f"Voxel {voxel_key}: Number of blocks created: {voxel_block_count}")

print(f"Total number of blocks created: {len(blocks)}")

# Save data to HDF5 format for PointNet++
with h5py.File(r"C:\Users\lukas\Desktop\pointcloud_blocks4.h5", 'w') as h5f:
    grp = h5f.create_group('voxels')
    for voxel_key, voxel_points in voxel_dict.items():
        voxel_name = f"voxel_{voxel_key[0]}_{voxel_key[1]}_{voxel_key[2]}"
        voxel_grp = grp.create_group(voxel_name)
        voxel_grp.create_dataset('points', data=voxel_points)
        voxel_grp.attrs['voxel_key'] = voxel_key

    blocks_grp = h5f.create_group('blocks')
    for i, block in enumerate(blocks):
        block_name = f"block_{i}"
        block_grp = blocks_grp.create_group(block_name)
        block_grp.create_dataset('points', data=block)
        block_grp.attrs['voxel_key'] = block_metadata[i]['voxel_key']
        block_grp.attrs['block_index'] = block_metadata[i]['block_index']

print("Data has been saved to HDF5 format.")