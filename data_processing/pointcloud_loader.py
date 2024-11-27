from importlib.metadata import metadata

import laspy
import numpy as np
import h5py
import os
import random

from scipy.constants import point

#TODO
"""
1. Input folder, output folder, if traning data, voxel size, voxel overlap
2. Loading from folder
3. Condition for training data
4. Add RANDOM and FPS
5. Use more effective voxelization
"""

class PointCloudLoader:
    def __init__(self, project_folder, las_path):
        self.project_folder = project_folder
        self.las_path = las_path
        self.points = None
        self.las_values = {
            'intensity': False,
            'rgb': False,
            'classification': False
        }
        self.num_points = 0
        self.classes = []
        self.voxel_dict = {}
        self.blocks = []
        self.block_metadata = []

    def load_las_file(self):
        with laspy.open(self.las_path) as las_file:
            las = las_file.read()
            self.points = np.vstack((las.x, las.y, las.z)).T

            if hasattr(las, 'intensity'):
                intensity = las.intensity[:, np.newaxis]
                self.las_values['intensity'] = True
                self.points = np.hstack((self.points, intensity))

            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                rgb = np.vstack((las.red, las.green, las.blue)).T / 65535.0
                self.las_values['rgb'] = True
                self.points = np.hstack((self.points, rgb))

            if hasattr(las, 'classification_trees'):
                classification = las.classification_trees[:, np.newaxis]
                self.las_values['classification'] = True
                self.points = np.hstack((self.points, classification))
                self.classes = np.unique(classification).tolist()
                print(self.classes)

        print(f"Points loaded {self.points.shape[0]}. Intensity {self.las_values['intensity']}, RGB {self.las_values['rgb']}, Classification {self.las_values['classification']}")


    def metadata_extraction(self):
        project_name = os.path.basename(os.path.normpath(project_folder))
        metadata_file_path = os.path.join(project_folder, f"{project_name}_metadata.txt")
        file_name = os.path.basename(os.path.normpath(self.las_path))
        header = "Filename,Number of points, Classes\n"

        metadata = [
                    file_name,
                    self.num_points,
                    self.classes
                    ]

        metadata = [str(item) for item in metadata]
        metadata_line = ",".join(metadata) + "\n"

        os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)

        if not os.path.exists(metadata_file_path):
            with open(metadata_file_path, "w") as metadata_file:
                metadata_file.write(header)
            print(f"Metadata file '{metadata_file_path}' created with header.")
        else:
            print(f"Metadata file '{metadata_file_path}' already exists. No action taken.")

        with open(metadata_file_path, "a") as metadata_file:
            metadata_file.write(metadata_line)

        print(f"Metadata written to '{metadata_file_path}'.")


    def voxelize(self, voxel_size_x, voxel_size_y, voxel_overlap_ratio):
        x_min, y_min, z_min = np.min(self.points[:, :3], axis=0)
        x_max, y_max, z_max = np.max(self.points[:, :3], axis=0)
        voxel_size_z = abs(z_min - z_max)

        voxel_origin_size_x = voxel_size_x * (1 - voxel_overlap_ratio)
        voxel_origin_size_y = voxel_size_y * (1 - voxel_overlap_ratio)
        voxel_origin_size_z = voxel_size_z

        x_indices = np.arange(x_min, x_max, voxel_origin_size_x)
        y_indices = np.arange(y_min, y_max, voxel_origin_size_y)
        z_indices = np.arange(z_min, z_max, voxel_origin_size_z)

        for x in x_indices:
            for y in y_indices:
                for z in z_indices:
                    mask = (
                        (self.points[:, 0] >= x) & (self.points[:, 0] < x + voxel_size_x) &
                        (self.points[:, 1] >= y) & (self.points[:, 1] < y + voxel_size_y) &
                        (self.points[:, 2] >= z) & (self.points[:, 2] < z + voxel_size_z)
                    )

                    voxel_points = self.points[mask]
                    if len(voxel_points) > 0:
                        self.voxel_dict[(x, y, z)] = voxel_points

        print("Voxelization complete. Number of voxels:", len(self.voxel_dict))

    # def voxelize(self, voxel_size_x, voxel_size_y, voxel_overlap_ratio):
    #     x_min, y_min, z_min = np.min(self.points[:, :3], axis=0)
    #     x_max, y_max, z_max = np.max(self.points[:, :3], axis=0)
    #     voxel_size_z = abs(z_min - z_max)
    #
    #     voxel_origin_size_x = voxel_size_x * (1 - voxel_overlap_ratio)
    #     voxel_origin_size_y = voxel_size_y * (1 - voxel_overlap_ratio)
    #
    #     # Calculate voxel grid boundaries
    #     x_edges = np.arange(x_min, x_max + voxel_size_x, voxel_origin_size_x)
    #     y_edges = np.arange(y_min, y_max + voxel_size_y, voxel_origin_size_y)
    #     z_edges = np.array([z_min, z_max])  # Single voxel in the z-direction (if fixed size)
    #
    #     # Assign points to voxel grid
    #     x_indices = np.floor((self.points[:, 0] - x_min) / voxel_origin_size_x).astype(int)
    #     y_indices = np.floor((self.points[:, 1] - y_min) / voxel_origin_size_y).astype(int)
    #     z_indices = np.zeros(len(self.points), dtype=int)  # All points in one layer for z
    #     voxel_indices = np.stack((x_indices, y_indices, z_indices), axis=1)
    #
    #     # Organize points by voxel
    #     self.voxel_dict = {}
    #     for idx, point in zip(map(tuple, voxel_indices), self.points):
    #         if idx not in self.voxel_dict:
    #             self.voxel_dict[idx] = []
    #         self.voxel_dict[idx].append(point)
    #
    #     # Convert lists to arrays for better performance
    #     self.voxel_dict = {k: np.array(v) for k, v in self.voxel_dict.items()}
    #     print("Voxelization complete. Number of voxels:", len(self.voxel_dict))


    def divide_to_blocks(self, block_size=1024, method = "RANDOM"):
        for voxel_key, voxel_points in self.voxel_dict.items():
            if len(voxel_points) < block_size:
                continue
            remaining_points = voxel_points.copy()
            voxel_block_count = 0

            while len(remaining_points) >= block_size:
                if method == "RANDOM":
                    selected_indices = self.random_sampling(remaining_points, block_size)
                elif method == "FPS":
                    selected_points = self.fps_sampling(remaining_points, block_size)
                else:
                    raise ValueError("Unknown sampling method specified")

                selected_points = remaining_points[selected_indices]
                self.blocks.append(selected_points)
                self.block_metadata.append({'voxel_key': voxel_key, 'block_index': voxel_block_count})
                voxel_block_count += 1

                mask = np.ones(len(remaining_points), dtype=bool)
                mask[selected_indices] = False
                remaining_points = remaining_points[mask]
            print(f"Voxel {voxel_key}: Number of blocks created: {voxel_block_count}")

        print(f"Total number of blocks created: {len(self.blocks)}")

    def random_sampling(self, remaining_points, block_size):
        indices = np.random.choice(len(remaining_points), size=block_size, replace=False)
        return indices

    def fps_sampling(self, remaining_points, block_size):
        print("pdal")
# Example usage
if __name__ == "__main__":
    project_folder = (r"C:\Users\lukas\Desktop")
    las_path = (r"C:\Users\lukas\Desktop\pointcloud_big.las")
    pcl_loader = PointCloudLoader(project_folder, las_path)
    pcl_loader.load_las_file()
    pcl_loader.metadata_extraction()
    pcl_loader.voxelize(2,2,0.1)
    pcl_loader.divide_to_blocks()
    pcl_loader.divide_to_blocks()


# # Load the LAS file efficiently
# with laspy.open(r"C:\Users\lukas\Desktop\pointcloud_velky.las") as las_file:
#     las = las_file.read()
#     points = np.vstack((las.x, las.y, las.z)).T
#
#     # Adding additional attributes if they exist
#     if hasattr(las, 'intensity'):
#         intensity = las.intensity[:, np.newaxis]
#         points = np.hstack((points, intensity))
#
#     if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
#         rgb = np.vstack((las.red, las.green, las.blue)).T / 65535.0  # Normalize to [0, 1]
#         points = np.hstack((points, rgb))
#         print(rgb)
#
#     if hasattr(las, 'classification_trees'):
#         classification = las.classification_trees[:, np.newaxis]
#         points = np.hstack((points, classification))
#
# # Calculate the total number of points
# total_points = len(points)
# print(f"total_points: {total_points}")
#
# # Voxelization with overlap
# voxel_size_x = float(input("Enter voxel size for x-axis: "))
# voxel_size_y = float(input("Enter voxel size for y-axis: "))
# voxel_size_z = np.max(points[:, 2]) - np.min(points[:, 2]) + 1
#
# # Define voxel overlap ratio (e.g., 0.5 for 50% overlap)
# voxel_overlap_ratio = float(input("Enter voxel overlap ratio (0 to 1): "))
#
# # Generate voxel bounds and indices with overlap
# x_min, y_min, z_min = np.min(points[:, :3], axis=0)
# x_max, y_max, z_max = np.max(points[:, :3], axis=0)
#
# # Calculate the adjusted voxel size to include overlap
# adjusted_voxel_size_x = voxel_size_x * (1 - voxel_overlap_ratio)
# adjusted_voxel_size_y = voxel_size_y * (1 - voxel_overlap_ratio)
# adjusted_voxel_size_z = voxel_size_z * (1 - voxel_overlap_ratio)
#
# # Generate overlapping voxel indices
# x_indices = []
# y_indices = []
# z_indices = []
# for i in range(int(np.ceil((x_max - x_min) / adjusted_voxel_size_x))):
#     x_indices.append(x_min + i * adjusted_voxel_size_x)
# for i in range(int(np.ceil((y_max - y_min) / adjusted_voxel_size_y))):
#     y_indices.append(y_min + i * adjusted_voxel_size_y)
# for i in range(int(np.ceil((z_max - z_min) / adjusted_voxel_size_z))):
#     z_indices.append(z_min + i * adjusted_voxel_size_z)
#
# voxel_indices = []
# for x in x_indices:
#     for y in y_indices:
#         for z in z_indices:
#             mask = (points[:, 0] >= x) & (points[:, 0] < x + voxel_size_x) & \
#                    (points[:, 1] >= y) & (points[:, 1] < y + voxel_size_y) & \
#                    (points[:, 2] >= z) & (points[:, 2] < z + voxel_size_z)
#             voxel_points = points[mask]
#             if len(voxel_points) > 0:
#                 voxel_indices.append((x, y, z, len(voxel_points)))
#
# # Store points for each voxel
# voxel_dict = {}
# for x, y, z, count in voxel_indices:
#     key = (x, y, z)
#     mask = (points[:, 0] >= x) & (points[:, 0] < x + voxel_size_x) & \
#            (points[:, 1] >= y) & (points[:, 1] < y + voxel_size_y) & \
#            (points[:, 2] >= z) & (points[:, 2] < z + voxel_size_z)
#     voxel_points = points[mask]
#     voxel_dict[key] = voxel_points
#
# # Divide each voxel into blocks with 1024 points (input to PointNet++) without overlap
# block_size = 1024
# blocks = []
# block_metadata = []
#
# for voxel_key in voxel_dict:
#     voxel_points = voxel_dict[voxel_key]
#
#     print(f"Voxel {voxel_key}: Number of points in voxel: {len(voxel_points)}")
#
#     if len(voxel_points) < block_size:
#         continue
#
#     # Create blocks without overlap
#     start_idx = 0
#     voxel_block_count = 0
#     while start_idx + block_size <= len(voxel_points):
#         block = voxel_points[start_idx:start_idx + block_size]
#         blocks.append(block)
#         block_metadata.append({'voxel_key': voxel_key, 'block_index': voxel_block_count})
#         voxel_block_count += 1
#         start_idx += block_size  # Move forward by block size to create non-overlapping blocks
#
#     print(f"Voxel {voxel_key}: Number of blocks created: {voxel_block_count}")
#
# print(f"Total number of blocks created: {len(blocks)}")
#
# # Save data to HDF5 format for PointNet++
# with h5py.File(r"C:\Users\lukas\Desktop\pointcloud_blocks4.h5", 'w') as h5f:
#     grp = h5f.create_group('voxels')
#     for voxel_key, voxel_points in voxel_dict.items():
#         voxel_name = f"voxel_{voxel_key[0]}_{voxel_key[1]}_{voxel_key[2]}"
#         voxel_grp = grp.create_group(voxel_name)
#         voxel_grp.create_dataset('points', data=voxel_points)
#         voxel_grp.attrs['voxel_key'] = voxel_key
#
#     blocks_grp = h5f.create_group('blocks')
#     for i, block in enumerate(blocks):
#         block_name = f"block_{i}"
#         block_grp = blocks_grp.create_group(block_name)
#         block_grp.create_dataset('points', data=block)
#         block_grp.attrs['voxel_key'] = block_metadata[i]['voxel_key']
#         block_grp.attrs['block_index'] = block_metadata[i]['block_index']
#
# print("Data has been saved to HDF5 format.")