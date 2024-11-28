from importlib.metadata import metadata

import laspy
import numpy as np
import h5py
import os
import random
import matplotlib.pyplot as plt

from scipy.constants import point
from sympy import print_tree

#TODO
"""
1. Input folder, output folder, if traning data, voxel size, voxel overlap
2. Loading from folder
3. Condition for training data
4. FPS
5. Add to voxelization overlap (maybe)
"""

class PointCloudLoader:
    def __init__(self, project_folder, las_path):
        self.project_folder = project_folder
        self.las_path = las_path
        self.output_path = os.path.join(project_folder, os.path.splitext(os.path.basename(las_path))[0] + ".h5")
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
        # Determine the bounds of the point cloud
        x_min, y_min, z_min = np.min(self.points[:, :3], axis=0)
        x_max, y_max, z_max = np.max(self.points[:, :3], axis=0)
        voxel_size_z = abs(z_min - z_max)

        voxel_origin_size_x = voxel_size_x  # * (1 - voxel_overlap_ratio)
        voxel_origin_size_y = voxel_size_y  # * (1 - voxel_overlap_ratio)
        voxel_origin_size_z = voxel_size_z

        # Calculate voxel indices for each point
        x_indices = np.floor((self.points[:, 0] - x_min) / voxel_origin_size_x).astype(int)
        y_indices = np.floor((self.points[:, 1] - y_min) / voxel_origin_size_y).astype(int)
        z_indices = np.floor((self.points[:, 2] - z_min) / voxel_origin_size_z).astype(int)

        # Stack voxel indices together
        voxel_keys = np.c_[x_indices, y_indices, z_indices]

        # Use np.unique to get unique voxel keys and their inverse indices
        unique_keys, inverse_indices = np.unique(voxel_keys, axis=0, return_inverse=True)

        # Sort points according to the voxels they belong to using inverse indices
        sorted_indices = np.argsort(inverse_indices)
        sorted_points = self.points[sorted_indices]
        sorted_inverse_indices = inverse_indices[sorted_indices]

        # Find the boundaries of the groups (each group corresponds to one voxel)
        boundaries = np.diff(sorted_inverse_indices, prepend=-1, append=len(inverse_indices))
        group_starts = np.where(boundaries != 0)[0]

        # Create an empty list for each voxel
        voxel_arrays = np.split(sorted_points, group_starts[1:])

        # Create a voxel dictionary with keys being unique voxel indices
        self.voxel_dict = {tuple(key): voxel_arrays[idx] for idx, key in enumerate(unique_keys)}

        print("Voxelization complete. Number of voxels:", len(self.voxel_dict))


    # def voxelize(self, voxel_size_x, voxel_size_y, voxel_overlap_ratio):
    #     # Determine the bounds of the point cloud
    #     x_min, y_min, z_min = np.min(self.points[:, :3], axis=0)
    #     x_max, y_max, z_max = np.max(self.points[:, :3], axis=0)
    #     voxel_size_z = abs(z_min - z_max)
    #
    #     voxel_origin_size_x = voxel_size_x #* (1 - voxel_overlap_ratio)
    #     voxel_origin_size_y = voxel_size_y #* (1 - voxel_overlap_ratio)
    #     voxel_origin_size_z = voxel_size_z
    #
    #     x_indices = np.floor((self.points[:, 0] - x_min) / voxel_origin_size_x).astype(int)
    #     y_indices = np.floor((self.points[:, 1] - y_min) / voxel_origin_size_y).astype(int)
    #     z_indices = np.floor((self.points[:, 2] - z_min) / voxel_origin_size_z).astype(int)
    #
    #     voxel_keys = list(zip(x_indices, y_indices, z_indices))
    #
    #     self.voxel_dict = {}
    #     for idx, key in enumerate(voxel_keys):
    #         if key not in self.voxel_dict:
    #             self.voxel_dict[key] = []
    #             print("1")
    #         self.voxel_dict[key].append(self.points[idx])
    #
    #     for key in self.voxel_dict:
    #         self.voxel_dict[key] = np.array(self.voxel_dict[key])
    #         print("2")
    #
    #     print("Voxelization complete. Number of voxels:", len(self.voxel_dict))

    def plot_voxel_grid(self):
        # Create a 2D grid representing the number of points in each voxel
        voxel_count = {}
        for (x_idx, y_idx, _), points in self.voxel_dict.items():
            if (x_idx, y_idx) not in voxel_count:
                voxel_count[(x_idx, y_idx)] = 0
            voxel_count[(x_idx, y_idx)] += len(points)

        x_indices = [key[0] for key in voxel_count.keys()]
        y_indices = [key[1] for key in voxel_count.keys()]
        counts = [voxel_count[key] for key in voxel_count.keys()]

        plt.figure(figsize=(10, 8))
        plt.scatter(x_indices, y_indices, c=counts, cmap='viridis', s=100, edgecolor='black')
        plt.colorbar(label='Number of Points per Voxel')
        plt.xlabel('Voxel X Index')
        plt.ylabel('Voxel Y Index')
        plt.title('2D Grid of Voxel Point Counts')
        plt.grid(True)
        plt.show()

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
        # Convert the points to LAS format for PDAL input
        point_data = remaining_points.astype(np.float32)
        #
        # # Use an in-memory point cloud for PDAL
        # json_pipeline = f"""
        # {{
        #   "pipeline": [
        #     {{
        #       "type": "filters.fps",
        #       "count": {block_size}
        #     }}
        #   ]
        # }}
        # """
        #
        # pipeline = pdal.Pipeline(json_pipeline, arrays=[point_data])
        # pipeline.execute()
        #
        # sampled_array = pipeline.arrays[0]  # Resulting downsampled points
        # selected_indices = np.isin(remaining_points, sampled_array).all(axis=1).nonzero()[0]
        #
        # return selected_indices

    def save_to_hdf5(self):
        with h5py.File(self.output_path, 'w') as h5f:
            grp = h5f.create_group('voxels')
            for voxel_key, voxel_points in self.voxel_dict.items():
                voxel_name = f"voxel_{voxel_key[0]}_{voxel_key[1]}_{voxel_key[2]}"
                voxel_grp = grp.create_group(voxel_name)
                voxel_grp.create_dataset('points', data=voxel_points)
                voxel_grp.attrs['voxel_key'] = voxel_key

            blocks_grp = h5f.create_group('blocks')
            for i, block in enumerate(self.blocks):
                block_name = f"block_{i}"
                block_grp = blocks_grp.create_group(block_name)
                block_grp.create_dataset('points', data=block)
                block_grp.attrs['voxel_key'] = self.block_metadata[i]['voxel_key']
                block_grp.attrs['block_index'] = self.block_metadata[i]['block_index']

            # Save additional LAS values
            for key, value in self.las_values.items():
                h5f.attrs[key] = value

            # Save number of points and classes
            h5f.attrs['num_points'] = self.num_points
            if self.las_values['classification']:
                h5f.create_dataset('classes', data=self.classes)

        print("Data has been saved to HDF5 format at", self.output_path)


# Example usage
if __name__ == "__main__":
    project_folder = (r"C:\Users\lukas\Desktop")
    las_path = (r"C:\Users\lukas\Desktop\pointcloud_velky.las")
    pcl_loader = PointCloudLoader(project_folder, las_path)
    pcl_loader.load_las_file()
    pcl_loader.metadata_extraction()
    pcl_loader.voxelize(2,2,0.1)
    pcl_loader.divide_to_blocks()
    pcl_loader.divide_to_blocks()
    pcl_loader.save_to_hdf5()



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