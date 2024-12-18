import laspy
import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
import ctypes
import open3d as o3d
import time
from concurrent.futures import ThreadPoolExecutor


class PointCloudLoader:
    def __init__(self, output_folder, las_path):
        self.output_folder = output_folder
        self.las_path = las_path
        self.output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(las_path))[0] + ".h5")
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
        self.fps_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "Release2/fpssampling.dll"))  # Load the DLL

        # Define the DLL function prototype
        self.fps_lib.farthest_point_sampling.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int)
        ]

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
        parent_folder = os.path.dirname(os.path.normpath(self.output_folder))
        project_name = os.path.basename(parent_folder)
        metadata_file_path = os.path.join(self.output_folder, f"{project_name}_metadata.txt")
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

        stride_x = voxel_size_x * (1 - voxel_overlap_ratio)
        stride_y = voxel_size_y * (1 - voxel_overlap_ratio)
        stride_z = voxel_size_z

        x_indices = np.arange(x_min, x_max, stride_x)
        y_indices = np.arange(y_min, y_max, stride_y)
        z_indices = np.arange(z_min, z_max, stride_z)

        x_grid, y_grid, z_grid = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        voxel_keys = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

        total_voxels = len(voxel_keys)
        print(f"Total voxels to process: {total_voxels}")

        for idx, key in enumerate(voxel_keys):
            x, y, z = key
            mask = (
                    (self.points[:, 0] >= x) & (self.points[:, 0] < x + voxel_size_x) &
                    (self.points[:, 1] >= y) & (self.points[:, 1] < y + voxel_size_y) &
                    (self.points[:, 2] >= z) & (self.points[:, 2] < z + voxel_size_z)
            )

            voxel_points = self.points[mask]
            if len(voxel_points) > 0:
                self.voxel_dict[(x, y, z)] = voxel_points

            if (idx + 1) % max(1, total_voxels // 100) == 0 or idx + 1 == total_voxels:
                percent = (idx + 1) / total_voxels * 100
                print(f"Progress: {percent:.2f}% ({idx + 1}/{total_voxels})")

        print("Voxelization complete. Number of voxels:", len(self.voxel_dict))

    def process_voxel(self, voxel_key, voxel_points, block_size, method):
        if len(voxel_points) < block_size:
            return
        remaining_points = np.array(voxel_points)
        voxel_block_count = 0

        while len(remaining_points) >= block_size:
            if method == "RANDOM":
                selected_indices = self.random_sampling(remaining_points, block_size)
            elif method == "FPS":
                selected_indices = self.fps_sampling_optimized(remaining_points, block_size)
            else:
                raise ValueError("Unknown sampling method specified")

            if selected_indices is None:
                print(f"Error: Unable to select points for voxel {voxel_key}")
                break

            selected_points = remaining_points[selected_indices]

            visualize = False
            if visualize:
                self.view_3d_blocks(selected_points)

            self.blocks.append(selected_points)
            self.block_metadata.append({'voxel_key': voxel_key, 'block_index': voxel_block_count})
            voxel_block_count += 1

            mask = np.ones(len(remaining_points), dtype=bool)
            mask[selected_indices] = False
            remaining_points = remaining_points[mask]
            # print(
            #     f"Voxel {voxel_key}: Remaining points after creating block {voxel_block_count}: {len(remaining_points)}")
        print(f"Time:{round(time.time() - start_time,3)} Voxel {voxel_key}: Number of blocks created: {voxel_block_count}")

    def divide_to_blocks(self, block_size=1024, method="FPS"):
        total_voxels = len(self.voxel_dict)
        print(f"Total voxels to process: {total_voxels}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for voxel_key, voxel_points in self.voxel_dict.items():
                futures.append(executor.submit(self.process_voxel_effective, voxel_key, voxel_points, block_size))

        # for voxel_key, voxel_points in self.voxel_dict.items():
        #   self.process_voxel_effective(voxel_key, voxel_points, block_size)

            for idx, future in enumerate(futures):
                future.result()
                if (idx + 1) % max(1, total_voxels // 100) == 0 or idx + 1 == total_voxels:
                    percent = (idx + 1) / total_voxels * 100
                    print(f"Progress: {percent:.2f}% ({idx + 1}/{total_voxels})")

        print(f"Total number of blocks created: {len(self.blocks)}")

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

    def process_voxel_effective(self, voxel_key, voxel_points, block_size):
        if len(voxel_points) < block_size:
            return

        # Convert voxel points to numpy array
        points = np.array(voxel_points)[:, :3].astype(np.float32)

        # Prepare output indices
        max_blocks = len(points) // block_size
        output_indices = np.zeros(max_blocks * block_size, dtype=np.int32)

        # Call the C++ function to divide the voxel into blocks
        num_blocks = self.fps_lib.divide_voxel_into_blocks(
            points.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(points),
            block_size,
            output_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )

        if num_blocks == 0:
            print(f"Error: Unable to divide voxel {voxel_key} into blocks")
            return

        # Process each block returned by the C++ function
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size
            selected_indices = output_indices[block_start:block_end]

            selected_points = voxel_points[selected_indices]

            visualize = False
            if visualize:
                self.view_3d_blocks(selected_points)

            self.blocks.append(selected_points)
            self.block_metadata.append({'voxel_key': voxel_key, 'block_index': block_idx})

        print(f"Time:{round(time.time() - start_time, 3)} Voxel {voxel_key}: Number of blocks created: {num_blocks}")

    def view_3d_blocks(self, points):
        """
        Visualize the 3D point cloud blocks using Open3D.

        Parameters:
            blocks (list of numpy.ndarray or numpy.ndarray): A list of point clouds or a single point cloud.
        """
        # Extract x, y, z coordinates (assume the first 3 columns are x, y, z)
        xyz = points[:, :3]

        # Create a PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)

        # If RGB values are available, use them for coloring
        if points.shape[1] >= 6:  # Assuming columns 4, 5, 6 are RGB
            rgb = points[:, 3:6]  # Extract RGB
            point_cloud.colors = o3d.utility.Vector3dVector(rgb)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])

    def random_sampling(self, remaining_points, block_size):
        if len(remaining_points) < block_size:
            return None
        indices = np.random.choice(len(remaining_points), size=block_size, replace=False)
        return indices

    def fps_sampling_optimized(self, remaining_points, block_size):
        """
        Perform farthest point sampling using the DLL.
        """
        # Extract only XYZ coordinates
        xyz_points = remaining_points[:, :3]  # Assuming the first three columns are X, Y, Z
        num_points = xyz_points.shape[0]

        # Flatten the XYZ array for C++ compatibility
        points_flat = xyz_points.astype(np.float32).flatten()

        # Allocate output array for indices
        output_indices = np.zeros(block_size, dtype=np.int32)

        # Call the FPS function from the DLL
        self.fps_lib.farthest_point_sampling(
            points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_points,
            block_size,
            output_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )

        return output_indices

    def save_to_h5(self):

        with h5py.File(self.output_path, 'w') as h5f:
            voxel_grp = h5f.create_group('voxels')
            for idx, (voxel_key, voxel_points) in enumerate(self.voxel_dict.items()):
                voxel_key_str = f"voxel_{idx}"
                voxel_group = voxel_grp.create_group(voxel_key_str)
                voxel_group.attrs[
                    'voxel_start_point'] = voxel_key


                blocks_grp = voxel_group.create_group('blocks')
                voxel_blocks = [block for block, meta in zip(self.blocks, self.block_metadata) if
                                meta['voxel_key'] == voxel_key]
                for block_idx, block_points in enumerate(voxel_blocks):
                    block_key_str = f"block_{block_idx}"
                    block_dataset = blocks_grp.create_dataset(block_key_str, data=block_points, compression="gzip")
                    block_dataset.attrs['block_index'] = block_idx
                    block_dataset.attrs['voxel_key'] = str(
                        voxel_key)  #

            h5f.attrs['num_voxels'] = len(self.voxel_dict)
            h5f.attrs['num_blocks'] = len(self.blocks)
            h5f.attrs['has_classification'] = bool(self.classes)
            h5f.attrs['classification'] = self.classes if self.classes else []
            h5f.attrs['has_intensity'] = 'intensity' in self.las_values and self.las_values['intensity']
            h5f.attrs['has_rgb'] = 'rgb' in self.las_values and self.las_values['rgb']

        print(f"Data successfully saved to {self.output_path}")


def process_all_las_files(input_folder, output_folder, voxel_size, voxel_overlap):
    las_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.las')]
    for las_file in las_files:
        las_path = os.path.join(input_folder, las_file)
        pcl_loader = PointCloudLoader(output_folder, las_path)

        parent_folder = os.path.dirname(os.path.normpath(output_folder))
        project_name = os.path.basename(parent_folder)
        metadata_file_path = os.path.join(output_folder, f"{project_name}_metadata.txt")

        if os.path.exists(metadata_file_path):
            with open(metadata_file_path, "r") as metadata_file:
                lines = metadata_file.readlines()
                # Skip the header and check if the file_name already exists in any line
                if any(las_file in line for line in lines[1:]):
                    print(f"File '{las_file}' already exists in the metadata. Skipping processing.")
                    continue  # Skip the rest of the loop iteration to avoid reprocessing

        pcl_loader.load_las_file()
        pcl_loader.metadata_extraction()
        pcl_loader.voxelize(voxel_size, voxel_size, voxel_overlap)
        pcl_loader.divide_to_blocks()
        pcl_loader.save_to_h5()

def start():
    parser = argparse.ArgumentParser(description="Process point cloud data into HDF5 format.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the input folder containing LAS files.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder to save HDF5 files.")
    parser.add_argument('--voxel_size', type=float, required=True, help="Size of the voxel grid.")
    parser.add_argument('--voxel_overlap', type=float, required=True, help="Overlap ratio for voxels (0-1).")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    process_all_las_files(args.input_folder, args.output_folder, args.voxel_size, args.voxel_overlap)

# Example usage
if __name__ == "__main__":
   start_time = time.time()
   start()
   # End the timer
   end_time = time.time() #14.5024 seconds,

   # Calculate the elapsed time
   elapsed_time = end_time - start_time
   print(f"Execution Time: {elapsed_time:.4f} seconds")