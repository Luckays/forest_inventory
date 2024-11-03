"""
Point Cloud Loader Module

This module provides functionality to load point cloud data from various file formats, such as LAS, PLY, and others, and preprocess it for further analysis.
"""

import os
import laspy
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

class PointCloudLoader:
    def __init__(self, file_path):
        """
        Initializes the PointCloudLoader with the provided file path.

        :param file_path: Path to the point cloud file (e.g., .las, .ply).
        """
        self.file_path = file_path
        self.point_cloud = None

    def load_point_cloud(self):
        """
        Loads the point cloud data from the specified file path.

        :return: Loaded point cloud as a pandas DataFrame.
        """
        file_extension = os.path.splitext(self.file_path)[-1].lower()

        if file_extension == ".las":
            self.point_cloud = self._load_las()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return self.point_cloud

    def _load_las(self):
        """
        Loads point cloud data from a LAS file, including intensity, RGB, and classification attributes.

        :return: Loaded point cloud as a pandas DataFrame.
        """
        with laspy.open(self.file_path) as las_file:
            las = las_file.read()
            points = np.vstack((las.x, las.y, las.z)).transpose()

            # Adding additional attributes if they exist
            if hasattr(las, 'intensity'):
                intensity = las.intensity[:, np.newaxis]
                points = np.hstack((points, intensity))

            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                rgb = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0  # Normalize to [0, 1]
                points = np.hstack((points, rgb))

            if hasattr(las, 'classification_trees'):
                classification = las.classification_trees[:, np.newaxis]
                points = np.hstack((points, classification))

        return points

    def visualize_point_cloud(self):
        """
        Visualizes the loaded point cloud using Open3D.
        """
        if self.point_cloud is None:
            raise ValueError("Point cloud data is not loaded. Call load_point_cloud() first.")
        points = self.point_cloud[:, :3].astype(np.float32)
        print(points)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)  # Use only x, y, z coordinates

        # Set classification as color if available
        if self.point_cloud.shape[1] > 7:  # Assuming columns [x, y, z, intensity, red, green, blue, classification]
            classification = self.point_cloud[:, -1]
            # Normalize classification to [0, 1] range and map to RGB colors
            max_class = int(np.max(classification))
            colors = np.array([plt.cm.jet(i / max_class)[:3] for i in classification])  # Use a colormap like 'jet'
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([point_cloud])

# Example usage
if __name__ == "__main__":
    loader = PointCloudLoader(r"C:\Users\lukas\Desktop\pointcloud.las")
    points = loader.load_point_cloud()
    print(points)

    # Visualize the point cloud
    loader.visualize_point_cloud()