# Geographic Clustering and Vehicle Routing Problem (VRP) with Genetic Algorithm

This repository contains a Python script that demonstrates how to use Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to cluster geographical data and solve the Vehicle Routing Problem (VRP) for each cluster using a Genetic Algorithm (GA). It uses the Google OR-Tools library to solve the VRP.

## Overview

The code performs the following tasks:

1. **Data Generation**: It generates random geographical coordinates within specified latitude and longitude ranges, which can be customized based on your requirements.

2. **Clustering with DBSCAN**: Utilizes the DBSCAN algorithm to cluster the generated data points based on density.

3. **Outlier Detection**: Identifies outlier nodes by finding data points assigned to cluster -1 by DBSCAN.

4. **Cluster Centroids**: Calculates cluster centroids and filters out small clusters based on a specified minimum cluster size.

5. **Vehicle Routing Problem (VRP) for Clusters**: For each significant cluster, it solves the VRP problem using a Genetic Algorithm. The GA algorithm aims to minimize the total travel distance for each cluster.

## Getting Started

1. Install the required dependencies by running the following command:

   ```bash
   pip install numpy scikit-learn ortools
2. Run the Python script geographic_clustering_vrp.py:

   ```bash
   python geographic_clustering_vrp.py

## Customization
- You can adjust the latitude and longitude ranges in the code to match your geographical area of interest.

- Modify parameters such as epsilon and min_samples for DBSCAN and GA parameters to fine-tune the clustering and optimization process.

## Dependencies
- NumPy
- Scikit-learn
- OR-Tools

## Acknowledgments
- This code demonstrates a basic approach to address clustering and VRP problems, and it can be further customized and extended for specific use cases.
- It uses the Google OR-Tools library to solve the VRP, and you can explore the library's documentation for more advanced features and optimization.
Feel free to use, modify, and extend this code as needed for your own projects. If you have any questions or encounter issues, please create an issue in this repository.

Happy coding!