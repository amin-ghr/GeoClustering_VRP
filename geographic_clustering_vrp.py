import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Generate random coordinates
def generate_random_coordinates(num_locations):
    coordinates = []
    for _ in range(num_locations):
        lat = random.uniform(24.396308, 49.384358)  # US latitude range
        lon = random.uniform(-125.000000, -66.934570)  # US longitude range
        coordinates.append([lat, lon])
    return coordinates

num_locations = 100
random_coordinates = generate_random_coordinates(num_locations)

# Convert coordinates to numpy array
data = np.array(random_coordinates)

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply DBSCAN
epsilon = 0.4
min_samples = 5
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(data_scaled)

# Identify outlier nodes
outlier_indices = np.where(clusters == -1)[0]
print("Outlier Node Indices:", outlier_indices)

# Calculate centroids for each cluster and filter out small clusters
unique_clusters = set(clusters)
cluster_centroids = []
min_cluster_size = 5  # Minimum cluster size to consider
valid_clusters = []

for cluster_id in unique_clusters:
    if cluster_id == -1:
        continue  # Skip noise points
    cluster_indices = np.where(clusters == cluster_id)[0]
    if len(cluster_indices) >= min_cluster_size:
        valid_clusters.append(cluster_indices)
        cluster_coordinates = data[cluster_indices]
        cluster_mean = np.mean(cluster_coordinates, axis=0)
        cluster_centroids.append(cluster_mean)

# Calculate VRP routes and distances
vrp_routes = []
distances = []

for i, cluster_indices in enumerate(valid_clusters):
    locations = data[cluster_indices]

    while len(locations) > 1:
        depot = [locations[0]]  # Depot is the first location
        manager = pywrapcp.RoutingIndexManager(len(locations) + 1, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)

            if 0 <= from_node < len(locations) and 0 <= to_node < len(locations):
                return np.linalg.norm(locations[from_node] - locations[to_node])

            return 0  # Return 0 for out-of-bounds indices

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(search_parameters)
        route = []
        index = routing.Start(0)
        total_distance = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            if len(route) > 1:
                total_distance += distance_callback(route[-2], route[-1])
            index = solution.Value(routing.NextVar(index))

        vrp_routes.append(route)
        distances.append(total_distance)

        # Remove the visited nodes from the locations
        unique_indices = np.unique(route)
        # Create a list of unique cluster indices excluding -1
        unique_indices = [i for i in range(len(clusters)) if clusters[i] != -1]

        # Filter out the locations based on unique_indices
        locations = data[unique_indices]


# Print VRP routes and distances
for i, (route, distance) in enumerate(zip(vrp_routes, distances)):
    cluster_indices = np.where(clusters == i)[0]
    indexes = [cluster_indices[manager.IndexToNode(node)] for node in route if 0 <= manager.IndexToNode(node) < len(cluster_indices)]
    print(f"Cluster {i+1} Index Route: {indexes}")
    print(f"Cluster {i+1} Total Distance: {distance}")
