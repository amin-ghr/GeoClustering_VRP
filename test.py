import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium

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
epsilon = 0.5
min_samples = 5
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(data_scaled)

# Calculate centroids for each cluster
unique_clusters = set(clusters)
cluster_centroids = []

for cluster_id in unique_clusters:
    if cluster_id == -1:
        continue  # Skip noise points
    cluster_indices = np.where(clusters == cluster_id)[0]
    cluster_coordinates = data[cluster_indices]
    cluster_mean = np.mean(cluster_coordinates, axis=0)
    cluster_centroids.append(cluster_mean)

# Print cluster centroids
for i, centroid in enumerate(cluster_centroids):
    print(f"Cluster {i+1} Centroid: {centroid}")

# Initialize a map centered at a location (you can adjust the coordinates)
map_center = [38.0, -97.0]  # Example center coordinates (adjust as needed)
m = folium.Map(location=map_center, zoom_start=5)

# Define colors for different clusters (you can define more colors if needed)
cluster_colors = ['red', 'blue', 'green', 'purple', 'orange']

# Calculate VRP routes and store them in a list
vrp_routes = []

for i, centroid in enumerate(cluster_centroids):
    depot = [centroid]  # Depot is the cluster centroid
    locations = data[np.where(clusters == i)[0]]  # Locations in the cluster

    manager = pywrapcp.RoutingIndexManager(len(locations) + 1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return np.linalg.norm(locations[from_node] - locations[to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    
    vrp_routes.append(route)

# Add cluster centroids and VRP routes to the map in order
for i, (centroid, route) in enumerate(zip(cluster_centroids, vrp_routes)):
    # Add cluster centroid marker with location number
    folium.Marker(
        location=centroid,
        icon=folium.DivIcon(html=f'<div style="font-weight: bold;">{i + 1}</div>'),
        popup=f"Cluster {i + 1} Centroid"
    ).add_to(m)

    # Define a function to create a polyline for a route
    def create_polyline(route, color, locations):
        coordinates = []
        for node in route:
            if node < len(locations):
                coordinates.append(locations[node])
            else:
                print(f"Warning: Node {node} is out of bounds for cluster {i + 1}.")
        return folium.PolyLine(
            locations=coordinates,
            color=color,
            weight=2.5,
            opacity=1.0
        )

    # Add VRP route to the map as a polyline, passing the cluster's locations
    color = cluster_colors[i % len(cluster_colors)]
    polyline = create_polyline(route, color, locations)
    if polyline is not None:
        polyline.add_to(m)

        
# Save the map to an HTML file
m.save("routes_map.html")
