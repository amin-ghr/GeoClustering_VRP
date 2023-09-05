import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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

#  Calculate centroids for each cluster
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


import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Solve VRP for each cluster
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

    print(f"Cluster {i+1} Route: {route}")

import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import math

# ... (Previous code for generating coordinates and clustering)

# Define Genetic Algorithm parameters
num_generations = 50
population_size = 100

# Create a fitness function using direct distance calculation
def haversine_distance(coord1, coord2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# Create a dictionary to store the best routes for each cluster
best_routes = {}

# Iterate through each cluster and apply GA
for cluster_id, centroid in enumerate(cluster_centroids):
    # Filter data points belonging to this cluster
    cluster_data = data[clusters == cluster_id]
    
    # Define a function to calculate distances within this cluster
    def evaluate_route(route):
        total_distance = 0.0
        for i in range(len(route) - 1):
            from_location = cluster_data[route[i]]
            to_location = cluster_data[route[i + 1]]
            distance = haversine_distance(from_location, to_location)
            total_distance += distance

        # Add distance from the last to the first location (circular route)
        first_location = cluster_data[route[0]]
        last_location = cluster_data[route[-1]]
        total_distance += haversine_distance(last_location, first_location)

        return total_distance,
    
    # Create the toolbox and individual for this cluster
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(cluster_data)), len(cluster_data))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_route)  # Register the evaluate function

    # Create the population for this cluster
    cluster_population = toolbox.population(n=population_size)
    
    # Create the Genetic Algorithm for this cluster
    cluster_algorithm = algorithms.eaSimple(cluster_population, toolbox, cxpb=0.7, mutpb=0.2)

    # Run the Genetic Algorithm for this cluster
    for gen in range(num_generations):
        cluster_algorithm, _ = algorithms.eaSimple(cluster_algorithm, toolbox, cxpb=0.7, mutpb=0.2, ngen=1)  # Use ngen=1
        best_individual = tools.selBest(cluster_algorithm, k=1)[0]
        best_route = evaluate_route(best_individual)[0]

        # Update fitness values for the entire population
        for ind in cluster_algorithm:
            ind.fitness.values = evaluate_route(ind)

        print(f"Cluster {cluster_id + 1}, Generation {gen + 1}: Best route length = {best_route:.2f} km")

    # Store the best route for this cluster
    best_routes[cluster_id] = (best_individual, best_route)

# Print the best routes for each cluster
for cluster_id, (best_individual, best_route) in best_routes.items():
    print(f"Best route for Cluster {cluster_id + 1}: Length = {best_route:.2f} km")

# ----------------------


# import folium
# import numpy as np

# # Create a map centered around the centroids
# map_center = np.mean(cluster_centroids, axis=0)
# m = folium.Map(location=map_center, zoom_start=6)

# # Function to plot a route on the map
# def plot_route(route, color):
#     coords = [locations[i] for i in route]
#     coords.append(coords[0])  # Close the loop
#     folium.PolyLine(locations=coords, color=color).add_to(m)

# # Run the Genetic Algorithm and visualize routes
# for gen in range(num_generations):
#     algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=1)
#     best_individual = tools.selBest(population, k=1)[0]
#     best_route = evaluate_route(best_individual)[0]

#     print(f"Generation {gen+1}: Best route length = {best_route:.2f} km")

#     # Plot the best route on the map
#     plot_route(best_individual, 'blue')

# # Save the map to an HTML file
# m.save('route_visualization.html')

# print("Best route found:", best_individual)


