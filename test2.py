import random
import math
import numpy as np
from deap import base, creator, tools, algorithms

# Generate random coordinates for 100 locations in the US (latitude and longitude)
np.random.seed(0)  # For reproducibility
num_locations = 100
data = np.random.uniform(low=(24, -125), high=(49, -66), size=(num_locations, 2))

# Dummy clustering (you can replace this with your DBSCAN clustering)
clusters = np.random.randint(0, 5, size=num_locations)

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

# Create a function to calculate distances using Haversine formula
def haversine_distance(coord1, coord2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Create a dictionary to store the best routes for each cluster
best_routes = {}

# Define Genetic Algorithm parameters
num_generations = 50
population_size = 100

# Create a function to evaluate routes within each cluster
def evaluate_route(route):
    total_distance = 0.0
    route_coords = cluster_data[route]

    for i in range(len(route) - 1):
        from_location = route_coords[i]
        to_location = route_coords[i + 1]
        distance = haversine_distance(from_location, to_location)
        total_distance += distance

    first_location = route_coords[0]
    last_location = route_coords[-1]
    distance = haversine_distance(last_location, first_location)
    total_distance += distance

    return total_distance,

# Iterate through each cluster and apply GA
for cluster_id, centroid in enumerate(cluster_centroids):
    cluster_data = data[clusters == cluster_id]

    # Create DEAP types for individuals and fitness
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create a DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(cluster_data)), len(cluster_data))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_route)

    # Create the population for this cluster
    cluster_population = toolbox.population(n=population_size)

    # Create the Genetic Algorithm for this cluster
    cluster_algorithm = algorithms.eaSimple(cluster_population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations)

    # Run the Genetic Algorithm for this cluster
    for gen in range(num_generations):
        # Create a DEAP population object for this generation
        population = toolbox.population(n=population_size)

        # Evaluate fitness for each individual in the new population
        fitness_values = [evaluate_route(ind)[0] for ind in population]

        # Assign fitness values to individuals
        for ind, fitness in zip(population, fitness_values):
            ind.fitness.values = (fitness,)

        # Update the population using the Genetic Algorithm
        population = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=1)  # Use ngen=1
        cluster_algorithm = list(population)  # Update the cluster_algorithm with the new population

        best_individual = min(cluster_algorithm, key=lambda ind: ind.fitness.values[0])
        best_route = best_individual.fitness.values[0]

        print(f"Cluster {cluster_id + 1}, Generation {gen + 1}: Best route length = {best_route:.2f} km")







    # Store the best route for this cluster
    best_routes[cluster_id] = (best_individual, best_route)

# Print the best routes for each cluster
for cluster_id, (best_individual, best_route) in best_routes.items():
    print(f"Best route for Cluster {cluster_id + 1}: Length = {best_route:.2f} km")
