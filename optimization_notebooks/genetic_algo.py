import numpy as np
import pandas as pd
from deap import creator, base, tools
from math import radians, sin, cos, sqrt, atan2
import folium
import time
import heapq
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from typing import List, Tuple
import multiprocessing
import psutil
import warnings
import platform
import os



# Define evaluation function



# Modify the class name to match your student number.
INF = 10_000_000_000.0




class r0915387:
    def __init__(self):
        self.alpha = 0.05
        self.lambdaa = 100
        self.mu = self.lambdaa * 2
        self.k = 3
        self.origFun = fitness_function		# Function handle to the objective function
        self.domfit = lambda x,distanceMatrix=np.ndarray, pop=None, fun=self.origFun: self.dominatedFitnessWrapper(distanceMatrix, x, self.origFun, pop)
        self.objf = lambda x, distanceMatrix=np.ndarray, pop=None, betaInit=0: self.sharedFitnessWrapper(distanceMatrix, num, myfun, x, pop, betaInit)	# Objective function employed by the evolutionary algorithm
        self.all_avg = []
        self.all_best = []
        self.times = []
        self.best_fitnesses=[]
        self.mean_fitnesses=[]
        self.edge_distances={}
        

    def optimize(data):
        
        warnings.filterwarnings("ignore", category=RuntimeWarning)
       
        # Load the dataset for existing AED locations
        
# Select relevant columns
        relevant_data = data[['latitude_permanence', 'longitude_permanence',
                      'latitude_intervention', 'longitude_intervention',
                      'vector_type', 'waiting_time']].dropna()
        relevant_data = relevant_data[relevant_data['latitude_intervention'] != 50.0]
        relevant_data = relevant_data[relevant_data['latitude_permanence'] != 50.0]
# Define constants for latitude and longitude ranges based on the dataset
        LAT_MIN = data[['latitude_permanence', 'latitude_intervention']].min().min()
        LAT_MAX = data[['latitude_permanence', 'latitude_intervention']].max().max()
        LONG_MIN = data[['longitude_permanence', 'longitude_intervention']].min().min()
        LONG_MAX = data[['longitude_permanence', 'longitude_intervention']].max().max()

# Rest of your script goes here


        
        
        all_avg=[]
        all_best=[]
        population = r0915387.initialize(data, lmda=50)
        best_individual = population[0]
        best_fitness = float("+inf")
        k=0
        for individual in population:
                fit = r0915387.fitness_function( relevant_data, individual)
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = individual
        avgisbest = 0
        numb = 0
        best_prev_fitness = np.inf
        best_fitnesses=[]
        mean_fitnesses=[]
        while  avgisbest<50:
            
          

            offsprings=[]
        
            
            selected = r0915387.selection( population, relevant_data, k=5)
            
            for i in range(50):
              
                parent1, parent2 = random.sample(population, 2)
                
                offspring=r0915387.one_point_crossover(parent1, parent2)
                offsprings.append(offspring[0])
                offsprings.append(offspring[1])
            
            population = [r0915387.local_search_mutation(population[i], LAT_MIN, LAT_MAX, LONG_MIN, LONG_MAX, mutation_range=0.01) for i in range(len(population))]
            #print(population3, 'population after mutation')
            
            joinedPopulation = population+offsprings
            #print('joinedp', joinedPopulation)
            
            #population = r0915387.shared_elimination(distanceMatrix, pop=joinedPopulation, keep=r0915387.lambdaa)
            
            population=r0915387.elimination(joinedPopulation, relevant_data, keep=50)
            
            #print('after elimination', population4)
            fvals=[r0915387.fitness_function(data, population[i]) for i in range(len(population))]
            meanObjective = np.mean(fvals, axis=0)
            bestObjective = np.min(fvals, axis=0)
            bestSolution=population[0]
            if bestObjective <best_fitness :
                best_fitness=bestObjective
                best_individual=population[0]
            #best_fitnesses.append(bestObjective)
            #mean_fitnesses.append(meanObjective)
            #meanObjective, bestObjective, bestSolution = r0915387.getRes(population, distanceMatrix)
           
            all_avg.append(meanObjective)
            all_best.append(bestObjective)
            
            if abs(bestObjective - meanObjective)/bestObjective < 10**-7:
                avgisbest += 1
            else:
                avgisbest = 0
            k += 1
            
            
            if k>50:
                print(best_individual)
                plt.plot(range(1, 52), all_best, marker='o', linestyle='-')
                plt.xlabel('Run Number')
                plt.ylabel('Best Fitness')
                plt.title('Best Fitness over 51 Runs')
                plt.grid(True)
                plt.show()
                r0915387.show_in_map(best_individual, LAT_MIN, LAT_MAX, LONG_MIN, LONG_MAX)
                break
            
            print(f'Number of runs: {k}\r\n\r\n')
        #print('Best ind', best_individual)
            print('Best fitness', best_fitness)
        #print(best_fitnesses, mean_fitnesses)
            mean_best_fitness = np.mean(best_fitnesses)
            std_best_fitnesses = np.std(best_fitnesses)
            mean_mean_fitness = np.mean(mean_fitnesses)
            std_mean_fitnesses = np.mean(best_fitnesses)
        
  

# Add labels and title


# Show the plot


   
    
    def initialize(data, lmda):
        
        dim=2
        LAT_MIN = data[['latitude_permanence', 'latitude_intervention']].min().min()
        LAT_MAX = data[['latitude_permanence', 'latitude_intervention']].max().max()
        LONG_MIN = data[['longitude_permanence', 'longitude_intervention']].min().min()
        LONG_MAX = data[['longitude_permanence', 'longitude_intervention']].max().max()
        coordinates_sets = []
        for _ in range(lmda):
            coordinates = [(random.uniform(LAT_MIN, LAT_MAX), random.uniform(LONG_MIN, LONG_MAX)) for _ in range(5)]
            coordinates_sets.append(coordinates)
        return coordinates_sets

    

    # Function to calculate distance between two points using Haversine formula
    def calculate_distance(lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1_rad, lon1_rad = radians(lat1), radians(lon1)
        lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = 6371 * c  # Radius of the Earth in kilometers
        return distance

# Fitness function
    def fitness_function(relevant_data, individual):
    # Calculate distance from each intervention to each AED location
        interventions = relevant_data[['latitude_intervention', 'longitude_intervention', 'waiting_time']].sort_values(by='waiting_time', ascending=False).head(1000)
        intervention_lat=interventions['latitude_intervention'].values
        intervention_long=interventions['longitude_intervention'].values
        aed_data = pd.read_csv('data/aed_bxl.parquet.csv')
        aed_data.dropna(subset=['latitude', 'longitude'], inplace=True)
        aed_lat=aed_data['latitude'].values
        aed_long=aed_data['longitude'].values
        aed_existing=[]
        for i in range(len(aed_lat)):
            aed_existing.append((aed_lat[i], aed_long[i]))
 
        distances = []
        for i in range(len(interventions)):
            intervention_lati, intervention_longi = intervention_lat[i], intervention_long[i]
            min_distance = float('inf')
        
    
            for aed_location in individual:
                aed_lat, aed_lon = aed_location
                distance = r0915387.calculate_distance(intervention_lati, intervention_longi, aed_lat, aed_lon)
                min_distance = min(min_distance, distance)
            distances.append(min_distance)
        sum_distances=sum(distances)
        penalty=[]
        
        for aed_location in individual:
            p=0
            aed_lat, aed_lon = aed_location
            for k in aed_existing :
                existing_lat, existing_long=k
                dist_to_aed=r0915387.calculate_distance(existing_lat,existing_long, aed_lat, aed_lon)
                if dist_to_aed<2.0 :
                    p+=1
            penalty.append(p)
        penalty_final=sum(penalty)
        sum_distances+=penalty_final*50
    # Return the sum of minimum distances as the fitness value
        return(sum_distances)

    
    
    def selection( population, relevant_data, k):
        selected = []
        for ii in range(50):
            ri = random.choices(range(len(population)), k=5)
            candidates = [population[i] for i in ri]
            #min_idx=np.argmax(r0915387.dominatedFitnessWrapper(distanceMatrix=distance_matrix, X=candidates, fun=r0915387.origFun, pop=candidates))
            min_idx = np.argmin(r0915387.fitness_function(relevant_data, x) for x in candidates )
            selected.append(candidates[min_idx])
        return selected
    

# Crossover
    def one_point_crossover(parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))  # Randomly select crossover point
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

# Mutation
    def swap_mutation(solution):
        mutated_solution = solution.copy()
        idx1, idx2 = np.random.choice(len(solution), 2, replace=False)  # Randomly select two indices to swap
        mutated_solution[idx1], mutated_solution[idx2] = mutated_solution[idx2], mutated_solution[idx1]
        return mutated_solution

    def local_search_mutation(solution, min_lat, max_lat, min_lon, max_lon, mutation_range=0.01):
        mutated_solution = solution.copy()
        idx = np.random.randint(len(solution))  # Randomly select an index to mutate
    # Perturb the latitude and longitude of the selected AED location within a small range
        lat, lon = mutated_solution[idx]
        new_lat = max(min(lat + np.random.uniform(-mutation_range, mutation_range), max_lat), min_lat)
        new_lon = max(min(lon + np.random.uniform(-mutation_range, mutation_range), max_lon), min_lon)
        mutated_solution[idx] = (new_lat, new_lon)
        return mutated_solution
# Termination Condition
    def check_termination_condition(current_generation, max_generations):
        return current_generation >= max_generations

   
 
    
    def elimination( joinedPopulation, relevant_data, keep):
        
                
        fvals = [r0915387.fitness_function(relevant_data, k) for k in joinedPopulation]
        perm = np.argsort(fvals)
        return [joinedPopulation[i] for i in perm[:keep]]
    
    def show_in_map(best_individual, LAT_MIN, LAT_MAX, LONG_MIN, LONG_MAX):
        map_center = [(LAT_MIN + LAT_MAX) / 2, (LONG_MIN + LONG_MAX) / 2]
        mymap = folium.Map(location=map_center, zoom_start=12)
        for i, ind in enumerate(best_individual):
            aed_location = [ind[0], ind[1]]
            folium.Marker(location=aed_location, popup=f"AED Device {i + 1}").add_to(mymap)
        aed_data = pd.read_csv('data/aed_bxl.parquet.csv')
        aed_data.dropna(subset=['latitude', 'longitude'], inplace=True)
        aed_lat=aed_data['latitude'].values
        aed_long=aed_data['longitude'].values
        
        for i in range(len(aed_lat)):
            aed_existing=[aed_lat[i], aed_long[i]]
            folium.Marker(location=aed_existing, popup=f"AED Existing Device {i + 1}", icon=folium.Icon(color='red')).add_to(mymap)

        mymap.save('genetic_aed_locations_map100.html')

   
    
 # Load the dataset for intervention locations
data = pd.read_csv('data/interventions_bxl.parquet.csv')
data['latitude_permanence'] = data['latitude_permanence'].apply(lambda x: float(str(x)[:2] + '.' + str(x)[2:]))
data['longitude_permanence'] = data['longitude_permanence'].apply(lambda x: float(str(x)[:1] + '.' + str(x)[1:]))
data['latitude_intervention'] = data['latitude_intervention'].astype(int).apply(
    lambda x: float(str(x)[:2] + '.' + str(x)[2:]))
data['longitude_intervention'] = data['longitude_intervention'].astype(int).apply(
    lambda x: float(str(x)[:1] + '.' + str(x)[1:]))

# Load the dataset for existing AED locations

# Select relevant columns
relevant_data = data[['latitude_permanence', 'longitude_permanence',
                      'latitude_intervention', 'longitude_intervention',
                      'vector_type', 'waiting_time']].dropna()
relevant_data = relevant_data[relevant_data['latitude_intervention'] != 50.0]
relevant_data = relevant_data[relevant_data['latitude_permanence'] != 50.0]
# Define constants for latitude and longitude ranges based on the dataset
LAT_MIN = data[['latitude_permanence', 'latitude_intervention']].min().min()
LAT_MAX = data[['latitude_permanence', 'latitude_intervention']].max().max()
LONG_MIN = data[['longitude_permanence', 'longitude_intervention']].min().min()
LONG_MAX = data[['longitude_permanence', 'longitude_intervention']].max().max()   

r0915387.optimize(data)


