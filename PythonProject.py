# -*- coding: utf-8 -*-
"""
ISE 535 Project

@author: Zach Sparano
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Store each day's data in a list of dataframes
Sept = []
path = 'C:\\Users\\styxz\\Desktop\\Python\\ProjectData'
for root, dirs, files in os.walk(path):
    for fileName in files:
        current_data = pd.read_csv(path+"/"+fileName ,  encoding = "UTF-8")
        Sept.append(current_data)

# I am making the assumption that every day we will have the entire battery's capacity 
# available to use.

# Objective function
# Params:
#    - Cbatt    : Battery energy capacity (kWh)
#    - x        : Shave level (kW)
#    - t        : Time (hours)
#    - L        : Load (kW)
def objfunc(Cbatt, x, t, L):
    # Difference between load and shave level
    # Load data is in 15-minute increments of kW. Must divide each increment by 4 to convert to kWh.
    diff = (L - x)*0.25
    
    # Indices where the difference between load and shave level is negative
    neg_pos = diff.index[diff < 0]

    # Since we aren't shaving when the shave level is above the load level, set those values to 0
    diff[neg_pos] = 0
    
    # Calculate the difference between the battery's capacity and energy needed from battery
    # at that shave level
    return abs(Cbatt - diff.sum())

# Creates list of every possible shave level to be tested. These are between the min and max load, stepping by 1.
def generate_shave_levels(L):
    min_pshave = math.ceil(L.min())
    max_pshave = math.floor(L.max())
    return np.arange(min_pshave, max_pshave+1, 1)


# Optimization algorithm
# Parameters:
#    - L          : Load (kW)
#    - Cbatt      : Battery energy capacity (kWh)
#    - limit      : Acceptable difference between energy needed for the building and energy in battery.
#    - t          : Time (hours)
# Return :
#    - best_pshave : Best power shave level (kW)
def optimize(t, L, Cbatt, limit):
    # Maximum number of iterations the optimization algorithm performs.
    itermax = 100 
    step=1

    # Generate a vector covering all possible shave levels
    pshave_levels = generate_shave_levels(L)
    
    # Compute the errors
    errors = [objfunc(Cbatt, x, t, L) for x in pshave_levels]
    
    # Find the minimum error and it's index
    min_error = min(errors)
    index = errors.index(min_error)
    
    # Initialize the shave level
    best_pshave = pshave_levels[index]
    
    optimized = False
    iter = 1
    while optimized == False:
        if iter > itermax:
            print("The optimization algorihtm did not converge.")
            break
        # Optimization will be achieved once the difference between capacity and kWh used 
        # from battery is less than the specified limit
        if min_error < limit:
            print(f"Best shave level = {best_pshave} found after {iter} iterations.")
            optimized = True
        else:
            # If the minimum error is still higher than our specified limit, 
            # we test shave levels around current best shave level, stepping by half as much this time.
            step = step/2
            
            # Update possible shave levels
            pshave_levels = np.arange(pshave_levels[index-2], pshave_levels[index+2], step)
            
            # Compute the errors
            errors = [objfunc(Cbatt, x, t, L) for x in pshave_levels]
            
            # Find the minimum error and it's index
            min_error = min(errors)
            index = errors.index(min_error)
            
            # Store the new best shave level
            best_pshave = pshave_levels[index]
            
        iter += 1
    return best_pshave


####### Plot functions ########

# Create a plot showing the error between capacity and energy used (y-axis) 
# for each possible shave level (x-axis)
def plot_objfunc(Cbatt, t, L):
    pshave_levels = generate_shave_levels(L)
    errors = [objfunc(Cbatt, x, t, L) for x in pshave_levels]
    plt.plot(pshave_levels, errors, label = 'error')
    plt.xlabel("Shave Level (kW)")
    plt.ylabel("Error (kWh)")
    plt.title("Objective Function")
    plt.show()

# Battery capacity (kWh)
Cbatt = 1000

plot_objfunc(Cbatt, Sept[12]['Time'], Sept[12]['kW'])


# Function to calculate the charge % of the battery
def estimate_soc(t, L, x, Cbatt):
    I = (L - x)*0.25

    # Since we aren't shaving when the shave level is above the current load level, set those values to 0
    neg_pos = I.index[I < 0]
    I[neg_pos] = 0
    
    # Calculate charge %
    soc = []
    for i in range(0,len(I)):
        # This subtracts the proportion of the battery used so far from 100%
        soc.append(100 - (np.sum(I[0:i]) / Cbatt)*100)
    return soc

# Plot the state of charge and the load
def plot_soc_load(t, L, pshave_level, Cbatt):
    # Estimate the SoC
    soc = estimate_soc(t, L, pshave_level, Cbatt)
    
    # Plot the SoC
    plt.plot(t, soc, color='green', label='Battery Charge %')
    
    # Plot the load
    plt.plot(t, L, color='blue', label='Building Load (kW)')
    
    # Plot the shave level
    pshave_level_list=[]
    for i in range(0,len(L)):
        pshave_level_list.append(pshave_level)
    plt.plot(t, pshave_level_list, color='red', label='Shave Level (kW)')
    plt.xlabel("Time (hours)")
    plt.ylabel("Power(kW)")
    plt.title("Peak Shaving")
    plt.legend(loc="upper right")
    plt.show()


limit = 0.5
# Put all of September's optimum shave levels in a vector
best_pshave_sept = []
for i in range(0,len(Sept)):
    print(f'September {i+1}:')
    best_pshave_sept.append(optimize(Sept[i]['Time'], Sept[i]['kW'], Cbatt, limit))

plot_soc_load(Sept[12]['Time'], Sept[12]['kW'], best_pshave_sept[12], Cbatt)

# Create a cumulative density function plot to find optimum shave level for the month
def plotCDF(best_pshave_sept):
    shaveSort = np.sort(best_pshave_sept)
    cdfData = 1. * np.arange(len(best_pshave_sept)) / (len(best_pshave_sept) - 1)
    plt.plot(shaveSort, cdfData)
    plt.xlabel("Shave Level (kW)")
    plt.ylabel("Percentile")
    plt.title("CDF")
    
    # Add data labels to graph
    for x,y in zip(shaveSort,cdfData):
        label = "{:n}".format(x)
        plt.annotate(label, # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(-10,10), # distance from text to points (x,y)
                     ha='center',
                     size=7.5)
    plt.show()

plotCDF(best_pshave_sept)

######### Results #########

def allShavePlots(day,Cbatt):
    # Plot the objective function
    plot_objfunc(Cbatt, Sept[day-1]['Time'], Sept[day-1]['kW'])
    
    # Plot the state of charge, building load, and shavel level on the same graph for a single day
    plot_soc_load(Sept[day-1]['Time'], Sept[day-1]['kW'], best_pshave_sept[day-1], Cbatt)
    
    plotCDF(best_pshave_sept)

allShavePlots(30,1000)