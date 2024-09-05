#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Process disturbance
mP = np.array([0, 0]) # Mean
cP = np.array([[1e-4, 0], [0, 1e-4]])

# Measurement Noise
mN = np.array([0])
cN = np.array([[1e-3]])

# Create distributions
pDist = multivariate_normal(mean=mP, cov=cP) # Process Disturbance
nDist = multivariate_normal(mean=mN, cov=cN) # Measurement Noise

# Construct a continuous-time system
m, ks, kd = 5, 200, 30
Ac = np.array([[0, 1], [-ks/m, -kd/m]])
Cc = np.array([[1, 0]])
Bc = np.array([[0], [1/m]])

# Discretize the system
h, simSteps = 5e-3, 1500
A = np.linalg.inv(np.eye(2) - h*Ac)
B = h * np.matmul(A, Bc)
C = Cc

# Select the initial state
x0 = np.array([[0.1], [0.01]])

# Control Input
cI = 100 * np.ones((1, simSteps))

# Zero-state trajectory
sT = np.zeros(shape=(2, simSteps+1))

# Output
output = np.zeros(shape=(1, simSteps))

# Set the initial state
sT[:, [0]] = x0

# Simulate the state-space model
for i in range(simSteps):
    sT[:, [i + 1]] = np.matmul(A, sT[:, [i]]) + np.matmul(B, cI[:, [i]]) + pDist.rvs(size=1).reshape(2,1)
    output[:, [i]] = np.matmul(C, sT[:, [i]])                            + nDist.rvs(size=1).reshape(1,1)

# Create a time vector
timeVector = np.linspace(0, (simSteps - 1) * h, simSteps)

# Plot the time response
plt.figure(figsize=(10,8))
plt.plot(timeVector,sT[0,0:simSteps], color='blue', linewidth=4)
plt.title("State", fontsize=14)
plt.xlabel("time", fontsize=14)
plt.ylabel("State",fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(visible=True)
plt.savefig("stateTrajectoryTime.png", dpi=600)
plt.show()
 
# plot the state-space trajectory
plt.figure(figsize=(10,8))
plt.plot(sT[0,0:simSteps], sT[1,0:simSteps], color='blue', linewidth=4, alpha=0.5)
plt.title("State", fontsize=16)
plt.xlabel("x1", fontsize=16)
plt.ylabel("x2", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(visible=True)
plt.scatter(sT[0,0], sT[1,0], s=500, c='r', marker='o', label='Start')
plt.scatter(sT[0,-1], sT[1,-1], s=500, c='k', marker='o', linewidth=6, label='End' )
plt.legend(fontsize=14)
plt.savefig("stateTrajectory.png", dpi=600)
plt.show()
