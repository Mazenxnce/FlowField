import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Initialize and create dataframe
headers = ['Particle_X', 'Particle_Y', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'Velocity_X', 'Velocity_Y']
dataset = pd.DataFrame(columns=headers)
compiled_data = []
row = []

### Physical Dimensions of Hele-Shaw Cell in 2D with number or reference particle/location = particle, n = number of inlet/outlet
H = 1
w = 1
particle = 1
n = 6
# X represents the position of the particle (must be within the boundary)

for k in tqdm(range(1000000)):
    velocity = []
    x = []


    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __str__(self):
            return str((self.x, self.y))


    class Circle:
        def __init__(self, origin, radius):
            self.origin = origin
            self.radius = radius


    origin = Point(0, 0)
    radius = w
    circle = Circle(origin, radius)

    for i in range(particle):
        p = random.random() * 2 * np.pi
        r = circle.radius * np.sqrt(random.random())
        x.append(r * np.cos(p))
        x.append(r * np.sin(p))

    ParticleLocation = np.array(x)
    row.append(x)

    # Q represents the vector that contains elements that represent the point flow rate values
    Q = []
    Q = StandardScaler().fit_transform(np.random.rand(6, 1, ))
    Q = Q.ravel().tolist()
    # for i in range(n - 1):
    #     Q.append(random.uniform(-1.0, 1.0))
    #     # Remember for mass conservation
    # Q.append(0 - sum(Q))
    row.append(Q)

    # R represents the vector that contains elements that represent the coordinates of the ith source/sink
    R = []
    for i in range(n):
        R_coordinates = [w * np.cos(2 * (i - 1) * np.pi / n), w * np.sin(2 * (i - 1) * np.pi / n)]
        R.append(R_coordinates)
    PointSource = np.array(R)

    # Given the defined inlet/outlets, the velocity of the particle
    for i in range(n):
        velocity.append(Q[i] * (ParticleLocation - (R[i])) / (np.linalg.norm((ParticleLocation - (R[i])))) ** 2)

    # Omit pi for non-dimensionalisation
    particle_vel = (sum(velocity))
    row.append(particle_vel.tolist())

    data_row = (sum(row, []))
    dataset.loc[len(dataset)] = data_row
    data_row.clear()
    row.clear()

# print(dataset)
# dataset.boxplot()
# plt.show()
dataset.to_csv("2D_Hele_Shaw_Data (test).csv", index=False)
