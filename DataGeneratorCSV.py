import datetime
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Initialize and create dataframe
headers = ['Particle_X', 'Particle_Y', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'Velocity_X', 'Velocity_Y', '_time', 'Description']
dataset = pd.DataFrame(columns=headers)
compiled_data = []
row = []

### Physical Dimensions of Hele-Shaw Cell in 2D with number or reference particle/location = particle, n = number of inlet/outlet
H = 1
w = 1
particle = 1
n = 6
# X represents the position of the particle (must be within the boundary)

for k in tqdm(range(10000)):
    velocity = []
    x = []


    class Origin:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __str__(self):
            return str((self.x, self.y))

    class Circle:
        def __init__(self, origin, radius):
            self.origin = origin
            self.radius = radius

    origin = Origin(0, 0)
    radius = w
    circle = Circle(origin, radius)

    for i in range(particle):
        p = random.random() * 2 * np.pi
        r = circle.radius * np.sqrt(random.random())
        x.append(r * np.cos(p))
        x.append(r * np.sin(p))

        # Append to db
        ParticleX = r * np.cos(p)
        ParticleY = r * np.sin(p)

    ParticleLocation = np.array(x)
    row.append(x)

    # Q represents the vector that contains elements that represent the point flow rate values
    Q = []
    Q = StandardScaler().fit_transform(np.random.rand(6, 1, ))
    Q = Q.ravel().tolist()
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
    data_row.append(datetime.datetime.utcnow().isoformat())
    data_row.append('CSV_TestingData')
    dataset.loc[len(dataset)] = data_row
    data_row.clear()
    row.clear()


cols = dataset.columns.tolist()
cols = cols[-2:] + cols[:-2]
dataset = dataset[cols]


dataset.to_csv("HeleShawAnalyticalSolution.csv", index=False)
