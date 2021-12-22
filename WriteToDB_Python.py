import datetime
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

token = "Eo8wFg4vDe4nsVMppA-gtsvuMjxCJiYIVjZgb-Ui3UTQ_LWPFNb5a-36zcnfckrBrIVVzl82xzbgOlvfWr_JqQ=="
org = "Maze"
bucket = "AnalyticalData"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)


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

    # Q represents the vector that contains elements that represent the point flow rate values
    Q = []
    Q = StandardScaler().fit_transform(np.random.rand(6, 1, ))
    Q = Q.ravel().tolist()
    row.append(Q)

    # Append to db
    q_1 = Q[0]
    q_2 = Q[1]
    q_3 = Q[2]
    q_4 = Q[3]
    q_5 = Q[4]
    q_6 = Q[5]


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

    # Append to db
    vel = particle_vel.tolist()
    VelX = vel[0]
    VelY = vel[1]
   
    # Write to DB
    datapoint = Point("AnalyticalSolution").tag("Data Description", "Testing Data").field("Particle_X",ParticleX).field("Particle_Y",ParticleY).field("Q1",q_1).field("Q2",q_2).field("Q3",q_3).field("Q4",q_4).field("Q5",q_5).field("Q6",q_6).field("Velocity_X",VelX).field("Velocity_Y",VelY).time(datetime.datetime.utcnow().isoformat())

    with client.write_api(write_options=WriteOptions(batch_size=10000,
                                                    flush_interval=10,
                                                    jitter_interval=0,
                                                    retry_interval=5_000,
                                                    max_retries=5,
                                                    max_retry_delay=30_000,
                                                    exponential_base=2)) as _write_client:
                                                    _write_client.write(bucket, org, datapoint) 

