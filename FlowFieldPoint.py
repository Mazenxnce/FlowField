import random
import numpy as np
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt

# Initialize and create dataframe
headers = ['Particle_X', 'Particle_Y', 'q1', 'q2',
           'q3', 'q4', 'q5', 'q6', 'Velocity_X', 'Velocity_Y']
dataset = pd.DataFrame(columns=headers)

# Physical Dimensions of 2D Hele-Shaw Cell
H = 1
w = 1
particle = 1
velocity = []
n = 6

# X represents the position of the particle (must be within the boundary)
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
    x.append([r * np.cos(p), r * np.sin(p)])

ParticleLocation = np.array(x)

# Q represents the vector that contains elements that represent the point flow rate values
# Summation of all Q must be = 0 to satisfy mass conservation
Q = []
for i in range(n-1):
    Q.append(random.uniform(-1.0, 1.0))
Q.append(0-sum(Q))

# R represents the vector that contains elements that represent the coordinates of the ith source/sink
R = []

# Plotting the Circle and its inlets/outlets #
theta = np.linspace(0, 2 * np.pi, 100)
r = np.sqrt(1.0)
x1 = w * np.cos(theta)
x2 = w * np.sin(theta)

fig, ax = plt.subplots(1)

ax.plot(x1, x2)
ax.set_aspect(1)

plt.xlim(-1.25 * w, 1.25 * w)
plt.ylim(-1.25 * w, 1.25 * w)

plt.grid(linestyle='--')
plt.title('2D Hele-Shaw Point Source Model', fontsize=8)
plt.savefig("plot_circle_matplotlib_01.png", bbox_inches='tight')

# Determines the coordinates of the point sources/sinks based on number of inlet/outlet defined by n
for i in range(n):
    R_coordinates = [w * np.cos(2 * (i - 1.0) * np.pi / n),
                     w * np.sin(2 * (i - 1.0) * np.pi / n)]
    R.append(R_coordinates)

PointSource = np.array(R)

for n, txt in enumerate(Q):
    ax.annotate('{:.3f}'.format(txt), (R[n]))

plt.scatter(*zip(*x))
plt.scatter(*zip(*R))
plt.show()

# Given the defined inlet/outlets, the velocity of the particle
for i in range(n):
    velocity.append(
        Q[i] * (ParticleLocation - (R[i])) / (np.linalg.norm((ParticleLocation - (R[i]))))**2)


# Prints numerical results (for testing)
print("The randomized particle location, X  : ")
print(ParticleLocation)

print("The randomized inlet and outlet flowrates Q  : ")
print(np.array(Q))

print("The locations of the inlet/outlet  : ")
print(PointSource)

print("The velocity of the particle  : ")
print((sum(velocity)))
