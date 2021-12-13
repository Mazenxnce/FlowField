import os
import time
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp2d
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

start = time.time()


def norm(x):
    def function(x):
        return 2 * ((x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))) - 1

    normalized_list = x.apply(function)
    return normalized_list


# Load model and test data from directory
model = keras.models.load_model('backend/Finalized Model.h5')
test_x = np.load('backend/test_x.npy')

# Check time for loading data - so far 0.83s
load_time = time.time()
print(round(load_time - start, 2), ' seconds for loading data')

###############################################################################################################

# Pre-determined Flowrate Matrix where Sum of Qs = 0
# Q = [1, -1, 0.5, -0.5, 0.3, -0.3]
Q = []
Q = StandardScaler().fit_transform(np.random.rand(6, 1, ))
Q = Q.ravel().tolist()

# Check time for generating randomized flowrates - so far instant
flow_time = time.time()
print(round(flow_time - start, 2), ' seconds for generating random flowrates')

# Flowrate and Position matrices
flowrate_matrix = np.array(Q)
x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
position_matrix = np.vstack((x.ravel(), y.ravel())).T

x = [row[0] for row in position_matrix]
y = [row[1] for row in position_matrix]

temp = []
for i in range(len(position_matrix)):
    c = np.hstack((position_matrix[i], flowrate_matrix))
    temp.append(c)
features = np.vstack(temp)

# Model Prediction
uv = model.predict(features)
u = uv[0]
v = uv[1]

predict_time = time.time()
print(round(predict_time - start, 2), ' seconds for generation')

###############################################################################################################

# regularly spaced grid spanning the domain of x and y
xi = np.array(x)
yi = np.array(y)

xii = np.linspace(xi.min(), xi.max(), xi.size)
yii = np.linspace(yi.min(), yi.max(), yi.size)

# bicubic interpolation
uCi = interp2d(xi, yi, u)(xii, yii)
vCi = interp2d(xi, yi, v)(xii, yii)

###############################################################################################################

speed = np.sqrt(uCi ** 2 + vCi ** 2)
lw = 2 * speed / speed.max() + .5

fig0, ax0 = plt.subplots(num=None, figsize=(
    11, 9), dpi=80, facecolor='w', edgecolor='k')
circle3 = plt.Circle((0, 0), 1, color='black', fill=False)

ax0.streamplot(xii, yii, uCi, vCi,
               density=4, color=speed, linewidth=lw, arrowsize=0.1, cmap=plt.cm.jet)
ax0.add_artist(circle3)

strm = ax0.streamplot(xii, yii, uCi, vCi,
                      color=speed, linewidth=lw, density=[4, 4], cmap=plt.cm.jet)
cbar = fig0.colorbar(strm.lines,
                     fraction=0.046, pad=0.04)
cbar.set_label('Velocity',
               rotation=270, labelpad=8)
cbar.set_clim(0, 1500)
cbar.draw_all()
ax0.set_ylim([-1.2, 1.2])
ax0.set_xlim([-1.2, 1.2])
ax0.set_xlabel('x [Length]')
ax0.set_ylabel('z [Width]')
ax0.set_aspect(1)
plt.title('Flow Field', y=1.01)
# plt.savefig('Flow Field.pdf', bbox_inches=0)

end = time.time()
print(round((end - start), 2), ' seconds for plotting')
plt.show()

###############################################################################################################
