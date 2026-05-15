import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def rotate_vector(v, degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    
    # Create the rotation matrix
    R = np.array(((c, -s), (s, c)))
    
    # Dot product for rotation
    return np.dot(R, v)

# Find all .npy files in the execution directory
npy_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.npy")))

if not npy_files:
    raise FileNotFoundError("No .npy files found in the execution directory.")

# Load all .npy files and stack into a single 3D array of shape (N, 4, 4)
data = np.stack([np.load(f) for f in npy_files], axis=0)

print(f"Loaded {len(npy_files)} file(s): {[os.path.basename(f) for f in npy_files]}")
print(f"Combined array shape: {data.shape}, dtype: {data.dtype}")

# Access individual transform matrices by index, e.g.:
# data[0]  -> first  4x4 matrix
# data[1]  -> second 4x4 matrix
# data[-1] -> last   4x4 matrix

vector = np.zeros((len(data), 2))

for i in range(len(data)):
    vector[i] = [data[i][0][3], data[i][1][3]]

#print(vector)

plt.scatter(vector[:, 0], vector[:, 1], lw=0.2, edgecolor='r')
plt.gca().add_patch(patches.Rectangle((0.54, -0.65), 0.7, 0.4, angle=130, 
                         edgecolor='r', facecolor='none', lw=2))
plt.scatter(np.average(vector[:, 0]), np.average(vector[:, 1]), color='blue', marker='x', s=100, label='Average Position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X vs Y from Transform Matrices')
plt.grid()
plt.show()


for i in range(len(vector)):
    vector[i] = rotate_vector(vector[i], -30)

plt.scatter(vector[:, 0], vector[:, 1], lw=0.2, edgecolor='r')
plt.gca().add_patch(patches.Rectangle((0.54, -0.65), 0.7, 0.4, angle=130, 
                         edgecolor='r', facecolor='none', lw=2))
plt.scatter(np.average(vector[:, 0]), np.average(vector[:, 1]), color='blue', marker='x', s=100, label='Average Position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X vs Y from Transform Matrices')
plt.grid()
plt.show()

mu = np.average(vector[:, 0]), np.average(vector[:, 1])
std = np.sqrt(np.var(vector[:, 0])), np.sqrt(np.var(vector[:, 1]))

hist_x = np.histogram(vector[:, 0], bins=20)
hist_y = np.histogram(vector[:, 1], bins=20)

plt.hist(vector[:, 0], bins=20, alpha=0.5, label='X')
plt.hist(vector[:, 1], bins=20, alpha=0.5, label='Y')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of X and Y from Transform Matrices')
plt.legend()
plt.grid()
plt.show()
