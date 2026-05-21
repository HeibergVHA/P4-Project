import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats

# Find all .npy files in the execution directory
npy_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "test_03", "*.npy"))) #"test_06", 

if not npy_files:
    raise FileNotFoundError("No .npy files found in the execution directory.")

# Load all .npy files and stack into a single 3D array of shape (N, 4, 4)
data = np.stack([np.load(f) for f in npy_files], axis=0)

#print(f"Loaded {len(npy_files)} file(s): {[os.path.basename(f) for f in npy_files]}")
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
plt.scatter(np.average(vector[:, 0]), np.average(vector[:, 1]), color='blue', marker='x', s=100, label='Average Position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X vs Y from template matching results')
plt.grid()
plt.show()



mu_X, std_X = stats.norm.fit(vector[:, 0])
mu_Y, std_Y = stats.norm.fit(vector[:, 1])

x_linspace = np.linspace(mu_X - 3 * std_X, mu_X + 3 * std_X, 100)
y_linspace = np.linspace(mu_Y - 3 * std_Y, mu_Y + 3 * std_Y, 100)

## Hypothesis testing and stuff
# scipy automatically computes t-based CI
confidence = 0.95
alpha = (1 - confidence)/2

t_crit_x = stats.t.ppf(1 - alpha/2, df=len(vector[:, 0])-1)
t_crit_y = stats.t.ppf(1 - alpha/2, df=len(vector[:, 1])-1)

prediction_margin_x = t_crit_x * np.std(vector[:, 0]) * np.sqrt(1 + 1/len(vector[:, 0]))
prediction_margin_y = t_crit_y * np.std(vector[:, 1]) * np.sqrt(1 + 1/len(vector[:, 1]))

prediction_interval_x = (
    np.mean(vector[:, 0]) - prediction_margin_x,
    np.mean(vector[:, 0]) + prediction_margin_x
)

prediction_interval_y = (
    np.mean(vector[:, 1]) - prediction_margin_y,
    np.mean(vector[:, 1]) + prediction_margin_y
)

print(f"95% Prediction Interval for X: {prediction_interval_x}")
print(f"95% Prediction Interval for Y: {prediction_interval_y}")
print(f"mean X: {mu_X}, std X: {std_X}")
print(f"mean Y: {mu_Y}, std Y: {std_Y}")

#plt.axvline(prediction_interval_x[0], linestyle='--', color='blue')
#plt.axvline(prediction_interval_x[1], linestyle='--', color='blue')
#plt.axvline(prediction_interval_y[0], linestyle='--', color='red')
#plt.axvline(prediction_interval_y[1], linestyle='--', color='red')

#plt.plot(x_linspace, stats.norm.pdf(x_linspace, mu_X, std_X), 'b-', linewidth=2)
#plt.plot(y_linspace, stats.norm.pdf(y_linspace, mu_Y, std_Y), 'r-', linewidth=2)

hist_x = np.histogram(vector[:, 0], bins=20)
hist_y = np.histogram(vector[:, 1], bins=20)

plt.hist(vector[:, 0], bins=20, alpha=0.5, label='X', color='blue')
plt.hist(vector[:, 1], bins=20, alpha=0.5, label='Y', color='red')
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Histogram of X and Y coordinates of template matching results')
plt.legend()
plt.grid()
plt.show()
