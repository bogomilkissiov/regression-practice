# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from matplotlib.animation import FuncAnimation
import helpers_regression as h
# pyrefly: ignore [missing-import]
import catppuccin as cp

# Load the dataset
raw_data = pd.read_csv("auto-mpg.csv")

# Drop the car_name column
raw_data = raw_data.drop(columns=['car_name'])

# Drop the 'origin' column (don't think one-hot encoding adds much value)
raw_data = raw_data.drop(columns=['origin'])

# Randomize the row order
raw_data = raw_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Standardize the data
raw_data = (raw_data - raw_data.mean()) / raw_data.std()

# Split features and labels using column names first, then convert to NumPy arrays
X = raw_data.drop(columns=['mpg']).to_numpy()
y = raw_data['mpg'].to_numpy()

# Project the data onto PC1 and PC2 for visualization purposes only
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# add column of 1s to X for bias term
X_pca = np.hstack([X_pca, np.ones((X_pca.shape[0], 1))])

# Initialize theta
theta = np.random.randn(X_pca.shape[1])

# train standard and ridge models
standard_theta_history, standard_obj_history = h.grad_descent(X_pca, y, lam=0, alpha=0.01, iterations=1000, initial_theta=theta)
ridge_theta_history, ridge_obj_history = h.grad_descent(X_pca, y, lam=1, alpha=0.01, iterations=1000, initial_theta=theta)

# compute analytical solutions
standard_analytical_theta = h.analytical_regression(X_pca, y)
ridge_analytical_theta = h.analytical_ridge_regression(X_pca, y, lam=1)

# Animations
# Apply Catppuccin Mocha colors manually (bypassing the style registry bug)
plt.rcParams.update({
    "axes.facecolor": cp.PALETTE.mocha.colors.base.hex,
    "figure.facecolor": cp.PALETTE.mocha.colors.mantle.hex,
    "text.color": cp.PALETTE.mocha.colors.text.hex,
    "axes.labelcolor": cp.PALETTE.mocha.colors.text.hex,
    "xtick.color": cp.PALETTE.mocha.colors.text.hex,
    "ytick.color": cp.PALETTE.mocha.colors.text.hex,
    "grid.color": cp.PALETTE.mocha.colors.surface0.hex,
})

# Set up the figure with 2 3D subplots side by side
fig = plt.figure(figsize=(16, 8))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95, wspace=0.05)
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Data points coordinates
pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]

# Data limits
pc1_min, pc1_max = pc1.min(), pc1.max()
pc2_min, pc2_max = pc2.min(), pc2.max()
y_min, y_max = y.min(), y.max()

# Create a meshgrid for the analytical plane surfaces (extends 20% beyond data)
pc1_margin_analyt = (pc1_max - pc1_min) * 0.2
pc2_margin_analyt = (pc2_max - pc2_min) * 0.2
pc1_range_analyt = np.linspace(pc1_min - pc1_margin_analyt, pc1_max + pc1_margin_analyt, 10)
pc2_range_analyt = np.linspace(pc2_min - pc2_margin_analyt, pc2_max + pc2_margin_analyt, 10)
PC1_mesh_analyt, PC2_mesh_analyt = np.meshgrid(pc1_range_analyt, pc2_range_analyt)
mesh_points_analyt = np.c_[PC1_mesh_analyt.ravel(), PC2_mesh_analyt.ravel(), np.ones_like(PC1_mesh_analyt.ravel())]

# Create a meshgrid for the evolving plane surfaces (extends 10% beyond data)
pc1_margin_gd = (pc1_max - pc1_min) * 0.1
pc2_margin_gd = (pc2_max - pc2_min) * 0.1
pc1_range_gd = np.linspace(pc1_min - pc1_margin_gd, pc1_max + pc1_margin_gd, 10)
pc2_range_gd = np.linspace(pc2_min - pc2_margin_gd, pc2_max + pc2_margin_gd, 10)
PC1_mesh_gd, PC2_mesh_gd = np.meshgrid(pc1_range_gd, pc2_range_gd)
mesh_points_gd = np.c_[PC1_mesh_gd.ravel(), PC2_mesh_gd.ravel(), np.ones_like(PC1_mesh_gd.ravel())]

# Calculate analytical planes
standard_analyt_y = (mesh_points_analyt @ standard_analytical_theta).reshape(PC1_mesh_analyt.shape)
ridge_analyt_y = (mesh_points_analyt @ ridge_analytical_theta).reshape(PC1_mesh_analyt.shape)

def init_ax(ax, title):
    # Plot the datapoints - use a brighter, more distinct color (e.g. yellow)
    ax.scatter(pc1, pc2, y, color=cp.PALETTE.mocha.colors.yellow.hex, alpha=0.9, s=20, label='Data')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('MPG (Standardized)')
    ax.set_title(title, pad=0)
    
    # Lock the axis limits tightly around the data so the extended planes cut through the boundaries
    ax.set_xlim(pc1_min - (pc1_max - pc1_min)*0.1, pc1_max + (pc1_max - pc1_min)*0.1)
    ax.set_ylim(pc2_min - (pc2_max - pc2_min)*0.1, pc2_max + (pc2_max - pc2_min)*0.1)
    ax.set_zlim(y_min - (y_max - y_min)*0.1, y_max + (y_max - y_min)*0.1)
    
    # Remove the background panes (the beige-ish boxes) by making them completely transparent
    transparent_color = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(transparent_color)
    ax.yaxis.set_pane_color(transparent_color)
    ax.zaxis.set_pane_color(transparent_color)
    
    # Remove pane borders (the lines connecting the panes)
    ax.xaxis.line.set_color(transparent_color)
    ax.yaxis.line.set_color(transparent_color)
    ax.zaxis.line.set_color(transparent_color)
    
    # Change viewing angle to be more parallel to the PC1/PC2 plane, slightly rotated CCW
    ax.view_init(elev=10, azim=-70)

init_ax(ax1, 'Standard Regression Evolution')
init_ax(ax2, 'Ridge Regression Evolution (lam=1)')

# Plot analytical planes in a neutral color (overlay0)
neutral_color = cp.PALETTE.mocha.colors.overlay0.hex
ax1.plot_surface(PC1_mesh_analyt, PC2_mesh_analyt, standard_analyt_y, color=neutral_color, alpha=0.3, label='Analytical')
ax2.plot_surface(PC1_mesh_analyt, PC2_mesh_analyt, ridge_analyt_y, color=neutral_color, alpha=0.3, label='Analytical')

# Placeholders for the evolving planes
surf_gd1 = [None]
surf_gd2 = [None]
gd_color = cp.PALETTE.mocha.colors.mauve.hex

# We have 1000 iterations. Use logarithmic sampling to show more early frames where learning is fast.
desired_frames = 100
frame_indices = np.unique(np.geomspace(1, len(standard_theta_history), num=desired_frames, dtype=int)) - 1
num_frames = len(frame_indices)

def animate(frame):
    # Remove previous surfaces if they exist
    if surf_gd1[0] is not None:
        surf_gd1[0].remove()
    if surf_gd2[0] is not None:
        surf_gd2[0].remove()
        
    idx = frame_indices[frame]
    
    theta_std = standard_theta_history[idx]
    theta_ridge = ridge_theta_history[idx]
    
    std_gd_y = (mesh_points_gd @ theta_std).reshape(PC1_mesh_gd.shape)
    ridge_gd_y = (mesh_points_gd @ theta_ridge).reshape(PC1_mesh_gd.shape)
    
    # Plot new surfaces
    surf_gd1[0] = ax1.plot_surface(PC1_mesh_gd, PC2_mesh_gd, std_gd_y, color=gd_color, alpha=0.8)
    surf_gd2[0] = ax2.plot_surface(PC1_mesh_gd, PC2_mesh_gd, ridge_gd_y, color=gd_color, alpha=0.8)
    
    return surf_gd1[0], surf_gd2[0]

print("Generating animation... This might take a minute.")
ani = FuncAnimation(fig, animate, frames=num_frames, interval=50, blit=False)

# Save animation as a GIF
gif_path = 'regression_comparison.gif'
ani.save(gif_path, writer='pillow')
print(f"Animation saved successfully to {gif_path}")
