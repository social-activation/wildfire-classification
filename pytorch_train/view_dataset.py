import os
import numpy as np
import matplotlib.pyplot as plt

def plot_fire_nofire_grid(fire_dir, nofire_dir, num_rows=5):
    # Load .npy files
    fire_files = sorted([f for f in os.listdir(fire_dir) if f.endswith('.npy')])
    nofire_files = sorted([f for f in os.listdir(nofire_dir) if f.endswith('.npy')])

    print(len(fire_files))

    # Select only 10 files from each class
    fire_samples = fire_files[:50]
    nofire_samples = nofire_files[:50]

    fig, axes = plt.subplots(num_rows, 4, figsize=(10, num_rows*3))
    axes = axes.flatten()

    idx_fire = 0
    idx_nofire = 0
    for row in range(num_rows):

        for col in range(2):  # Fire columns
            print(idx_fire)
            if idx_fire < len(fire_samples):
                data = np.load(os.path.join(fire_dir, fire_samples[idx_fire]))
                axes[row * 4 + col].imshow(data, cmap='hot')
                axes[row * 4 + col].set_title(f"Fire {idx_fire + 1}")
                axes[row * 4 + col].axis('off')
                idx_fire += 1

        for col in range(2, 4):  # NoFire columns
            if idx_nofire < len(nofire_samples):
                data = np.load(os.path.join(nofire_dir, nofire_samples[idx_nofire]))
                axes[row * 4 + col].imshow(data, cmap='gray')
                axes[row * 4 + col].set_title(f"NoFire {idx_nofire + 1}")
                axes[row * 4 + col].axis('off')

                idx_nofire += 1

    plt.tight_layout()
    plt.savefig("wildfire_dataset/example_images.png")

# Example usage:
plot_fire_nofire_grid('wildfire_dataset/train/fire', 'wildfire_dataset/train/nofire', num_rows=6)
