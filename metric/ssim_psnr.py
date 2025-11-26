import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import glob
'''
def calculate_metrics(original_path, modified_path):
    original = cv2.imread(original_path)
    modified = cv2.imread(modified_path)

    # Convert to grayscale for SSIM
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_mod = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)

    # SSIM
    ssim_score, _ = ssim(gray_orig, gray_mod, full=True)

    # PSNR
    psnr_score = cv2.PSNR(original, modified)

    return ssim_score, psnr_score




original_folder = 'C:/Users/co33599/Documents/video_dataset_calib/Testing3/frames-Testv3/'
modified_folder = 'C:/Users/co33599/Documents/video_dataset_calib/output/Testing3/fout-Testv3/'

original_images = sorted(glob.glob(original_folder + '*.jpg'))
modified_images = sorted(glob.glob(modified_folder + '*.jpg'))

ssim_total = 0
psnr_total = 0
count = 0

for orig, mod in zip(original_images, modified_images):
    ssim_score, psnr_score = calculate_metrics(orig, mod)
    ssim_total += ssim_score
    psnr_total += psnr_score
    count += 1

print(f"Average SSIM: {ssim_total / count:.4f}")
print(f"Average PSNR: {psnr_total / count:.2f} dB")
'''
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import glob
import matplotlib.pyplot as plt  # Added for plotting

def calculate_metrics(original_path, modified_path):
    original = cv2.imread(original_path)
    modified = cv2.imread(modified_path)

    # Convert to grayscale for SSIM
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_mod = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)

    # SSIM
    ssim_score, _ = ssim(gray_orig, gray_mod, full=True)

    # PSNR
    psnr_score = cv2.PSNR(original, modified)

    return ssim_score, psnr_score

# Paths
original_folder = 'C:/Users/co33599/Documents/video_dataset_calib/Testing5/frames-Testv5/'
modified_folder = 'C:/Users/co33599/Documents/video_dataset_calib/output/Testing5/fout-Testv5/'

# Image files
original_images = sorted(glob.glob(original_folder + '*.jpg'))
modified_images = sorted(glob.glob(modified_folder + '*.jpg'))

# Store metrics
ssim_values = []
psnr_values = []
image_indices = []

# Loop through all image pairs
for idx, (orig, mod) in enumerate(zip(original_images, modified_images), start=1):
    ssim_score, psnr_score = calculate_metrics(orig, mod)
    ssim_values.append(ssim_score)
    psnr_values.append(psnr_score)
    image_indices.append(idx)

# Compute and print averages
avg_ssim = sum(ssim_values) / len(ssim_values)
avg_psnr = sum(psnr_values) / len(psnr_values)

print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average PSNR: {avg_psnr:.2f} dB")

'''
# === PLOTTING ===
plt.figure(figsize=(12, 6))

# SSIM plot
plt.plot(image_indices, ssim_values, label='SSIM', color='blue')


# PSNR plot
plt.plot(image_indices, psnr_values,  label='PSNR (dB)', color='red')

# Labels, legend, etc.
plt.title(" PSNR per Image")
plt.xlabel("Image Index")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
'''
fig, ax1 = plt.subplots()

# Plot SSIM on left y-axis
ax1.plot(image_indices, ssim_values, label='SSIM', color='blue')
ax1.set_xlabel("Frames")
ax1.set_ylabel("SSIM")
ax1.tick_params(axis='y')

# Create a second y-axis for PSNR
ax2 = ax1.twinx()
ax2.plot(image_indices, psnr_values, label='PSNR (dB)', color='green')
ax2.set_ylabel("PSNR (dB)")
ax2.tick_params(axis='y')

# Add title and grid
plt.title("PSNR and SSIM per frames")
ax1.grid(True)

# Legends for both lines combined
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

plt.tight_layout()
plt.show()

# === Scatter Plot: SSIM vs PSNR ===
plt.figure(figsize=(10, 6))

# Normalize frame indices for color mapping
colors = np.linspace(0, 1, len(ssim_values))

scatter = plt.scatter(ssim_values, psnr_values,color='green' , s=40, edgecolor='k')#c=colors, cmap='viridis'

# Axis labels and title
plt.xlabel('SSIM')
plt.ylabel('PSNR (dB)')
plt.title('SSIM vs PSNR (750 Frames)')

# Add colorbar to show frame progression
#cbar = plt.colorbar(scatter)
#cbar.set_label('Frame Index')

# Optional: highlight low-quality points
for i, (s, p) in enumerate(zip(ssim_values, psnr_values)):
    if s < 0.9 or p < 30:
        plt.scatter(s, p, color='red', edgecolors='black', s=40)
        

#plt.grid(True)
plt.tight_layout()
plt.show()
