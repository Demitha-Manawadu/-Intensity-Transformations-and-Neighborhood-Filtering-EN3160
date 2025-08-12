import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load the image 
img_orig = cv.imread(r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\jeniffer.jpg')
jennifer_hsv = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)
jennifer_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)

# Split into HSV planes
H, S, V = cv.split(jennifer_hsv)

# Apply thresholding on Saturation plane
_, mask = cv.threshold(S, 12, 255, cv.THRESH_BINARY)

# Extract the foreground using the mask
foreground = cv.bitwise_and(img_orig, img_orig, mask=mask)

# ---------- FIGURE 1: Row1 HSV / Row2 Foreground + Mask ----------
fig, ax = plt.subplots(2, 3, figsize=(15, 8))

# Row 1: H, S, V
ax[0, 0].imshow(H, cmap='gray', vmin=0, vmax=255)
ax[0, 0].set_title('Hue'); ax[0, 0].axis("off")

ax[0, 1].imshow(S, cmap='gray', vmin=0, vmax=255)
ax[0, 1].set_title('Saturation'); ax[0, 1].axis("off")

ax[0, 2].imshow(V, cmap='gray', vmin=0, vmax=255)
ax[0, 2].set_title('Value'); ax[0, 2].axis("off")

# Row 2: Foreground (RGB) and Mask; leave last empty
ax[1, 0].imshow(cv.cvtColor(foreground, cv.COLOR_BGR2RGB))
ax[1, 0].set_title('Extracted Foreground'); ax[1, 0].axis('off')

ax[1, 1].imshow(mask, cmap='gray')
ax[1, 1].set_title('Mask'); ax[1, 1].axis('off')

ax[1, 2].axis('off')  # empty slot

plt.tight_layout()
plt.savefig(r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\outputs\Q6_split and extracted_images.png')
plt.show()

# ---------- HISTOGRAM PREP ----------
foreground_hsv = cv.cvtColor(foreground, cv.COLOR_BGR2HSV)
H_fg, S_fg, V_fg = cv.split(foreground_hsv)

# Histogram of Value channel (foreground only)
hist = cv.calcHist([V_fg], [0], mask, [256], [0, 256])
cdf = hist.cumsum()
pixels = cdf[-1]

# Define equalization LUT and apply
t = np.array([(256 - 1) / (pixels) * cdf[k] for k in range(256)]).astype('uint8')
V_eq = t[V_fg]

# Equalized histogram
hist_eq = cv.calcHist([V_eq], [0], mask, [256], [0, 256])

# ---------- FIGURE 2: Three histogram-related plots ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Histogram of V
axs[0].bar(np.arange(256), hist.flatten(), color='black', width=1)
axs[0].set_title('Histogram of Value (Foreground)')
axs[0].set_xlabel('Pixel Intensity'); axs[0].set_ylabel('Frequency')
axs[0].set_xlim([0, 256]); axs[0].grid(True)

# Cumulative histogram
axs[1].plot(cdf, color='black')
axs[1].set_title('Cumulative Histogram')
axs[1].set_xlabel('Pixel Intensity'); axs[1].set_ylabel('Frequency')
axs[1].grid(True)

# Equalized histogram
axs[2].bar(np.arange(256), hist_eq.flatten(), color='black', width=1)
axs[2].set_title('Equalized Histogram (Value Channel)')
axs[2].set_xlabel('Pixel Intensity'); axs[2].set_ylabel('Frequency')
axs[2].set_xlim([0, 256]); axs[2].grid(True)

plt.tight_layout()
plt.savefig(r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\outputs\Q6_Hist.png')

plt.show()

# ---------- FINAL MERGE (unchanged from your version) ----------
merged = cv.merge([H_fg, S_fg, V_eq])
foreground_modified = cv.cvtColor(merged, cv.COLOR_HSV2RGB)
background = cv.bitwise_and(img_orig, img_orig, mask=cv.bitwise_not(mask))
result = cv.add(cv.cvtColor(background, cv.COLOR_BGR2RGB), foreground_modified)

# FIGURE 3: Original vs Foreground Equalized
fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs[0].imshow(jennifer_rgb); axs[0].set_title('Original'); axs[0].axis('off')
axs[1].imshow(result); axs[1].set_title('Foreground Equalized'); axs[1].axis('off')
plt.tight_layout()
plt.savefig(r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\outputs\Q6_comparison.png')
plt.show()
