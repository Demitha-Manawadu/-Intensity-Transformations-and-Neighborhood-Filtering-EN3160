import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Paths ---
IMG_PATH = r"C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\emma.jpg"
OUT_DIR = r"C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\outputs"
os.makedirs(OUT_DIR, exist_ok=True)  # make sure folder exists

FUNC_PATH = os.path.join(OUT_DIR, "q1_function.png")
COMP_PATH = os.path.join(OUT_DIR, "q1_comparison.png")
TRANSFORMED_SAVE = os.path.join(OUT_DIR, "emma_transformed.jpg")

# Load image
image = cv2.imread(IMG_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image at: {IMG_PATH}")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Making the transform matrix
int_mat = np.zeros(256, dtype=np.uint8)
for i in range(256):
    if i <= 50:
        int_mat[i] = int(i)
    elif 50 < i < 150:
        int_mat[i] = int(((256 - 100) / 100) * int(i) + 22)
    else:
        int_mat[i] = int(i)

# Apply the transformation
transformed_image = cv2.LUT(gray_image, int_mat)

# -------- Figure 1: Intensity Transformation Curve --------
plt.figure()
plt.plot(range(256), int_mat)
plt.title('Intensity Transformation Curve')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)
plt.tight_layout()
plt.savefig(FUNC_PATH, dpi=200)
plt.show()

# Save transformed image
cv2.imwrite(TRANSFORMED_SAVE, transformed_image)

# -------- Figure 2: Comparison (Original vs Transformed) --------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original (Grayscale)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(transformed_image, cmap='gray')
plt.title('Transformed')
plt.axis('off')

plt.tight_layout()
plt.savefig(COMP_PATH, dpi=200)
plt.show()


