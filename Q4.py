import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load the image 
img_orig = cv.imread(r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\spider.png', cv.IMREAD_COLOR)

# Split into HSV planes
hsv_img = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv_img)

# Apply the intensity transformation to saturation plane
a = 0.5 
sigma = 70
transformation = lambda x: np.minimum(x + a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2)), 255)
s_transformed = transformation(s).astype(np.uint8)

# Recombine the planes
hsv_enhanced = cv.merge([h, s_transformed, v])
enhanced_img = cv.cvtColor(hsv_enhanced, cv.COLOR_HSV2BGR)

# Plot the intensity transformation
x = np.arange(256)
trans_plot = transformation(x)

# -------- FIGURE 1: Hue, Saturation, Value --------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(h, cmap='gray')
plt.title('Hue Plane')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(s, cmap='gray')
plt.title('Saturation Plane')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(v, cmap='gray')
plt.title('Value Plane')
plt.axis('off')

plt.tight_layout()
plt.savefig(r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\outputs\Q4_hsv.png')
plt.show()

# -------- FIGURE 2: Original, Enhanced, Transformation --------
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(enhanced_img, cv.COLOR_BGR2RGB))
plt.title(f'Vibrance-Enhanced Image (a = {a})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.plot(x, trans_plot)
plt.title('Intensity Transformation')
plt.xlabel('Input Saturation')
plt.ylabel('Output Saturation')
plt.grid(True)

plt.tight_layout()
plt.savefig(r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\outputs\Q4_enhanced.png')
plt.show()
