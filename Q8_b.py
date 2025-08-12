import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

img_orig = cv.imread(
    r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\a1q5images\im01small.png',
    cv.IMREAD_COLOR
)
img_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)

def ssd(img1, img2):
    # Sum of squared differences (cast to float to avoid uint8 overflow)
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    return float(np.sum((a - b) ** 2))

def zooming(original_image, zoom_factor):
    # ---- Bilinear zoom (unchanged logic) ----
    height, width, channels = original_image.shape
    zoomed_height = int(height * zoom_factor)
    zoomed_width  = int(width  * zoom_factor)
    zoomed_image = np.zeros((zoomed_height, zoomed_width, channels), dtype=np.uint8)
    
    y_scale = height / zoomed_height
    x_scale = width  / zoomed_width
    
    for i in range(zoomed_height):
        for j in range(zoomed_width):
            original_y = i * y_scale
            original_x = j * x_scale

            x1, y1 = int(original_x), int(original_y)
            x2, y2 = x1 + 1, y1 + 1

            if x2 >= width:  x2 = width  - 1
            if y2 >= height: y2 = height - 1

            weight_x = original_x - x1
            weight_y = original_y - y1

            pixel_interpolated = (
                (1 - weight_x) * (1 - weight_y) * original_image[y1, x1] +
                weight_x * (1 - weight_y) * original_image[y1, x2] +
                (1 - weight_x) * weight_y * original_image[y2, x1] +
                weight_x * weight_y * original_image[y2, x2]
            )

            zoomed_image[i, j] = pixel_interpolated

    zoomed_image_rgb = cv.cvtColor(zoomed_image, cv.COLOR_BGR2RGB)

    # ---- SSD vs big image (resize big to match zoomed if needed) ----
    big_path = r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\a1q5images\im01.png'
    img_BIG = cv.imread(big_path, cv.IMREAD_COLOR)
    if img_BIG is None:
        raise FileNotFoundError(f"Could not read big image at: {big_path}")

    if img_BIG.shape[:2] != zoomed_image.shape[:2]:
        img_BIG_for_ssd = cv.resize(img_BIG, (zoomed_image.shape[1], zoomed_image.shape[0]), interpolation=cv.INTER_LINEAR)
    else:
        img_BIG_for_ssd = img_BIG

    ssd_value = ssd(img_BIG_for_ssd, zoomed_image)

    # ---- Plot with info UNDER the plots (no prints) ----
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Zooming Factor: {zoom_factor}", fontsize=14, fontweight='bold')

    axs[0].imshow(img_rgb)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[0].text(
        0.5, -0.15,
        f"Shape: {height}x{width}x{channels}",
        ha='center', va='center', transform=axs[0].transAxes, fontsize=10
    )

    axs[1].imshow(zoomed_image_rgb)
    axs[1].set_title('Zoomed Image (Bilinear)')
    axs[1].axis('off')
    axs[1].text(
        0.5, -0.15,
        f"Shape: {zoomed_height}x{zoomed_width}x{channels}\nSSD: {ssd_value:.2f}",
        ha='center', va='center', transform=axs[1].transAxes, fontsize=10
    )

    plt.tight_layout()
    # Save the figure
    plt.savefig(r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\outputs\Q8a_ bilinearinterpolation.png')

    plt.show()

    return zoomed_image

zoom_factor = 4
zoomed_image = zooming(img_orig, zoom_factor)

# Save (same folder, clean filename)
out_dir = r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\a1q5images'
out_path = os.path.join(out_dir, 'im01small_zoomed_bilinear.png')
cv.imwrite(out_path, zoomed_image)
