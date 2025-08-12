import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

img_orig = cv.imread(
    r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\a1q5images\taylor_small.jpg',
    cv.IMREAD_COLOR
)
img_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)

def ssd(img1, img2):
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    return float(np.sum((a - b) ** 2))

def zooming(original_image, zoom_factor):
    h, w, c = original_image.shape

    # Zoomed image creation
    zh = int(h * zoom_factor)
    zw = int(w * zoom_factor)
    zoomed_image = np.zeros((zh, zw, c), dtype=np.uint8)

    for i in range(zh):
        for j in range(zw):
            zoomed_image[i, j] = original_image[int(i / zoom_factor), int(j / zoom_factor)]

    zoomed_image_rgb = cv.cvtColor(zoomed_image, cv.COLOR_BGR2RGB)

    # Save zoomed image
    out_dir = r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\a1q5images'
    out_path = os.path.join(out_dir, 'taylor_small_nearest_neighbour_zoom.png')
    cv.imwrite(out_path, zoomed_image)

    # Read and resize big image
    big_path = r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\a1q5images\taylor.jpg'
    img_BIG = cv.imread(big_path, cv.IMREAD_COLOR)
    if img_BIG is None:
        raise FileNotFoundError(f"Could not read big image at: {big_path}")

    img_BIG_resized = cv.resize(img_BIG, (zw, zh), interpolation=cv.INTER_LINEAR)
    ssd_value = ssd(img_BIG_resized, zoomed_image)

    # Plot with ALL text under images
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Zooming Factor: {zoom_factor}", fontsize=14, fontweight='bold')

    axs[0].imshow(img_rgb)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[0].text(0.5, -0.15,
                f"Shape: {h}x{w}x{c}",
                ha='center', va='center', transform=axs[0].transAxes, fontsize=10)

    axs[1].imshow(zoomed_image_rgb)
    axs[1].set_title('Zoomed Image')
    axs[1].axis('off')
    axs[1].text(0.5, -0.15,
                f"Shape: {zh}x{zw}x{c}\nSSD: {ssd_value:.2f}",
                ha='center', va='center', transform=axs[1].transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(r'C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\outputs\Q8a_nearest-neighbor,.png')

    plt.show()

zoom_factor = 4
zooming(img_orig, zoom_factor)
