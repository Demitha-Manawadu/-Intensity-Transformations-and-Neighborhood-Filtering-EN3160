import cv2
import numpy as np
import matplotlib.pyplot as plt

#load image
image = cv2.imread(r"C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\emma.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#making the transform matrix
int_mat = np.zeros(256, dtype=np.uint8)
for i in range(256):
    if i <= 50:
        int_mat[i] = int(i)
    elif 50 < i < 150:
        int_mat[i] = int(((256-100)/100)*int(i)+22)
    else:
        int_mat[i] = int(i)

#apply the transformation
transformed_image = cv2.LUT(gray_image, int_mat)

#printing transformation curve
plt.figure()
plt.plot(range(256), int_mat)
plt.title('Intensity Transformation Curve')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)
plt.show()

#save
cv2.imwrite(r"C:\Academic\robo games\git repo\-Intensity-Transformations-and-Neighborhood-Filtering-EN3160\a1images\a1images\outputs\emma_transformed.jpg", transformed_image)
plt.imshow(transformed_image, cmap='gray')
plt.title("Transformed Image")
plt.axis('off')
plt.show()
