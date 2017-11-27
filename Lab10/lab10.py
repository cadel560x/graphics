# 3) Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 4) Getting the image
imgOrig = cv2.imread('GMIT.jpg',)

# 5) Convert to grayscale
gray_image = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray_image.png',gray_image)
# cv2.imshow('gray_image', gray_image)
# input()

# 6) Plotting images with Python and Matplotlib
img = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB)

nrows = 2
ncols = 2

plt.subplot(nrows,ncols,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(gray_image,cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
# plt.show()

# 7) Applying blur with a 3x3
KernelSizeWidth = 3
KernelSizeHeight = 3

imgBlur3x3 = cv2.GaussianBlur(gray_image,(KernelSizeWidth,KernelSizeHeight),0)

# 8) Plot the 3x3 blur
plt.subplot(nrows, ncols,3),plt.imshow(imgBlur3x3,cmap = 'gray')
plt.title('Blur 3x3'), plt.xticks([]), plt.yticks([])


# 7) Applying blur with a 13x13
KernelSizeWidth = 13
KernelSizeHeight = 13

imgBlur13x13 = cv2.GaussianBlur(gray_image,(KernelSizeWidth,KernelSizeHeight),0)

# 8) Plot the 13x13 blur
plt.subplot(nrows, ncols,4),plt.imshow(imgBlur13x13,cmap = 'gray')
plt.title('Blur 3x3'), plt.xticks([]), plt.yticks([])
plt.show()