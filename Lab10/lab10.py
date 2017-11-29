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
# cv2.waitKey()

# 6) Plotting images with Python and Matplotlib
img = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB)

nrows = 2
ncols = 4

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
plt.title('Blur 13x13'), plt.xticks([]), plt.yticks([])

# 9) Sobel edge detection
sobelHorizontal = cv2.Sobel(imgBlur3x3,cv2.CV_64F,1,0,ksize=5)
sobelVertical = cv2.Sobel(imgBlur3x3,cv2.CV_64F,0,1,ksize=5)

# 10) Plot Sobel processed images
plt.subplot(nrows, ncols,5),plt.imshow(sobelHorizontal,cmap = 'gray')
plt.title('Sobel x - Blur 3x3'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,6),plt.imshow(sobelVertical,cmap = 'gray')
plt.title('Sobel y - Blur 3x3'), plt.xticks([]), plt.yticks([])

# 11) Adding the two Sobel processed images
sobelSum = sobelHorizontal + sobelVertical

plt.subplot(nrows, ncols,7),plt.imshow(sobelSum,cmap = 'gray')
plt.title('Sobel sum'), plt.xticks([]), plt.yticks([])

# 12) Apply Canny edge detection
# cannyThreshold = 30
# cannyParam2 = 60
# canny = cv2.Canny(imgBlur3x3, cannyThreshold, cannyParam2)
# cv2.imshow('Canny', canny)
# cv2.waitKey()

# cannyThreshold = 30
# cannyParam2 = 90
# canny = cv2.Canny(imgBlur3x3, cannyThreshold, cannyParam2)
# cv2.imshow('Canny', canny)
# cv2.waitKey()

# cannyThreshold = 30
# cannyParam2 = 120
# canny = cv2.Canny(imgBlur3x3, cannyThreshold, cannyParam2)
# cv2.imshow('Canny', canny)
# cv2.waitKey()

# cannyThreshold = 20
# cannyParam2 = 40
# canny = cv2.Canny(imgBlur3x3, cannyThreshold, cannyParam2)
# cv2.imshow('Canny', canny)
# cv2.waitKey()

cannyThreshold = 20
cannyParam2 = 60
canny = cv2.Canny(imgBlur3x3, cannyThreshold, cannyParam2)
# cv2.imshow('Canny', canny)
# cv2.waitKey()

# cannyThreshold = 20
# cannyParam2 = 80
# canny = cv2.Canny(imgBlur3x3, cannyThreshold, cannyParam2)
# cv2.imshow('Canny', canny)
# cv2.waitKey()

# cannyThreshold = 5
# cannyParam2 = 20
# canny = cv2.Canny(imgBlur3x3, cannyThreshold, cannyParam2)
# cv2.imshow('Canny', canny)
# cv2.waitKey()

plt.subplot(nrows, ncols,8),plt.imshow(canny,cmap = 'gray')
plt.title('Canny edge detection'), plt.xticks([]), plt.yticks([])
plt.show()

# 13) The above process with an arbitrary image
imgOrig2 = cv2.imread('campus-nice.jpg',)

# Convert to grayscale
gray_image2 = cv2.cvtColor(imgOrig2, cv2.COLOR_BGR2GRAY)

# Plotting images with Python and Matplotlib
img2 = cv2.cvtColor(imgOrig2, cv2.COLOR_BGR2RGB)

plt.subplot(nrows,ncols,1),plt.imshow(img2,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(gray_image2,cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

# Applying blur with a 3x3
KernelSizeWidth = 3
KernelSizeHeight = 3
imgBlur3x3_2 = cv2.GaussianBlur(gray_image2,(KernelSizeWidth,KernelSizeHeight),0)

# Plot the 3x3 blur
plt.subplot(nrows, ncols,3),plt.imshow(imgBlur3x3_2,cmap = 'gray')
plt.title('Blur 3x3'), plt.xticks([]), plt.yticks([])

# Sobel edge detection
sobelHorizontal2 = cv2.Sobel(imgBlur3x3_2,cv2.CV_64F,1,0,ksize=5)
sobelVertical2 = cv2.Sobel(imgBlur3x3_2,cv2.CV_64F,0,1,ksize=5)

# Plot Sobel processed images
plt.subplot(nrows, ncols,4),plt.imshow(sobelHorizontal2,cmap = 'gray')
plt.title('Sobel x - Blur 3x3'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,5),plt.imshow(sobelVertical2,cmap = 'gray')
plt.title('Sobel y - Blur 3x3'), plt.xticks([]), plt.yticks([])

# Adding the two Sobel processed images
sobelSum2 = sobelHorizontal2 + sobelVertical2

plt.subplot(nrows, ncols,6),plt.imshow(sobelSum2,cmap = 'gray')
plt.title('Sobel sum'), plt.xticks([]), plt.yticks([])

# Apply Canny edge detection
cannyThreshold = 20
cannyParam2 = 60
canny2 = cv2.Canny(imgBlur3x3_2, cannyThreshold, cannyParam2)

plt.subplot(nrows, ncols,7),plt.imshow(canny2,cmap = 'gray')
plt.title('Canny edge detection'), plt.xticks([]), plt.yticks([])
plt.show()

# Advanced exercise 1)
# Got this from StackOverflow:
# https://stackoverflow.com/questions/26445153/iterations-through-pixels-in-an-image-are-terribly-slow-with-python-opencv
nrows = 1
ncols = 2

# Threshold between 0 and 255
sobelThreshold = 32

# Looks like 'sobelSum' is type 'float64', we are going to do some casts later
print("Image type: ", sobelSum.dtype)
print("Image size(amount of pixels): ", sobelSum.size)

height, width = sobelSum.shape

for i in range(0, height):
     for j in range(0, width):
        #  Cast 'pixel' into a unsigned integer (0-255), so comparing against the threshold is more understandable
         pixel = np.uint8(sobelSum[i, j])

         if pixel < sobelThreshold:
             sobelSum[i, j] = 0
         else:
             sobelSum[i, j] = 255

# cv2.imshow('Sobel sum edge detection', sobelSum)
# cv2.waitKey()

plt.subplot(nrows, ncols,1),plt.imshow(sobelSum,cmap = 'gray')
plt.title('Sobel sum edge detection'), plt.xticks([]), plt.yticks([])
plt.show()