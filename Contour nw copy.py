import cv2 as cv
import  cvzone

img1 = cv.imread('Man without tshirt 2.jpg')
img2 = cv.imread('t-shirt-sea-green.jpg')

img_2_shape = img2.shape
roi = img1[0:img_2_shape[0], 0:img_2_shape[1]]
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

ret, mask = cv.threshold(img2gray, 127, 255, cv.THRESH_BINARY)

mask_inv = cv.bitwise_not(mask)

# black-out the area of t-shirt in ROI
img1_bg = cv.bitwise_and(roi, roi, mask=mask)
print(img1.shape, mask.shape)

# Take only region of t-shirt from t-shirt image.
img2_fg = cv.bitwise_and(img2, img2, mask=mask_inv)

x, y = 150, 200
img1_height, img1_width, _ = img1.shape

x_pixels = int(x * img1_width)
y_pixels = int(y * img1_height)

# Put t-shirt in ROI and modify the main image
dst = cv.add(img1_bg, img2_fg)
img1[0:img_2_shape[0], 0:img_2_shape[1]] = dst


#Create resizable windows for our display images
#cv.namedWindow('img1_bg', cv.WINDOW_NORMAL)
#cv.namedWindow('img2_fg', cv.WINDOW_NORMAL)
#cv.namedWindow('mask', cv.WINDOW_NORMAL)
#cv.namedWindow('maskinv', cv.WINDOW_NORMAL)
cv.namedWindow('res', cv.WINDOW_NORMAL)
#cv.namedWindow('img2gray', cv.WINDOW_NORMAL)
#cv.imshow('mask', mask)
#cv.imshow('maskinv', mask_inv)
#cv.imshow('img1_bg', img1_bg)
#cv.imshow('img2_fg', img2_fg)
cv.imshow('res', img1)
#cv.imshow('img2gray', img2gray)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()