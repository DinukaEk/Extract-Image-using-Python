import cv2


img1 = cv2.imread('Man without tshirt 2.jpg')
img2 = cv2.imread('t-shirt-sea-green.jpg')

img_2_shape = img2.shape
roi = img1[0:img_2_shape[0], 0:img_2_shape[1]]
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)

# black-out the area of t-shirt in ROI
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
print(img1.shape, mask.shape)

# Take only region of t-shirt from t-shirt image.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

# Put t-shirt in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
img1[0:img_2_shape[0], 0:img_2_shape[1]] = dst


#Create resizable windows for our display images
cv2.namedWindow('img1_bg', cv2.WINDOW_NORMAL)
cv2.namedWindow('img2_fg', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('maskinv', cv2.WINDOW_NORMAL)
cv2.namedWindow('res', cv2.WINDOW_NORMAL)
cv2.namedWindow('img2gray', cv2.WINDOW_NORMAL)
cv2.namedWindow('nw', cv2.WINDOW_NORMAL)
cv2.imshow('mask', mask)
cv2.imshow('maskinv', mask_inv)
cv2.imshow('img1_bg', img1_bg)
cv2.imshow('img2_fg', img2_fg)
cv2.imshow('res', img1)
cv2.imshow('img2gray', img2gray)
cv2.imshow('nw', img_2_shape)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


