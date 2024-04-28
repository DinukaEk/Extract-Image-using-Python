import cv2
import numpy as np


img1 = cv2.imread('t-shirt-army-green.jpg')
img2 = cv2.imread('t-shirt-sea-green.jpg')
print(img1.shape)
print(img2.shape)
img1_img2 = np.hstack((img1[:, :455], img2[:, 455:]))



cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("img1_img2", img1_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()