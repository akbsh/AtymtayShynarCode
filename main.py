# first we need to download packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

#opening using cv2
RBC= cv2.imread('RBC.jpg')
#turning into gray via cv2 cvtcolor
gray = cv2.cvtColor(RBC, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray');

#in order to detect cells, we need to turn the image into blur, this way we can avoid the noise around
#i used Gaussianblur method
#the 9 x 9 is kernel, that hover over whole picture, 0 is stdv
#the more you increase the size of entered kernel the more blured it becomes
Gaussianblur = cv2.GaussianBlur(gray, (9,9), 0)
plt.imshow(Gaussianblur, cmap='gray');

#for detecting the edges of cells Cany edge algorithm can be used in this case
# the blurred image should be used to avoid the noise around the desired round circles
#the 3 number is default, it is for finding edges
#lower and upper values are set to identify the edges, we can adjust it for our picture, with best adjustment the right number of cell can be found
contour = cv2.Canny(Gaussianblur, 70, 160, 3)
plt.imshow(contour, cmap='gray')

#this can be used to have finished closed edges contour
#the more iterations are made the thicker are the contours
str_con= cv2.dilate(contour, (1,1), iterations = 3)
plt.imshow(str_con, cmap='gray')


#counting
#cv2.retr_external means counting only external edges
#cv2.chain_approx_none means considering all contours
(cnt, heirarchy) = cv2.findContours(str_con.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#here i want to contour with blue cells in original image
#convertation of bgr to rgb matplot reads in brg, original is rgb
RGB = cv2.cvtColor(RBC, cv2.COLOR_BGR2RGB)
#first is our RGB image, then cnt is contours, -1 means for all contours, (0,0,255) means blue, the last is how thick
cv2.drawContours(RGB, cnt, -1, (0, 0, 255), 1)

#final picture
plt.imshow(RGB)

#printing how much cells were counted
print('Round cells in the image: ', len(cnt))
plt.show()
