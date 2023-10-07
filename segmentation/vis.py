import cv2
import numpy as np
from scipy import ndimage

c1 = cv2.imread("images/item24Class1.tif", cv2.IMREAD_GRAYSCALE)
c2 = cv2.imread("images/item24Class2.tif", cv2.IMREAD_GRAYSCALE)
c3 = cv2.imread("images/item24Class3.tif", cv2.IMREAD_GRAYSCALE)
c4 = cv2.imread("images/item24Class4.tif", cv2.IMREAD_GRAYSCALE)
c5 = cv2.imread("images/item24Class5.tif", cv2.IMREAD_GRAYSCALE)
# c6 = cv2.imread("images/item24Class6.tif", cv2.IMREAD_GRAYSCALE)
# c7 = cv2.imread("images/item24Class7.tif", cv2.IMREAD_GRAYSCALE)
# c8 = cv2.imread("images/item24Class8.tif", cv2.IMREAD_GRAYSCALE)
# c9 = cv2.imread("images/item24Class9.tif", cv2.IMREAD_GRAYSCALE)
# c10 = cv2.imread("images/item24Class10.tif", cv2.IMREAD_GRAYSCALE)

new_image = np.zeros((3,c1.shape[0], c1.shape[1])).astype(int)

for i in range(len(c1)):
    for j in range(len(c1[i])):
        if c1[i][j] > c2[i][j] and c1[i][j] > c3[i][j] and c1[i][j] > c4[i][j] and c1[i][j] > c5[i][j]:
                #and c1[i][j] >= c6[i][j] and c1[i][j] > c7[i][j] and c1[i][j] > c8[i][j] and c1[i][j] > c9[i][j] and c1[i][j] > c10[i][j]:
            new_image[0][i][j] = 255
            new_image[1][i][j] = 255
            new_image[2][i][j] = 255
        elif c2[i][j] > c3[i][j] and c2[i][j] > c4[i][j] and c2[i][j] > c5[i][j]:
                #and c2[i][j] > c6[i][j] and c2[i][j] > c7[i][j] and c2[i][j] > c8[i][j] and c2[i][j] > c9[i][j] and c2[i][j] > c10[i][j]:
            new_image[0][i][j] = 255
            new_image[1][i][j] = 0
            new_image[2][i][j] = 0
        elif c3[i][j] > c4[i][j] and c3[i][j] > c5[i][j]:
                #and c3[i][j] > c6[i][j] and c3[i][j] > c7[i][j] and c3[i][j] > c8[i][j] and c3[i][j] > c9[i][j] and c3[i][j] > c10[i][j]:
            new_image[0][i][j] = 0
            new_image[1][i][j] = 255
            new_image[2][i][j] = 0
        elif c4[i][j] > c5[i][j]:
                #and c4[i][j] > c6[i][j] and c4[i][j] > c7[i][j] and c4[i][j] > c8[i][j] and  c4[i][j] > c9[i][j] and c4[i][j] > c10[i][j]:
            new_image[0][i][j] = 0
            new_image[1][i][j] = 0
            new_image[2][i][j] = 255
        else:
            pass
        '''
        elif c5[i][j] > c6[i][j] and c5[i][j] > c7[i][j] and c5[i][j] > c8[i][j] and c5[i][j] > c9[i][j] and c5[i][j] > c10[i][j]:
            new_image[0][i][j] = 255
            new_image[1][i][j] = 255
            new_image[2][i][j] = 0
        elif c6[i][j] > c7[i][j] and c6[i][j] > c8[i][j] and c6[i][j] > c9[i][j] and c6[i][j] > c10[i][j]:
            new_image[0][i][j] = 255
            new_image[1][i][j] = 0
            new_image[2][i][j] = 255
        elif c7[i][j] > c8[i][j] and c7[i][j] > c9[i][j] and c7[i][j] > c10[i][j]:
            new_image[0][i][j] = 0
            new_image[1][i][j] = 255
            new_image[2][i][j] = 255
        elif c8[i][j] > c9[i][j] and c8[i][j] > c10[i][j]:
            new_image[0][i][j] = 125
            new_image[1][i][j] = 125
            new_image[2][i][j] = 255
        elif c9[i][j] > c10[i][j]:
            new_image[0][i][j] = 225
            new_image[1][i][j] = 125
            new_image[2][i][j] = 83
        '''


new_image = new_image.T / 255
new_image = cv2.flip(ndimage.rotate(new_image, 90, axes=(1, 0)),0)
#new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)

test = np.ones((200,200,3)).astype(int)
test[:,:,1] *= 255

cv2.imshow("disp", new_image)
cv2.waitKey(200000)