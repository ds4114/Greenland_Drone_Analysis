import cv2
import numpy as np


def disp_image(item_number):
    b = cv2.imread(f"images_test/initial_images/item{item_number}Class4.tif", cv2.IMREAD_GRAYSCALE).astype(np.float32) # b
    r = cv2.imread(f"images_test/initial_images/item{item_number}Class3.tif", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    g = cv2.imread(f"images_test/initial_images/item{item_number}Class5.tif", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #b = cv2.imread(f"images_test/initial_images/item{item_number}Class4.tif", cv2.IMREAD_GRAYSCALE).astype(np.float32)

    im = np.rot90(np.concatenate([r[np.newaxis, :, :],g[np.newaxis, :, :],b[np.newaxis, :, :]]).T/255)
    #im = np.fliplr(im)
    im = np.flipud(im)

    cv2.imshow("Image",im)
    cv2.waitKey()

disp_image(25)