import numpy as np
import cv2

img = cv2.imread('dog.jpg')
img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
#img = img.transpose(2,0,1)
img = np.expand_dims(img, 0)
img = (img-127.5)/127.5
np.savez('dog.npz', img)
