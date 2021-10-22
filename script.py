import cv2

img=cv2.imread('media\galaxy.jpg',0)

print(type(img))
print(img)
print(img.shape[1])
print(img.shape[0])
print(img.ndim)
print('hello cv2')

resized_image=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/3)))
cv2.imshow('galaxy',resized_image)
cv2.imwrite("Galaxy_resized.jpg",resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()