import os
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation


capture=cv2.VideoCapture(2)
capture.set(3,640)
capture.set(4,480)
capture.set(5,30)

img_list=os.listdir("img_dir")
list_img=[]
for image in img_list:
    img_1=cv2.imread(f"img_dir\\{image}")
    list_img.append(img_1)

segment=SelfiSegmentation(1)

while True:
    _,img=capture.read()
    img_2=segment.removeBG(img,list_img[1], cutThreshold=0.65)
    image_out=cvzone.stackImages([img,img_2],2,1)


    cv2.imshow("Image",image_out)
 
    cv2.waitKey(1)  