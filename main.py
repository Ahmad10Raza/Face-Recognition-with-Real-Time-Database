import os
import pickle
import numpy as np
import cv2
# import face_recognition
# import cvzone
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db
# from firebase_admin import storage
# import numpy as np
from datetime import datetime




# webcap capture
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


# backgroung image for webcam
imgBackground=cv2.imread("Resources/background.png")

# importing mode images iunto list
folderModePath="Resources/Modes"
modePathList=os.listdir(folderModePath)
imgModeList=[]
#print(modePathList)
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
    
print(len(imgModeList))










while True:
    success,img=cap.read()
    
    imgBackground[162:162+480,55:55+640]=img
    imgBackground[44:44+633,808:808+414]=imgModeList[0]
    #cv2.imshow("Image",img)
    cv2.imshow("Background",imgBackground)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break












cap.release()
cv2.destroyAllWindows()