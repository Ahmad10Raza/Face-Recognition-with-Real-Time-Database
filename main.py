import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db
# from firebase_admin import storage
# import numpy as np
from datetime import datetime




# webcap capture
cap=cv2.VideoCapture(1)
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
    
#print(len(imgModeList))



# load encoding file
print("Loading Encoding File")
file=open("EncodeFile.pickle","rb")
encodeListKnownWithIDs=pickle.load(file)
file.close()
encodeListKnown,studentIDs=encodeListKnownWithIDs
#print(encodeListKnown,studentIDs)
print("Encoding File Loaded")





while True:
    success,img=cap.read()
    
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    
    imgBackground[162:162+480,55:55+640]=img
    imgBackground[44:44+633,808:808+414]=imgModeList[3]
    
    
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches", matches)
        # print("faceDis", faceDis)
    
        matchIndex = np.argmin(faceDis)
        print("Match Index", matchIndex)
    
        if matches[matchIndex]:
            # print("Tujhe PahChan Liya!")
            # print(studentIDs[matchIndex])
            
            # adding rectangle around face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox=55+x1, 162+y1, x2-x1, y2-y1
            # cv2.rectangle(imgBackground, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 255), 2)    
            imgBackground=cvzone.cornerRect(imgBackground,bbox,20,rt=0)    
    
    
    
    
    
    
    
    
    
    
    
    #cv2.imshow("Image",img)
    cv2.imshow("Background",imgBackground)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break












cap.release()
cv2.destroyAllWindows()