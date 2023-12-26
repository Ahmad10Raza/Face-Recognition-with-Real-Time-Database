import cv2
import face_recognition
import pickle
import os




# importing student images

folderPath="Images"
pathList=os.listdir(folderPath)
imgList=[]
studentIDs=[]
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    #print(path)
    studentIDs.append(os.path.splitext(path)[0])
#print(studentIDs)




# function to encode images 
def findEncodings(imgList):
    encodeList=[]
    for img in imgList:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started")
encodeListKnown=findEncodings(imgList)
encodeListKnownWithIDs=[encodeListKnown,studentIDs]
print(encodeListKnownWithIDs)
print("Encoding Completed")


file=open("EncodeFile.pickle","wb")
pickle.dump(encodeListKnownWithIDs,file)
print("Encoding File Created")