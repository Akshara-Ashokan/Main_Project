import cv2
import numpy as np

def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def rectContour(contours):
    rectangleContour = []
    for i in contours:
        area = cv2.contourArea(i)
        # print('Area: ', area)
        if area > 50:
            perimeter = cv2.arcLength(i, True)
            # approximation of corner points of each rectangle
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
            # print('corner points', len(approx))
            if len(approx) == 4:
                rectangleContour.append(i)

    rectangleContour = sorted(rectangleContour, key=cv2.contourArea, reverse=True)
    return rectangleContour

def getCornerPoints(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
    return approx

# reordring the points because if we dont do it then the image will not get properly
def reorder(myPoints):
    # print(myPoints)
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    diff = np.diff(myPoints, axis=1)
    # print(add)
    # print(diff)
    # adding the points so that we can under stand the points order.
    # the minimum numbered point will be the top right and the maximum will be the bottom right
    myPointsNew[0] = myPoints[np.argmin(add)] # top left
    myPointsNew[3] = myPoints[np.argmax(add)] # bottom right
    myPointsNew[1] = myPoints[np.argmin(diff)] # bottom left
    myPointsNew[2] = myPoints[np.argmax(diff)] # top right
    # print(myPointsNew)

    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img, 5)

    boxes = []
    for r in rows:
        cols =  np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes

def showAnswers(img, myIndex, grading, answers, questions, choices):
    # print(myIndex)
    # print(answers)
    sectionWidth = int(img.shape[1]/questions)
    sectionHeight = int(img.shape[1]/choices)

    for i in range(0, questions):
        myAns = myIndex[i]

        cX = (myAns * sectionWidth) + sectionWidth // 2
        cY = (i * sectionHeight) + sectionHeight // 2

        if grading[i]:
            cv2.circle(img, (cX, cY), 50, (0, 255, 0), cv2.FILLED)
        else:
            cv2.circle(img, (cX, cY), 50, (0, 0, 255), cv2.FILLED)

            cX = (answers[i] * sectionWidth) + sectionWidth // 2
            cv2.circle(img, (cX, cY), 20, (0, 255, 0), cv2.FILLED)

    return img

def convertAnswers(answers):
    result = []

    for letter in answers:
        value = ord(letter) - ord('a')
        result.append(value)

    return result
