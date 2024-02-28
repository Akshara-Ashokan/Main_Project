import cv2
import numpy as np
import utils

path = 'dataset/img/1.jpg'
widthImg = 700
heightImg = 700
questions = 5
choices = 5
# answers = [0, 2, 0, 2, 3]
answers = ['b', 'd', 'a', 'a', 'e']

# converting answers to indexes
answers = utils.convertAnswers(answers)

cap = cv2.VideoCapture(0)
cap.set(10, 150)

while True:

    img = cv2.imread(path)

    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()
    imgGradeContours = img.copy()

    ## image preprocessing
    # coverting to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adding blur
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # detecting edges
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    # finding and drawing all contours
    contours, heirarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # finding rectangles
    rectangleContours = utils.rectContour(contours)
    # we are getting every points of the pixels. we want the corner points only
    biggestContour = utils.getCornerPoints(rectangleContours[0])
    gradePoints = utils.getCornerPoints(rectangleContours[1])  # second biggest
    # print(biggestContour)

    if biggestContour.size != 0 and gradePoints.size != 0:
        cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
        cv2.drawContours(imgBiggestContours, gradePoints, -1, (0, 0, 255), 20)

        biggestContour = utils.reorder(biggestContour)
        gradePoints = utils.reorder(gradePoints)
        # print(gradePoints)

        point1 = np.float32(biggestContour)
        point2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(point1, point2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        gradePoint1 = np.float32(gradePoints)
        gradePoint2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        gradeMatrix = cv2.getPerspectiveTransform(gradePoint1, gradePoint2)
        imgGradeWarpColored = cv2.warpPerspective(img, gradeMatrix, (325, 150))
        # cv2.imshow('Grade Image', imgGradeWarpColored)

        # now we need to get the markings inside the bubbles of omr sheet,
        # the bubbles with more pixels will bet the answer marked and the less pixels will be unmarked
        # applying the binary threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

        # box with most non zero values will be the answer of that row
        boxes = utils.splitBoxes(imgThresh)
        # cv2.imshow('boxes', boxes[0])
        # print(cv2.countNonZero(boxes[i]))

        pixelVal = np.zeros((questions, choices))
        columnCount = 0
        rowCount = 0

        # finding index vales of the correct answers
        for box in boxes:
            pixelVal[rowCount][columnCount] = cv2.countNonZero(box)
            columnCount += 1
            if columnCount == choices:
                columnCount = 0
                rowCount += 1

        # print(pixelVal)

        myIndex = []
        for i in range(0, questions):
            arr = pixelVal[i]
            myIndexVal = np.where(arr == np.amax(arr))
            # print(myIndexVal[0][0])
            myIndex.append(myIndexVal[0][0])

        # print(myIndex)

        # Grading
        grading = []
        for i in range(0, questions):
            if answers[i] == myIndex[i]:
                grading.append(1)
            else:
                grading.append(0)

        # print(grading)
        score = (sum(grading) / questions) * 100
        # print(score)

        # displaying the answers in the image
        imgResult = imgWarpColored.copy()
        imgResult = utils.showAnswers(imgResult, myIndex, grading, answers, questions, choices)

    imgBlank = np.zeros_like(img)
    # making image array
    imageArray = ([img, imgGray, imgBlur, imgCanny],
                  [imgContours, imgBiggestContours, imgWarpColored, imgThresh])
                  

    labels = [["Original", "Gray", "Blur", "Canny"],
              ["Contours", "Biggest con", "Warpped", "Threshold"]]
              

    # imported function from utils to show the images used as a stack
    imgStacked = utils.stackImages(imageArray, 0.3, labels)

    cv2.imshow('Stacked images', imgStacked)

    key = cv2.waitKey(1) & 0xFF

    if key != 0xFF:
        break

cap.release()
cv2.destroyAllWindows()
