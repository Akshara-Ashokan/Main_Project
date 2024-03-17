import cv2
import numpy as np
import utils  # utils is the python file for writing the helper functions.


def get_answers(path, answers, negative_marking, questions, choices):
    widthImg = 700
    heightImg = 700

    # converting answers to indexes. calling the function from utils file
    answers = utils.convertAnswers(answers)

    while True:
        img = cv2.imread(path)

        img = cv2.resize(img, (widthImg, heightImg))
        imgContours = img.copy()
        finalImg = img.copy()
        imgBiggestContours = img.copy()
        imgGradeContours = img.copy()

        ## image preprocessing
        # first converting the image to canny to detect the rectangles in the picture
        # were big rectangle will have the OMR bubbles and the small rectangle will be the grade marking box.
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # adding blur
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        # detecting edges
        imgCanny = cv2.Canny(imgBlur, 10, 50)

        try:
            # then we find out the rectangles in the picture and finding out the corner points.
            contours, heirarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

            # finding rectangles. calling the function from the utils file
            rectangleContours = utils.rectContour(contours)
            # we are getting every points of the pixels. we want the corner points only
            biggestContour = utils.getCornerPoints(rectangleContours[0])
            gradePoints = utils.getCornerPoints(rectangleContours[1])  # second biggest

            if biggestContour.size != 0 and gradePoints.size != 0:
                cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
                cv2.drawContours(imgBiggestContours, gradePoints, -1, (0, 0, 255), 20)

                # rearranging the the corner points correctly of both rectangles
                # (like finding out top left, top right, bottom left, bottom right corner points)
                # calling the function from the utils file
                biggestContour = utils.reorder(biggestContour)
                gradePoints = utils.reorder(gradePoints)

                # After getting the points extracting the both rectangles (OMR, Grade) from the image.
                point1 = np.float32(biggestContour)
                point2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                matrix = cv2.getPerspectiveTransform(point1, point2)
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

                gradePoint1 = np.float32(gradePoints)
                gradePoint2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
                gradeMatrix = cv2.getPerspectiveTransform(gradePoint1, gradePoint2)
                imgGradeWarpColored = cv2.warpPerspective(img, gradeMatrix, (325, 150))

                # After extracting the OMR rectangle converting that extracted image into binary image
                # (only black and white pixels will be there. 0 and 255 only)

                # applying the binary threshold
                imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
                imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

                # Then we split the image into 5 for each row to get each bubbles.
                # caling the function from utils file
                boxes = utils.splitBoxes(imgThresh, questions, choices)

                pixelVal = np.zeros((questions, choices))
                columnCount = 0
                rowCount = 0

                # for a row which bubble have the most number of 255s in it, that will be the answer marked in the image.
                # like that we find every answers that marked in the image.

                # box with most non zero values will be the answer of that row
                # finding index values of the correct answers
                for box in boxes:
                    pixelVal[rowCount][columnCount] = cv2.countNonZero(box)
                    columnCount += 1
                    if columnCount == choices:
                        columnCount = 0
                        rowCount += 1

                myIndex = []
                for i in range(0, questions):
                    arr = pixelVal[i]
                    count_greater_than_5 = sum(1 for value in arr if value > 5000)
                    # Check if the count is greater than 1
                    if count_greater_than_5 > 1 or count_greater_than_5 < 1:
                        myIndex.append(-1)
                    else:
                        myIndexVal = np.where(arr == np.amax(arr))
                        myIndex.append(myIndexVal[0][0])

                # also calculating the grade by comparing it with the answers that we have predefined.
                # Grading
                grading = []
                for i in range(0, questions):
                    if answers[i] == myIndex[i]:
                        grading.append(1)
                    else:
                        grading.append(0)

                temp_grading = grading.copy()

                if negative_marking:
                    for i in range(len(myIndex)):
                        if myIndex[i] != -1 and temp_grading[i] == 0:
                            temp_grading[i] -= 1 / 3

                score = (sum(temp_grading) / questions) * 100
                score = round(score, 1)
                score = max(score, 0.0)
                if score == 100.0:
                    score = 100

                # now we have to mark the right and wrong in the OMR sheet that we extracted from the image.
                # for that also we use cv2 to draw green if the answer is correct, red if the answer marked is wrong.
                # if its wrong marking the correct answer with a small green dot.
                # displaying the answers in the image
                imgResult = imgWarpColored.copy()
                # calling the function from utils file
                imgResult = utils.showAnswers(imgResult, myIndex, grading, answers, questions, choices)

                # now we need to make these markings inside the original image. so taking the markings
                imgRawDrawing = np.zeros_like(imgWarpColored)
                imgRawDrawing = utils.showAnswers(imgRawDrawing, myIndex, grading, answers, questions, choices)

                # after that we took the drawings(markings) only. we need to show that in the real image.
                # the real image may not be in correct shape. it may be little tilted.
                # so we make inverse that drawings to be correctly fit in the real image.
                invMatrix = cv2.getPerspectiveTransform(point2, point1)
                imgInvWrap = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

                # after inversing the drawings we add that to the real image and we can get the markings will be correctly aligned to each bubbles.
                # making the final image
                finalImg = cv2.addWeighted(finalImg, 0.8, imgInvWrap, 1, 0)

                # also we found out the grade before. and we add that grade into the small rectangle.
                # adding the Grade in the box
                imgRawGrade = np.zeros_like(imgGradeWarpColored)
                cv2.putText(imgRawGrade, str(score) + '%', (20, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 256, 256), 3)

                gradeInvMatrix = cv2.getPerspectiveTransform(gradePoint2, gradePoint1)
                imgInvGradeWarp = cv2.warpPerspective(imgRawGrade, gradeInvMatrix, (widthImg, heightImg))

                # The final Image
                finalImg = cv2.addWeighted(finalImg, 1, imgInvGradeWarp, 1, 0)
                # cv2.imshow('final image', finalImg)

                return finalImg, score

            imgBlank = np.zeros_like(img)
            # making image array
            imageArray = ([img, imgGray, imgBlur, imgCanny],
                          [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
                          [imgResult, imgRawDrawing, imgInvWrap, finalImg])

        except:
            imgBlank = np.zeros_like(img)
            # making image array
            imageArray = ([img, imgGray, imgBlur, imgCanny],
                          [imgBlank, imgBlank, imgBlank, imgBlank],
                          [imgBlank, imgBlank, imgBlank, imgBlank])

        labels = [["Original", "Gray", "Blur", "Canny"],
                  ["Contours", "Biggest con", "Warpped", "Threshold"],
                  ["Result", "Raw drawing", "Inv Warpped", "Final"]]

        # imported function from utils to show the images used as a stack
        imgStacked = utils.stackImages(imageArray, 0.3, labels)

        # cv2.imshow('Stacked images', imgStacked)
        # cv2.imshow('final image', finalImg)

        key = cv2.waitKey(1) & 0xFF

        # press s key to save the image
        if key != 0xFF:
            break

    cv2.destroyAllWindows()
