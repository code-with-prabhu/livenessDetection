from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time


classID = 1  # it requires for yolo object detection 0=flase, 1=true
output_folder = 'Dataset/mydata'
save = True
blurThreshold = 35
confidence = 0.8
offsetPercentageW = 10
camWidth, camHeight = 640, 480
floatingPoint = 6
offsetPercentageH = 20


def main():

    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    while True:

        success, img = cap.read()
        imgOut = img.copy()
        # Detect faces in the image
        # img: Updated image
        # bboxs: List of bounding boxes around detected faces
        img, bboxs = detector.findFaces(img, draw=False)
        listBlur = []  # state if the image is blury or not
        listInfo = []  # state normalize value of the images

        if bboxs:
            for bbox in bboxs:
                x, y, w, h = bbox["bbox"]
                score = bbox["score"][0]
                #only selecting those images having more than 80% of confidence
                if score > confidence:
                    # adding an offset to increase the square
                    offsetW = (offsetPercentageW / 100) * w
                    x = int(x - offsetW)
                    w = int(w + offsetW * 2)
                    offsetH = (offsetPercentageH / 100) * h
                    y = int(y - offsetH * 3)
                    h = int(h + offsetH * 3.5)

                    # if the face not in camera
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if w < 0: w = 0
                    if h < 0: h = 0

                    # blur handling
                    imgFace = img[y:y + h, x:x + w]
                    cv2.imshow("Face", imgFace)
                    blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                    if blurValue > blurThreshold:
                        listBlur.append(True)
                    else:
                        listBlur.append(False)

                    #normalizing the values
                    ih, iw, _ = img.shape
                    xc, yc = x + w / 2, y + h / 2

                    xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                    wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)


                    if xcn > 1: xcn = 1
                    if ycn > 1: ycn = 1
                    if wn > 1: wn = 1
                    if hn > 1: hn = 1

                    listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n") #it is storing the details of that image

                    cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                       scale=1, thickness=2)

                    if save:
                        if all(listBlur) and listBlur != []:
                            #image save in the dataset/mydata file
                            timeNow = time()
                            timeNow = str(timeNow).split('.')
                            timeNow = timeNow[0] + timeNow[1]
                            cv2.imwrite(f"{output_folder}/{timeNow}.jpg", img)
                            #associating a label txt file to the image for better understanding about the picture
                            for info in listInfo:
                                f = open(f"{output_folder}/{timeNow}.txt", 'a')
                                f.write(info)
                                f.close()

        # Display the image in a window named 'Image'
        cv2.imshow("Image", imgOut)
        # Wait for 1 millisecond, and keep the window open
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
