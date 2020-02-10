import cv2
import csv
import cv2 as cv
import numpy as np


width2 = 320
height2 = 240

cap = cv2.VideoCapture('IMG_0192.mov')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

f = open('right.csv' , 'w',newline='')
wr = csv.writer(f)
wr.writerow(['Xposition' , 'Yposition'])


def detect(gray,frame):
    eyes = eyeCascade.detectMultiScale(gray,1.1,3)
    #for(ex,ey,ew,eh) in eyes:
        #cv2.rectangle(frame,(ex+20,ey+20) , (ex+ew-30, ey+eh-30) , (0,255,0) ,2)
        #cv2.rectangle(frame,(ex+int(ew/2) , ey+int(eh/2)), (ex+int(ew/2),ey+int(eh/2)) , (0,0,255),2)
    return frame

def draw_center(frame,minx,maxx,miny,maxy):
    gravity_x = int((minx + maxx) / 2)
    gravity_y = int((miny + maxy) / 2)
    cv2.rectangle(ROI_img2, (gravity_x, gravity_y), (gravity_x , gravity_y ), (255, 0, 0), 3)

    return frame

while(cap.isOpened()):

    ret2, frame2 = cap.read()
    frame_resize2 = cv2.resize(frame2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ROI_img2 = frame_resize2[165:270, 550:780]
    #cv2.imshow('ROI area' ,  ROI_img2)


    # <-------------gray화------------------>
    gray2 = cv2.cvtColor(ROI_img2, cv2.COLOR_BGR2GRAY)

    # <-------------눈감지------------------>
    detect_frame2 = detect(gray2,ROI_img2)


    # <-------------노이즈제거------------------>
    gray_roi2 = cv2.GaussianBlur(gray2, (7, 7), 0)



    # <-------------이진화------------------>
    threshold2 = cv2.adaptiveThreshold(gray_roi2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 23, 20)
    ret2, pupil2 = cv2.threshold(gray_roi2, 30, 255, cv2.THRESH_BINARY)
    cv2.imshow('pupil', pupil2)

    # <--------------------눈 왼쪽아래죄표 잡기-------------------------------------->
    canny_img2 = cv2.Canny(threshold2, 10, 20)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed2 = cv2.morphologyEx(canny_img2, cv2.MORPH_CLOSE, kernel2)
    contours2, _ = cv2.findContours(closed2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total2 =0
    #contours_img = cv2.drawContours(ROI_img2,contours,-1,(0,255,0),2)
    contours_xy2 = np.array(contours2)
    contours_xy2.shape

    # <--------------------x의 min과 max찾기-------------------------------------->
    x_min2, x_max2 = 0, 0
    value2 = list()
    for i in range(len(contours_xy2)):
        for j in range(len(contours_xy2[i])):
            value2.append(contours_xy2[i][j][0][0])
            x_min2 = min(value2)
            x_max2 = max(value2)

    # <--------------------y의 min과 max찾기-------------------------------------->
    y_min2, y_max2 = 0, 0
    value2 = list()
    for i in range(len(contours_xy2)):
        for j in range(len(contours_xy2[i])):
            value2.append(contours_xy2[i][j][0][1])
            y_min2 = min(value2)
            y_max2 = max(value2)

    x2 = x_min2
    y2 = y_min2
    w2 = x_max2 - x_min2
    h2 = y_max2 - y_min2

#<--------------------동공 왼쪽아래죄표 잡기-------------------------------------->
    pupil_canny_img2 = cv2.Canny(pupil2, 10, 20)
    pupil_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    pupil_closed2 = cv2.morphologyEx(pupil_canny_img2, cv2.MORPH_CLOSE, pupil_kernel2)
    pupil_contours2, _ = cv2.findContours(pupil_closed2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pupil_total2 = 0
    pupil_contours_img2 = cv2.drawContours(ROI_img2, pupil_contours2, -1, (0, 0, 255), 2)
    pupil_contours_xy2 = np.array(pupil_contours2)
    pupil_contours_xy2.shape

    # <--------------------pupil_x의 min과 max찾기-------------------------------------->
    pupil_x_min2, pupil_x_max2 = 0, 0
    pupil_value2 = list()
    for i in range(len(pupil_contours_xy2)):
        if (pupil_contours_xy2.shape[0] == 1):
            for j in range(len(pupil_contours_xy2[i])):
                pupil_value2.append(pupil_contours_xy2[i][j][0][0])
                pupil_x_min2 = min(pupil_value2)
                pupil_x_max2 = max(pupil_value2)
        else:
            i = 1
            for j in range(len(pupil_contours_xy2[i])):
                pupil_value2.append(pupil_contours_xy2[i][j][0][0])
                pupil_x_min2 = min(pupil_value2)
                pupil_x_max2 = max(pupil_value2)


# <--------------------pupil_y의 min과 max찾기-------------------------------------->
    pupil_y_min2, pupil_y_max2 = 0, 0
    pupil_value2 = list()
    for i in range(len(pupil_contours_xy2)):
        if (pupil_contours_xy2.shape[0] == 1):
            for j in range(len(pupil_contours_xy2[i])):
                if (pupil_contours_xy2.shape[0] == 1):
                    pupil_value2.append(pupil_contours_xy2[i][j][0][1])
                    pupil_y_min2 = min(pupil_value2)
                    pupil_y_max2 = max(pupil_value2)
        else:
            i = 1
            for j in range(len(pupil_contours_xy2[i])):
                if (pupil_contours_xy2.shape[0] == 1):
                    pupil_value2.append(pupil_contours_xy2[i][j][0][1])
                    pupil_y_min2 = min(pupil_value2)
                    pupil_y_max2 = max(pupil_value2)

    pupil_x2 = pupil_x_min2
    pupil_y2 = pupil_y_min2
    pupil_w2 = pupil_x_max2 - pupil_x_min2
    pupil_h2 = pupil_y_max2 - pupil_y_min2
    pupil_center_img2 = draw_center(ROI_img2, pupil_x_min2, pupil_x_max2, pupil_y_min2, pupil_y_max2)

    # <--------------------눈 크기에 맞춰 화면조절 후 출력-------------------------------------->
    img_trim2 = pupil_center_img2[y2:y2 + h2, x2:x2 + w2]
    cv2.imwrite('org_trim2.jpg', img_trim2)
    org_image2 = cv2.imread('org_trim2.jpg')
    height2, width2 = org_image2.shape[:2]
    org_image_x42 = cv2.resize(org_image2, (4 * width2, 4 * height2), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("x4 org_image2", org_image_x42)


    # <--------------------동공을 찾았을경우 엑셀파일에 쓰기-------------------------------------->
    if(((pupil_x_max2- pupil_x_min2) > 10) and ((pupil_y_max2-pupil_y_min2) >10)):
        print("X좌표 : ", int(pupil_x_max2-(pupil_x_max2 - pupil_x_min2) / 2), "Y 좌표 : ", int(pupil_y_max2 - (pupil_y_min2 - pupil_y_max2)/2))
        wr.writerow([int(pupil_x_max2 - (pupil_x_max2 - pupil_x_min2) / 2), int(pupil_y_max2 - (pupil_y_max2 - pupil_y_min2) / 2)])
    else:
        print("동공을 찾을 수 없습니다.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        f.close()
        break


f.close()
cap.release()
cv2.destroyAllWindows()