import cv2
import csv
import numpy as np


width = 320
height = 240

cap = cv2.VideoCapture('IMG_0192.mov')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

f = open('left.csv' , 'w',newline='')
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
    cv2.rectangle(ROI_img, (gravity_x, gravity_y), (gravity_x , gravity_y ), (255, 0, 0), 3)

    return frame


while(cap.isOpened()):
    ret, frame = cap.read()
    frame_resize = cv2.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
    ROI_img = frame_resize[160:270, 190:420]  #resize


    # <-------------gray화------------------>
    gray = cv2.cvtColor(ROI_img, cv2.COLOR_BGR2GRAY)

    # <-------------눈감지------------------>
    detect_frame = detect(gray,ROI_img)


    # <-------------노이즈제거------------------>
    gray_roi = cv2.GaussianBlur(gray, (7, 7), 0)



    # <-------------이진화------------------>

    threshold = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 23, 20)
    ret, pupil = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY)

    # <--------------------눈 왼쪽아래죄표 잡기-------------------------------------->
    canny_img = cv2.Canny(threshold, 10, 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    closed = cv2.morphologyEx(canny_img,cv2.MORPH_CLOSE,kernel)
    contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total =0
    #contours_img = cv2.drawContours(ROI_img,contours,-1,(0,255,0),2)
    contours_xy = np.array(contours)
    contours_xy.shape

    # x의 min과 max 찾기
    x_min, x_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0])
            x_min = min(value)
            x_max = max(value)

    # y의 min과 max 찾기
    y_min, y_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1])
            y_min = min(value)
            y_max = max(value)

    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

#<--------------------동공 왼쪽아래죄표 잡기-------------------------------------->
    pupil_canny_img = cv2.Canny(pupil, 10, 20)
    pupil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    pupil_closed = cv2.morphologyEx(pupil_canny_img, cv2.MORPH_CLOSE, pupil_kernel)
    pupil_contours, _ = cv2.findContours(pupil_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pupil_total = 0
    pupil_contours_img = cv2.drawContours(ROI_img, pupil_contours, -1, (0, 0,255), 2)
    pupil_contours_xy = np.array(pupil_contours)
    pupil_contours_xy.shape

    # pupil_x의 min과 max 찾기
    pupil_x_min, pupil_x_max = 0, 0
    pupil_value = list()
    for i in range(len(pupil_contours_xy)):
        if (pupil_contours_xy.shape[0] == 1):
            for j in range(len(pupil_contours_xy[i])):
                pupil_value.append(pupil_contours_xy[i][j][0][0])
                pupil_x_min = min(pupil_value)
                pupil_x_max = max(pupil_value)
        else:
            i = 1
            for j in range(len(pupil_contours_xy[i])):
                pupil_value.append(pupil_contours_xy[i][j][0][0])
                pupil_x_min = min(pupil_value)
                pupil_x_max = max(pupil_value)


    # pupil_y의 min과 max 찾기
    pupil_y_min, pupil_y_max = 0, 0
    pupil_value = list()
    for i in range(len(pupil_contours_xy)):
        if (pupil_contours_xy.shape[0] == 1):
            for j in range(len(pupil_contours_xy[i])):
                if (pupil_contours_xy.shape[0] == 1):
                    pupil_value.append(pupil_contours_xy[i][j][0][1])
                    pupil_y_min = min(pupil_value)
                    pupil_y_max = max(pupil_value)
        else:
            i = 1
            for j in range(len(pupil_contours_xy[i])):
                if (pupil_contours_xy.shape[0] == 1):
                    pupil_value.append(pupil_contours_xy[i][j][0][1])
                    pupil_y_min = min(pupil_value)
                    pupil_y_max = max(pupil_value)

    pupil_x = pupil_x_min
    pupil_y = pupil_y_min
    pupil_w = pupil_x_max - pupil_x_min
    pupil_h = pupil_y_max - pupil_y_min
    pupil_center_img = draw_center(ROI_img, pupil_x_min, pupil_x_max, pupil_y_min, pupil_y_max)

    # <--------------------눈 크기에 맞춰 화면조절 후 출력-------------------------------------->
    img_trim = pupil_center_img[y:y + h, x:x + w]
    cv2.imwrite('org_trim.jpg', img_trim)
    org_image = cv2.imread('org_trim.jpg')
    height, width = org_image.shape[:2]
    org_image_x42 = cv2.resize(org_image, (4 * width, 4 * height), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("x4 org_image2", org_image_x42)

    # <--------------------동공을 찾았을경우 엑셀파일에 쓰기-------------------------------------->
    if(((pupil_x_max- pupil_x_min) > 10) and ((pupil_y_max-pupil_y_min) >10)):
        print("X좌표 : ",int(pupil_x_max-(pupil_x_max - pupil_x_min) / 2), "Y 좌표 : " ,int(pupil_y_max - (pupil_y_max - pupil_y_min)/2))
        wr.writerow([int(pupil_x_max-(pupil_x_max - pupil_x_min) / 2),int(pupil_y_max - (pupil_y_max - pupil_y_min)/2)])

    else:
        print("동공을 찾을 수 없습니다.")


    cv2.imshow('pupil' , pupil)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        f.close()
        break

f.close()
cap.release()
cv2.destroyAllWindows()