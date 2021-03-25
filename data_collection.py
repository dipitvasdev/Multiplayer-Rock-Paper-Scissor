import cv2
import numpy as np 
import os 
dataset_path='./data/' 
skip=0 
move_name=input("Logging data for? Rock/Paper/Scissor:")
move_name=move_name.lower()
check=False
bg = None 
number_of_runs = 0
cap=cv2.VideoCapture(0) 
while True: 
    ret,frame=cap.read()
    rectangle=cv2.rectangle(frame,(10,10),(260,260),(0,97,213),3)
    cv2.imshow("Frame",frame)
    hand_section=frame[13:257,13:257]
    hand_section_grey = cv2.cvtColor(hand_section, cv2.COLOR_BGR2GRAY) 
    hand_section_grey = cv2.GaussianBlur(hand_section_grey, (7,7), 50)
    if number_of_runs < 100:
        if bg is None : 
            bg = hand_section_grey.copy().astype("float")
            continue
        cv2.imshow("bg",bg)
        if number_of_runs == 99 :
            print("Calibrated...")
        elif number_of_runs < 99:
            print("Calibrating Background...")
        cv2.accumulateWeighted(hand_section_grey,bg, 0.5)
    else:
        diff = cv2.absdiff(bg.astype("uint8"), hand_section_grey)
        cv2.imshow("diff",diff)
        threshold = cv2.threshold(diff,30 , 255 , cv2.THRESH_BINARY)[1]
        cv2.imshow("Thresholded", threshold)
        key_pressed=cv2.waitKey(1) & 0xFF
        if(key_pressed==ord('s')):
            check=True  
        if check:
            skip+=1
            if skip%5==0:
                PATH_TO_SAVE= dataset_path + move_name + "/"
                i=0 
                while os.path.exists(PATH_TO_SAVE+move_name+"%s.jpg"%i):
                    i+=1 
                FILENAME=PATH_TO_SAVE+move_name+str(i)+".jpg"
                cv2.imwrite(FILENAME,threshold)
                print("Snapshot at "+FILENAME)
        key_pressed=cv2.waitKey(1) & 0xFF 
        if key_pressed==ord('q'):
            break 
    number_of_runs += 1
print("DATA LOGGED!")
cap.release()
cv2.destroyAllWindows()