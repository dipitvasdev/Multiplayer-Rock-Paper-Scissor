import numpy as np 
import cv2 
import os
from model_rps import predict_on_img
cap=cv2.VideoCapture(0)
class_to_move={1:"Rock",0:"Paper",2:"Scissor"}
check = False 
number_of_runs = 0
bg1 = None
bg2 = None 
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    cv2.putText(frame,"WELCOME TO ROK PAPR SCISSR",(170,30),cv2.FONT_HERSHEY_PLAIN,1,(200,100,100),2,cv2.LINE_AA)
    rectangle_1=cv2.rectangle(frame,(50,60),(270,280),(0,0,0),3)
    rectangle_2=cv2.rectangle(frame,(350,60),(570,280),(255,255,255),3)
    
    image_section_1=frame[63:267,53:267]
    image_section_2=frame[63:277,353:567]
    image_section_1_grey = cv2.cvtColor(image_section_1,cv2.COLOR_BGR2GRAY)
    image_section_2_grey = cv2.cvtColor(image_section_2,cv2.COLOR_BGR2GRAY)
    image_section_1_grey = cv2.GaussianBlur(image_section_1_grey, (7,7), 50)
    image_section_2_grey = cv2.GaussianBlur(image_section_2_grey, (7,7) ,50)
    if number_of_runs<100:
        if (bg1 is None) and (bg2 is None):
            bg1 = image_section_1_grey.copy().astype("float")
            bg2 = image_section_2_grey.copy().astype("float")
            continue
        if number_of_runs == 99 :
            print("Calibrated...")
        elif number_of_runs < 99:
            print("Calibrating Background...")
        cv2.accumulateWeighted(image_section_1_grey,bg1, 0.5)
        cv2.accumulateWeighted(image_section_2_grey,bg2, 0.5)
    else: 
        diff1 = cv2.absdiff(bg1.astype("uint8"), image_section_1_grey)
        diff2 = cv2.absdiff(bg2.astype("uint8"),image_section_2_grey)
        threshold1 = cv2.threshold(diff1,30 , 255 , cv2.THRESH_BINARY)[1]
        threshold2 = cv2.threshold(diff2,30 , 255 , cv2.THRESH_BINARY)[1]
        cv2.imshow("T1",threshold1)
        cv2.imshow("T2",threshold2)
        key_pressed=cv2.waitKey(1) & 0xFF 
        if key_pressed==ord('s') or check == True:
            check = True
            i=0 
            while os.path.exists("online/"+"%s.jpg"%i):
                i+=1
            FILE_1="./online/"+str(i)+".jpg"
            FILE_2="./online/"+str(i+1)+".jpg"
            cv2.imwrite(FILE_1,threshold1)
            cv2.imwrite(FILE_2,threshold2)
            prediction_1,value_1= predict_on_img(FILE_1)
            prediction_2,value_2= predict_on_img(FILE_2)
            cv2.putText(image_section_1,class_to_move[value_1] , (int(image_section_1.shape[0] * 0.3) , int(image_section_1.shape[1] * 0.9 )) ,cv2.FONT_HERSHEY_SIMPLEX,1,(71,55,255),3,cv2.LINE_AA)
            cv2.putText(image_section_2,class_to_move[value_2] , (int(image_section_2.shape[0] * 0.3) , int(image_section_2.shape[1] * 0.9 )) ,cv2.FONT_HERSHEY_SIMPLEX,1,(71,55,255),3,cv2.LINE_AA)
            Left =class_to_move[value_1]
            Right = class_to_move[value_2]
            result = None
            if Left == "Rock" and Right == "Scissor" : 
                result = "Left Player Wins !!" 
            elif Left == "Rock" and Right == "Paper" :
                result = "Right Player Wins !!"
            elif Left == "Paper"  and Right == "Rock" : 
                result = "Left Player Wins !!"
            elif Left == "Paper" and Right == "Scissor" :
                result = "Right Player Wins !!"
            elif Left == "Scissor" and Right == "Rock" :
                result = "Right Player Wins !!"
            elif Left == "Scissor" and Right == "Paper" :
                result = "Left Player Wins !!"
            else:
                result = "Draw !!"
            
            cv2.putText(frame,result, (180,370),cv2.FONT_HERSHEY_TRIPLEX,1.2,(0,214,255),2,cv2.LINE_AA)
            key_pressed=cv2.waitKey(1) & 0xFF
            if key_pressed==ord('q'):
                break 
    number_of_runs += 1
    cv2.imshow("Frame",frame)
cap.release()
cv2.destroyAllWindows()