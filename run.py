## Import Libraries 
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import os
from model_rps import predict_on_img
## Process Videos
class VideoProcessor:
    def __init__(self):
        self.number_of_runs = 0
        self.bg1 = None 
        self.bg2 = None
        self.class_to_move = {1:"Rock",0:"Paper",2:"Scissor"}
        self.prediction1 = None 
        self.prediction2 = None
        self.value1 = None
        self.value2 = None
        self.left = None
        self.right = None
    def recv(self, frame):
        self.number_of_runs += 1
        
        img = frame.to_ndarray(format = "bgr24")
        flip = cv2.flip(img,1)
        ## Draw playing area 
        
        cv2.rectangle(flip,(30,40),(280,290) ,(0,0,255),3)
        cv2.rectangle(flip,(370,40),(620,290),(0,255,0),3)
        
        ## Extract image from playing area 

        image_section_1 = flip[43:287, 33:277]
        image_section_2 = flip[43:287, 373:617]

        ## Convert to grayscale 

        image_section_1_grey = cv2.cvtColor(image_section_1,cv2.COLOR_BGR2GRAY)
        image_section_2_grey = cv2.cvtColor(image_section_2,cv2.COLOR_BGR2GRAY)

        ## Apply GaussianBlur

        image_section_1_grey = cv2.GaussianBlur(image_section_1_grey, (7,7), 50)
        image_section_2_grey = cv2.GaussianBlur(image_section_2_grey, (7,7) ,50)

        ## Get some time for the background to settle and get background matrices
        if self.number_of_runs < 200:

            ## Get initial background
            if(self.bg1 is None and self.bg2 is None):

                self.bg1 = image_section_1_grey.copy().astype("float")
                self.bg2 = image_section_2_grey.copy().astype("float")
            ## Running Average of 200 images, that contrain background 
            
            cv2.accumulateWeighted(image_section_1_grey,self.bg1, 0.5)
            cv2.accumulateWeighted(image_section_2_grey,self.bg2, 0.5)

            ## Ask user to wait

            cv2.putText(image_section_1,"PLEASE WAIT..." , (int(image_section_1.shape[0] * 0.05) , int(image_section_1.shape[1] * 0.9 )) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(71,55,255),3,cv2.LINE_AA)
            cv2.putText(image_section_2,"PLEASE WAIT..." , (int(image_section_2.shape[0] * 0.05) , int(image_section_2.shape[1] * 0.9 )) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(71,55,255),3,cv2.LINE_AA)
        else:
            ## Print that the background is calibrated for logging purpose

            if(self.number_of_runs == 200):
                print("Calibrated Background:-")
            
            ## Calculate difference between current value and running background average
            
            diff1 = cv2.absdiff(self.bg1.astype("uint8"), image_section_1_grey)
            diff2 = cv2.absdiff(self.bg2.astype("uint8"), image_section_2_grey)

            ##  Apple Binary Thresholding 

            threshold1 = cv2.threshold(diff1,30 , 255 , cv2.THRESH_BINARY)[1]
            threshold2 = cv2.threshold(diff2,30 , 255 , cv2.THRESH_BINARY)[1]

            ## Only if hand is detected, proceed for both blocks

            if(np.count_nonzero(threshold1) > 2500):
                
                ## Save the image for future training


                FILE_1="area1.jpg"
                cv2.imwrite(FILE_1, threshold1)

                ## Get Prediction Value from the model
                 
                self.prediction1,self.value1 = predict_on_img(FILE_1)
                self.left = self.class_to_move[self.value1]
                ## Print the result
                
                cv2.putText(image_section_1,self.class_to_move[self.value1] , (int(image_section_1.shape[0] * 0.3) , int(image_section_1.shape[1] * 0.9 )) ,cv2.FONT_HERSHEY_SIMPLEX,1,(71,55,255),3,cv2.LINE_AA)

            else:
                ## If not detected the hand, print message
                self.prediction1 = None 
                self.value1 = None
                self.left = None
                cv2.putText(image_section_1,"Not detected!" , (int(image_section_1.shape[0] * 0.05) , int(image_section_1.shape[1] * 0.1 )) ,cv2.FONT_HERSHEY_TRIPLEX,0.9,(0,0,0),2,cv2.LINE_AA)
                cv2.putText(image_section_1,"Try Adjusting hand position!" , (int(image_section_1.shape[0] * 0.01) , int(image_section_1.shape[1] * 0.3 )) ,cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),1,cv2.LINE_8)
            

            if(np.count_nonzero(threshold2) > 2500):

                ## Save the image for future training           
                
                FILE_2="area2.jpg"
                cv2.imwrite(FILE_2,threshold2)

                ## Get predicted value from the model

                self.prediction2,self.value2 = predict_on_img(FILE_2)
                self.right = self.class_to_move[self.value2]
                ## Print the result

                cv2.putText(image_section_2,self.class_to_move[self.value2] , (int(image_section_2.shape[0] * 0.3) , int(image_section_2.shape[1] * 0.9 )) ,cv2.FONT_HERSHEY_SIMPLEX,1,(71,55,255),3,cv2.LINE_AA)
            else:
                ## If not detected the hand, print message
                self.prediction2 = None 
                self.value2 = None
                self.right = None

                cv2.putText(image_section_2,"Not detected!" , (int(image_section_2.shape[0] * 0.05) , int(image_section_2.shape[1] * 0.1 )) ,cv2.FONT_HERSHEY_TRIPLEX,0.9,(0,0,0),2,cv2.LINE_AA)
                cv2.putText(image_section_2,"Try Adjusting hand position!" , (int(image_section_2.shape[0] * 0.01) , int(image_section_2.shape[1] * 0.3 )) ,cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),1,cv2.LINE_8)
            
            ## Rock Paper Scissor Algorithm 

            if(self.left is not None and self.right is not None):
                Left = self.left
                Right = self.right
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

                cv2.putText(flip,result, (180,370),cv2.FONT_HERSHEY_TRIPLEX,1.2,(0,214,255),2,cv2.LINE_AA)
            
        return av.VideoFrame.from_ndarray(flip, format="bgr24")


st.title("Welcome to Rok Papr Scissor- Live Multiplayer Rock Paper Scissor")
st.text("Steps to play:- ")
st.text("1. To begin, press the Start button and then click on the Play button.")
st.text("2. Choose a fixed background and allow the game to calibrate for a brief moment.")
st.text("3. Once the 'Please Wait...' message disappears, the game is ready to play. Please refrain from moving the camera any further as it may cause the background calibration to fail.")
st.text("4. Two players can show their hands in separate boxes on the screen.")
st.text("5. The game will display the result below the boxes.")

st.caption("Please note that it is important to avoid moving the camera during gameplay. In case the camera position is changed, kindly re-run the application from the top-right menu to recalibrate the background.")

webrtc_ctx = webrtc_streamer(
    key="PlayArea",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

