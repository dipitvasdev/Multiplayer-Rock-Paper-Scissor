# Rok Papr Scissr

## Description 

The standard Rock Paper Scissor ,but with 2 players !

Rok Papr Scissor is a multiplayer game that can be played on computer devices. Two players point their hands (rock, paper and scissor) at the camera and then using computer vision techiniques computer tells which player has won the game. 

## Image Processing 

1. Using <b>Running Average</b> to extract background. 
2. Apply gaussion filter to smoothen the edges 
3. Apply Binary Thresholding with a threshold value of 30. 
4. Save this image. 
5. Use `Deep Leaening Model` to predict on this image
6. Do this for both boxes 

## DL Model 

The architecture of DL Model is as follows : 
![image](https://user-images.githubusercontent.com/23167934/112476089-8a039200-8d97-11eb-93b2-2303b19d1756.png)


## Installation 

1. Clone this repository in your system. 
2. Download model file form here 
3. As the data is already, collected run the `game.py` 
4. You can then play the game 

## File Description and Steps Involved 

1. `data_collection.py` : This file is used to collect data for rock paper and scissor. All the image processing techiniques are applied and the images are stored in `data` folder.

![image](https://user-images.githubusercontent.com/23167934/112476725-38a7d280-8d98-11eb-9bf0-eac844467a01.png)

2. `game.py` : This file is the actual game file with 2 boxes for each player, it takes the images feeds them to the model and then displays the predictions and result on the window frame.

![image](https://user-images.githubusercontent.com/23167934/112477362-edda8a80-8d98-11eb-8b48-472931290ed9.png)

