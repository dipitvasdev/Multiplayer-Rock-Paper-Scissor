
# Live Multiplayer Rock Paper Scissor using OpenCV

Rok Papr Scissor is a fantastic multiplayer game that can be easily accessed and played on computer devices. Unlike traditional Rock Paper Scissor games, Rok Papr Scissor allows two players to participate, adding a new level of excitement and competition to the gameplay.

To play the game, both players simply need to point their hands (forming rock, paper, or scissor shapes) towards the camera. The computer then utilizes advanced computer vision techniques to determine which player has won the game.

I am confident that you will enjoy this thrilling and engaging game. So, grab a friend and get ready to experience the ultimate gaming adventure with Rok Papr Scissor!






## Demo


#### Video

[![Watch the video](https://i.ibb.co/d0LnNrJ/Screenshot-2023-02-28-at-1-50-21-AM.png)](https://youtu.be/BBuC6eP-F30)


## Under the Hood


### Image Processing 

1. Use Running Average method to extract the background from the live camera feed.

2. Apply Gaussian filter to smoothen the edges and remove any unwanted noise or artifacts.

3. Apply Binary Thresholding with a threshold value of 30 to convert the image into a binary form.

By following these advanced image processing techniques, I was able to accurately detect and interpret the players' hand gestures.


### DL Model Architecture 

For the Rok Papr Scissor project, I have utilized the widely-used MobileNetV2 model in TensorFlow as the base model. To enhance the accuracy of the model, I have implemented data augmentation and customized top layers.

The model architecture is as follows:- 

![DL Model](https://i.ibb.co/qmL6NMB/Screenshot-2023-02-28-at-12-36-33-AM.png)

By utilizing the MobileNetV2 model and implementing advanced techniques, I have developed an accurate and reliable model capable of predicting the hand gestures in the Rok Papr Scissor game.


### Data Collection 

To train the model, I have collected the necessary data using a Python file called 'data_collection.py'. This file utilizes OpenCV to capture the required data from the camera feed. Once the data is collected, I apply the image processing techniques mentioned earlier to enhance the quality and accuracy of the data.

### Training 

During the model training process, all layers in the 'base_model' were frozen. The model was trained for 5 epochs using the carefully collected and processed training data. Additionally, separate validation data was collected and stored in data.zip to validate the accuracy of the model.


### Fine Tuning

To further boost the accuracy of the model is fine-tuned and the previously frozen layers, some of the layers are unfrozen. The model is then trained for an additional 3 epochs, which helps to achieve an impressive 98% validation accuracy.

### Convert to Tf Lite

To reduce model size and improve speed, the model is coverted to Tf Lite model and then involked in 'run.py' 

### Developing App


1. Capture 2 images from the 2 different boxes shown in the App's video.

2. Apply the image processing techniques mentioned above on these images to ensure consistency.

3. Invoke the Tf Lite model for both images and get the result for both images 

4. Analyze the predictions obtained from both boxes using the Rock Paper Scissor algorithm to determine the game's result, including whether it is a draw or if one player has won over the other.


### Google Colab and Jupyter Notebook

To see how the model was trained refer to link here:- https://colab.research.google.com/drive/1n1mNqck3eL4gIJo1alCzDyjeXmncXZvq?usp=sharing

The same notebook is also found in repository. 

## Run Locally

Clone the project

```bash
  git clone https://github.com/dipitvasdev/Multiplayer-Rock-Paper-Scissor.git
```

Go to the project directory

```bash
  cd Multiplayer-Rock-Paper-Scissor/
```

Install requirements

```bash
  pip install -r requirements.txt
```

Start the streamlit server

```bash
  streamlit run run.py
```


## 🚀 About Me
I am Dipit Vasdev, a highly motivated problem solver with a passion for neural networks and machine learning. I am currently pursuing a Master's degree in Computer Engineering at New York University, and my greatest strength lies in my drive for solving complex problems in computer science. 
I possess a wealth of technical skills in machine learning, deep learning, Android development, and more, and I have taken part in various projects and internships to continuously improve my skills and knowledge.


## 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dipit-vasdev/)



## Feedback

If you have any feedback, please reach out to me at dipit.vasdev@nyu.edu
