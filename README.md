# Face_Emotion_Recognition
# Overview:
We intend to achieve face detection as well as emotion recognition as a result of this project. For this, we used Transfer learning which actually means that we will build a model on a model which is already trained on a large dataset and then fine tune it as per our project requirements. For face detection we used a pre-trained face detection model (Haar Cascade) that detects faces in a frame using classical computer vision, and when it comes to Emotion recognition as we discussed above we used transfer learning for training the model.
# Dataset:
The dataset which we have used for achieving Face and emotion recognition is FER 2013 which is Face and emotion recognition 2013. The reason for me choosing this particular dataset is it perfectly fits our project objectives. It consists of images of Face used for face detection and divided these images into 7 classes of emotions. Namely, Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. Our dataset contains nearly 36,000 images out of which 30,000 approx is for training and the remaining is for testing. Each image in this dataset is a gray scale image of 48 x 48 pixels. 
# Data pre-processing:
Data preprocessing is literally preparing the data we have and making it suitable for our model creation. This is the most important phase in achieving our results. As, our images are already in gray scale so we need not separately do it. Firstly, using Haar Cascade we extract the face using a bounding box from the entire image because at the end of the day we only want the face to extract the relevant features, not the entire image. Then, as we said we trained the model on pre-trained models called MobilenetV2 and Resnet50. So, these models expect the size of the picture to be 224x224 pixels but our dataset consists of images 48x48. So we need to resize the images to 224x224 pixels. We then normalise the entire data which we are using for training to increase the stability of the model. We will convert the data type because we are using tensorflow, so we change the data type as per the requirements of tensorflow. 
# Transfer Learning:
Transfer learning is a machine learning technique where a pre-trained model developed for one task is reused as a starting point for a model on a second , related task. Instead of training a model scratch, you leverage the knowledge the model has already learned, which makes the entire training process much faster.
---> Why did we use Transfer learning?
Deep learning models like CNN requires:
Thousands to millions of images.
Huge computing power.
Days or weeks of training time.
With transfer learning, we can:
Use pre-trained models, in our case we used MobileNetV2 and ResNet50.
Makes the training process very fast.
Achieve high accuracy.
When it comes to our project we used MobileNetV2 and ResNet50 for emotion recognition. Both these models are pre trained on ImageNet which consists of 1000 classes like edges, textures, shapes, patterns etc. But as we are focusing on Emotion recognition, we don't need all these classes, so we are gonna add some custom layers to this model as per our requirements.                                               
We will remove the top layer and add New dense layers, ReLU, and softmax layer with 7 classes of all our emotions for emotion classification. We did exactly the same changes to the ResNet50 model, then we trained this customised model with our FER 2013 training dataset with 20 epochs. After the entire training we realised that the ResNet model was more accurate with nearly 98% accuracy whereas MobileNetV2 has 91% accuracy.  
Training time for the entire model was too high for ResNet compared to MobileNet. To be precise, MobileNet took around 4 hours for the training whereas ResNet took around 14 hours for training the dataset on model. At the end of training ResNet proved to be a better model compared to MobileNet.
