# Image-Processing-Model
*************************************************************************************************
This is a machine learning image processing model developed using **neural network** with **TensorFlow.** It can predict the class of fashion wear with an **accuracy of 90%**

**************************************************************************************************
# Table of Contents
- [Introduction](#introduction)
- [Objective](#objective)
- [Dataset](#dataset)
- [Evaluation Criteria](#evaluationcriteria)
- [Performence](#performence)
- [How To Use](#howtouse)

## Introduction:
- The Fashion-MNIST dataset is a popular dataset used for classification in the Machine Learning community to build and test neural networks. Although MNIST is widely used for digit classification, it is considered a trivial dataset for neural networks as they can achieve accuracy above 90% quite quickly. Experts recommend moving away from MNIST dataset for model benchmarking and validation. On the other hand, Fashion-MNIST is more complex and a much better dataset for evaluating models than MNIST.
## Objective:-
- We aim to construct a neural network through Tenserflow , exclusively incorporating convolutional neural neworks. The objective is to achieve optimal accuracy in classifying ten categories of apparel images within the Fashion-MNIST dataset
## Dataset:
- The dataset is composed of 60,000 training images and 10,000 testing images, with each image in the dataset categorized into one of the ten classes.
- For our implementation, we will utilize the Fashion-MNIST dataset provided by TensorFlow's corresponding package. This approach ensures a streamlined and pre-processed dataset, facilitating the seamless association of each image with its corresponding label. This structured format simplifies the iteration process during both model training and testing.

## Performence:-
- We have observed that the network is able to accurately identify the image with a high degree of precision. This is promising progress. Now, we need to evaluate our trained network on a complete test set. After conducting the evaluation, we found that our model achieved a prediction accuracy of 88.10 %. This is a great result, especially considering that our model is a simple neural network.
- It is possible to enhance the performance of a model and achieve an accuracy above 90% by experimenting with the architecture and tuning hyperparameters. However, obtaining an accuracy between 95-98% using a model that only employs fully connected layers is challenging. This is because the Fashion-MNIST dataset is more complicated than the MNIST dataset. When images of 28x28 are flattened into a vector of 784 elements to be processed by a fully connected layer, they lose their spatial structural information.

## How To Use
- Open (.py) file from the repository.
- Just enter the index(anything between **0 to 10000**) of picture which you want our model to predict.
   
- And just hit the enter butten.
you will see your selected model's the rough picure(plotted by matplotlib) and its real class and the predicted class guess by our model.

************************************************************************************************
**Pls have a try, have a nice day ! :) **
************************************************************************************************


