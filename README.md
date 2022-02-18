# Pneumonia image recognition/classification project

# What is pneumonia?

<img src='https://github.com/314ka4y/Image_classification/blob/main/img/300_pneu.png' width=300 align='right'/>
<br>
Pneumonia is an inflammatory condition of the lung primarily affecting the small air sacs known as alveoli. Symptoms typically include some combination of productive or dry cough, chest pain, fever, and difficulty breathing. The severity of the condition is variable.
Pneumonia is usually caused by infection with viruses or bacteria, and less commonly by other microorganisms.Identifying the responsible pathogen can be difficult. Diagnosis is often based on symptoms and physical examination. Chest X-rays, blood tests, and culture of the sputum may help confirm the diagnosis
Each year, pneumonia affects about 450 million people globally (7% of the population) and results in about 4 million deaths.

Source: https://en.wikipedia.org/wiki/Pneumonia 


# Overview
With the increasing number of pneumonia cases during COVID-19 are hospitals are overwhelmed with additional work. One of the main methods that are used to diagnosed pneumonia and COVID-19 is by making X rays of chrest. I was hired by a local hospital to create a model that can automatically classify person on having or not pneumonia based on their X-ray. This system should have high accuracy and high recall.

To acheive this goal, We used image recognition with DeepLearning techniques including CNN and pretrained CNN.


# Business Understanding
Our stakeholder wants to have model that can be reliable in predicting when person have pneumonia.
Additional requirement - recall at least 95%.

# Used metrics
##### Our project will answer following question:
Can we predict people with pneumonia based on their chrest X-ray?

##### Hypothesis:
H0 - Person has pneumonia

HA - There is statisticaly significant proof that the preson doesnt' have pneumonia

##### TP, TN, FP, FN definition
TP - we predicted pneumonia and it actually exist.

TN - we predicted that person didn't have pneumonia and the person actually didn't have it.

FP - We predicted pneumonia but there was no pneumonia in real life.

FN - We predicted that there is no pneumonia but it actually existed.

##### Metrics used
To compare models we will focus on 2 major metrics:

Recall(Sensitivity) - Health of people is our priority, we will be focused to minimize FN.
Accuracy - how good we can predict TP and TN. General metrics that will show model performance.


# Data Understanding
The data used for this project was sourced from a dataset:

1) Chest X-Ray Images. Year: 2018 Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images”, Mendeley Data, V3, doi: 10.17632/rscbjbr9sj.3

https://data.mendeley.com/datasets/rscbjbr9sj/3

##### The dataset contains the following images:

Train set:

There are 1349 normal images, image name example, NORMAL-2552119-0002.jpeg

There are 4489 pneumonia images, image name example, BACTERIA-4038442-0001.jpeg

Test set:

There are 234 normal images, image name example, NORMAL-8698006-0001.jpeg

There are 390 pneumonia images, image name example, VIRUS-2040583-0001.jpeg


# Modeling

This project uses image recognition with deep learning, using the following techniques:

- Fully Connected Neural Networks

- Convolutional Neural Networks

- Pretrained Convolutional Neural Networks

1. The modeling began with a baseline Fully Connected Neural Networks. We face overfitting using this method 
2. In attemps to increase our accuracy score, we different regularization tactics 
3. We increased image resolution to overcome overfitting.
4. We used  augmentation to increase our accuracy. 
5. We used pretraince CNN models with augmentation, using ImageDataGenerator and tuning shear_range, zoom_range, and horizontal_flip. This model resulted in the hgihest accuracy, lowest loss
6. Finally we tuned decision boundaries for our classification problem and propose several methods how it can help to change model performance based on stakeholders demands. 

#  Results
Our regression model results are as follows:




Below is an image showing the confusion matrix:



By optimizing thresholds we can get these results:



# Conclusion
Due to the randomness of training results may vary when the model runs on a local machine.
This type of methodology can be extremely useful in the identification of infections and abnormalities in medical imaging (not just x-ray but MRI, CT's, etc..). The use of machine learning techniques has the potential to be extremely useful in the medical field, but it is very dependable on the quality of input training data. Machine learning algorithms trained on incorrect data might give false results in the future. 


# Further Questions
See the full analysis in the [Jupyter Notebook](https://github.com/314ka4y/Image_classification/blob/main/Project_classification.ipynb) or review [this presentation]()


# Repository Structure
```
├── data                # contains original datasets and saved models
│   ├── normal          # Train normal X-Ray images 
│   ├── pneumonia       # Train pneumonia X-Ray images
│   ├── test            # Test set 
│   ├── models          # Saved tensorflow model
├── images              # Images used in this project
├── README.md
├── Pneumonia_image_recognition.ipynb
└── Pneumonia_image_recognition.pdf