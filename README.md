# Speech-Emotion-Recognition

In this project, we are going to develop a system for recognizing the speech of a person and identifying the emotion of the person. This is done by using the Convolution Neural Network (CNN). For this project, we have used the Speech Emotion Recognition (SER) dataset from Kaggle. This dataset contains several audio recordings that are recorded by professional voice artists who are both male and female. We will design and train our own Convolutional Neural Network (CNN) to classify the audio recordings into different emotions, this dataset contains audio recordings that are labelled with seven audio recordings which are Angry, Sad, Fear, Disgust, Happy, Neutral, and Surprise. For the data preprocessing of the audio signals of the speech, we have extracted features such as MFCCs and used them as input to the Convolutional Neural Network (CNN). Also, for additional preprocessing of the audio signals, we reduce the duration of the audio signals which were too long and for the audio signals which were too short, we added some noise to the input audio signal.
After the data preprocessing, we designed our own CNN model and trained it with the input audio signal recordings. For the CNN model, we have used several layers to achieve high accuracy. The performance of the CNN model was also evaluated such as precision, F1-score, Recall, and accuracy. Our model achieved an overall validation accuracy of 96.38% and a test accuracy of 95.86%. This was achieved after setting the epochs to 30, from this we can say that the CNN model performed very well in recognizing the emotions from the audio signals of the speech. And this system can have several applications in various fields such as customer service and diagnosis of patients with mental health issues.
