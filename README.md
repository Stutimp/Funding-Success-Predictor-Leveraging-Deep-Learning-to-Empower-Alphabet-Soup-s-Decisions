# Funding Success Predictor: Leveraging Deep Learning to Empower Alphabet Soup's Decisions

# Overview
This project aims to develop a deep learning model to predict the success of applicants who receive funding from Alphabet Soup, a nonprofit foundation. Leveraging a dataset of over 34,000 organizations, the goal is to build a binary classifier that can determine whether an applicant is likely to succeed based on various features such as application type, affiliation, income classification, and funding amount requested.

Alphabet Soup seeks to create an algorithm capable of predicting the success of funding applicants. By applying machine learning and neural networks, we will utilize the features in the provided dataset to develop a binary classifier that can accurately forecast whether applicants will be successful if funded by Alphabet Soup.

## Purpose of the Analysis
The primary objective of this analysis is to evaluate the performance of a neural network model developed for the binary classification task of predicting applicant success. Through a systematic examination of the model's performance, we aim to assess its effectiveness, identify areas for improvement, and explore alternative modeling strategies that may enhance the overall predictive accuracy.

## Steps Taken

1. **Data PreProcessing**

The dataset underwent several preprocessing steps to ensure optimal input quality for model training. First, the EIN and NAME columns were dropped, as they were not relevant for the model. The remaining columns were treated as features for the model. The data was then split into training and testing sets.

- **Target Variable:** IS_SUCCESSFUL—indicating whether the organization was successful after receiving funding.
- **Feature Variables:** Include application type, affiliation, classification, use case, organization type, status, income amount, special considerations, and funding amount requested.
- **Removed Variables:** EIN and NAME—identification columns that are not relevant for prediction.
- **Encoding:** Used pd.get_dummies() to encode categorical variables.
- **Data Splitting:** Split the data into training and testing sets using train_test_split.
- **Scaling:** Applied StandardScaler to normalize the data.

2.  **Model Creation, Training, and Evaluating the Model:**

- **Model Architecture:** Created a neural network model using TensorFlow and Keras with an input layer based on the number of features, one or more hidden layers with appropriate activation functions, and an output layer for binary classification.
- **Training:** Compiled and trained the model, saving the weights every five epochs.
- **Evaluation:** Evaluated the model using test data to determine loss and accuracy. The results were saved in the file `AlphabetSoupCharity.h5.`

 ## Model evaluation

Our deep learning model has achieved a loss of approximately 0.5620 and an accuracy of about 72.60% on the test dataset. These results give us insight into how well the model is performing on unseen data, which is crucial for understanding its generalization capabilities. The loss indicates how well the model predicts the actual outcomes, with lower values being better, and the accuracy indicates the percentage of correct predictions made out of all predictions.


### Overall Results Summary:
- **Performance:** The model has a moderate level of accuracy, suggesting it has learned patterns from the training data that generalize to some extent to unseen data. However, there might still be room for improvement, especially if the application requires higher accuracy( in our case, 75% accuracy.)

- **Generalization:** The loss value suggests that while the model has learned to classify to a certain degree, it's not yet optimal. The difference between training and test performance ( 0.7418>0.7260) indicates how our model is performing lower in test dataset compared to training dataset.

### Recommendations for Improvement:
To potentially improve upon these results or tackle the classification problem more effectively, considering a different model or architecture might be beneficial.


3.  ## Optimizing the model
Optimizing our model with the method you've provided involves using Keras Tuner, a library for hyperparameter tuning in TensorFlow. This method allows us to systematically search through a range of hyperparameters to find the most effective configuration for our neural network model. This process systematically explores a range of configurations and optimizes the model based on the specified objective, in this case, validation accuracy. It's a powerful approach to refine our model's architecture and hyperparameters to achieve better performance.


### Evaluate the Model
The multiple evaluations of our neural network model on the test dataset show slight variations in performance across different runs, Here’s a summary of our results:

**First Evaluation:**

Loss: 0.5587
Accuracy: 72.75%
Second Evaluation:

Loss: 0.5575
Accuracy: 72.64%
Third Evaluation:

Loss: 0.5579
Accuracy: 72.62%

### Observations:
- **Consistent Accuracy:** Across all three evaluations, the model maintains an accuracy in the range of 72.62% to 72.75%. This consistency indicates that our model is stable and provides reliable predictions when presented with unseen data, which is a positive indicator of its generalization ability.

- **Slight Variations in Loss:** There are minor differences in the loss values across evaluations, with the lowest being 0.5575 and the highest at 0.5587. 

- **Good Generalization:** The relatively stable performance metrics suggest that the model generalizes well to new data. The closeness of accuracy values across runs indicates that the model's performance is not highly sensitive to the specific subsets of data it is tested on, which is desirable.

### **Conclusion:**
Our neural network model exhibits consistent performance with a slight variance in loss and accuracy across multiple evaluations. The model's accuracy, hovering around 72.6% to 72.75%, demonstrates its capability to reliably classify unseen data, making it a potentially valuable tool for your binary classification task.

In the analysis, I have tried many ways with minimum 3 optimization methods by creating sequential models with different hyperparameter options to solve the same problem, but I was not able to get the accuracy rate above and equal to 75%, which is our desired accuracy rate for the analysis. 
Given these observations, future work could involve investigating methods to further increase accuracy ,and decrease loss, such as experimenting with different architectures, further hyperparameter tuning, or increasing the dataset size and diversity. Additionally, examining model performance on a more granular level, such as analyzing the types of errors it makes, could provide insights into specific areas for improvement.


Thank you !

Author

Stuti Poudel




