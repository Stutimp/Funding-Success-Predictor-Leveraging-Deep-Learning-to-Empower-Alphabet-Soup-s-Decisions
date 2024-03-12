### Deep Learning Challenge: 

## Charity Funding Predictor

## Overview:

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not
applicants for funding will be successful. With knowledge of machine learning and neural
networks, we must use the features in the provided dataset to create a binary classifier that is
capable of predicting whether applicants will be successful if funded by Alphabet Soup.


### Purpose of the Analysis

The primary objective of this analysis is to evaluate the performance of a neural network model developed for a binary classification task. Through systematic examination, we aim to understand the model's capabilities, identify areas for improvement, and explore alternative modeling strategies for potentially enhanced outcomes.

### Data PreProcessing

The dataset underwent preprocessing steps such as dropping scaling and potentially augmentation, ensuring optimal input quality for model training. We dropped EIN and
NAME and the remaining columns were to be considered features for the model. The data was then split for training and testing 
sets. The target variable for the model was labeled “IS_SUCCESSFUL” and has the value of 1
for yes and 0 for no. APPLICATION data was analyzed and “CLASSIFICATION value was used
for binning. We used several data points as a cutoff to bin “rare” variables together with the new
value of “Other” for each unique value. Categorical variables were encoded by get_dummies()
after checking to see if the binning was successful.

### Results: Using bulleted lists and images to support your answers, address the following questions:

***Data Preprocessing***

***What variable(s) are the target(s) for your model?***
- Target Variable: "IS_SUCCESSFUL" which is refered with 'y' in the code.

***What variable(s) are the features for your model?***
- Feature Variables: All other columns in application_df_dummies after dropping "IS_SUCCESSFUL", which is 'X' in our code.

**What variable(s) should be removed from the input data because they are neither targets nor features?**
- "EIN" and "NAME" , both are neither targets nor features contributing to input feature for machine learning model, hence we should remove both of them.

### Compiling, Training, and Evaluating the Model

***How many neurons, layers, and activation functions did you select for your neural network model, and why?***

- First Hidden Layer: This layer consists of 80 neurons (hidden nodes) and uses the ReLU (Rectified Linear Unit) activation function. The choice of 80 neurons for this layer suggests a relatively complex model capable of learning from a large number of input features. The ReLU activation function is widely used in deep learning due to its efficiency and the ability to solve the vanishing gradient problem, making it a good choice for the first hidden layer.

- Second Hidden Layer: This layer has 30 neurons and also uses the ReLU activation function. A smaller number of neurons in this layer compared to the first hidden layer might be intended to begin the process of reducing the complexity of the model's representations as it prepares for output, while still allowing for a rich hierarchy of learned features. Using ReLU again here maintains consistency in activation functions across the network, promoting efficient learning.

- Output Layer: The output layer has a single neuron with a sigmoid activation function. This configuration is typical for binary classification tasks, where the goal is to predict two possible outcomes (0 or 1). The sigmoid function outputs a value between 0 and 1, which can be interpreted as the probability of the input belonging to the positive class.

**Were you able to achieve the target model performance?**
- Our goal was to achieve 75% accuracy with our neural network model. Upon evaluation, our model achieved 72.60%, which does not meet our initial targets.  I also adjusted neuron Counts. Increasing the number of neurons can help the model capture more complex relationships, while decreasing them can prevent overfitting.Likewise, I adjusted the depth of the model which can help in learning more complex patterns or reduce overfitting by simplifying the model.

**What steps did you take in your attempts to increase model performance?**
- Improving a neural network model's performance often involves a series of systematic steps. In order to optimize the model using hyperparameter tuning further on, I tried with different activation functions (e.g., ReLU, ELU, Leaky ReLU) in different layers to improve training efficiency and model accuracy.

### Compiling, Training, and Evaluating the Model:

There were three layers total for each model after applying Neural Networks. The number of
hidden nodes were dictated by the number of features.


### Evaluation of the model

Our deep learning model has achieved a loss of approximately 0.5620 and an accuracy of about 72.60% on the test dataset. These results give us insight into how well the model is performing on unseen data, which is crucial for understanding its generalization capabilities. The loss indicates how well the model predicts the actual outcomes, with lower values being better, and the accuracy indicates the percentage of correct predictions made out of all predictions.


### Overall Results Summary:
- Performance: Our models (Neural basic Model and optimization models) all have  moderate level of accuracies, suggesting it has learned patterns from the training data that generalize to some extent to unseen data. However, there might still be room for improvement, especially because the models are not able to achieve desired higher accuracy rate( in our case, 75% accuracy.)

- Generalization: The loss value suggests that while the model has learned to classify to a certain degree, it's not yet optimal. The difference between training and test performance ( 0.7418>0.7260) in neural network model indicate how our model is performing lower in test dataset compared to training dataset. The accuracy score of 0.7275 and loss rate of 0.5587 also suggest the model is not performing as per our expectations.

### Recommendations for Improvement:
To potentially improve upon these results or tackle the classification problem more effectively, considering a different model or architecture might be beneficial..


### Optimizing the model
Optimizing our model with the method you've provided involves using Keras Tuner, a library for hyperparameter tuning in TensorFlow. This method allows us to systematically search through a range of hyperparameters to find the most effective configuration for our neural network model. This process systematically explores a range of configurations and optimizes the model based on the specified objective, in this case, validation accuracy. It's a powerful approach to refine our model's architecture and hyperparameters to achieve better performance.


### Evaluate the Model
The multiple evaluations of your neural network model on the test dataset show slight variations in performance across different runs, Here’s a summary of your results:

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
- Consistent Accuracy: Across all three evaluations, the model maintains an accuracy in the range of 72.62% to 72.75%. This consistency indicates that our model is stable and provides reliable predictions when presented with unseen data, which is a positive indicator of its generalization ability.

- Slight Variations in Loss: There are minor differences in the loss values across evaluations, with the lowest being 0.5575 and the highest at 0.5587. 

- Good Generalization: The relatively stable performance metrics suggest that the model generalizes well to new data. The closeness of accuracy values across runs indicates that the model's performance is not highly sensitive to the specific subsets of data it is tested on, which is desirable.

### **Conclusion:**
Our neural network model exhibits consistent performance with a slight variance in loss and accuracy across multiple evaluations. The model's accuracy, hovering around 72.6% to 72.75%, demonstrates its capability to reliably classify unseen data, making it a potentially valuable tool for your binary classification task.

In the analysis, I have tried many ways with minimum 3 optimization methods by creating sequential models with different hyperparameter options to solve the same problem, but I was not able to get the accuracy rate above and equal to 75%, which is our desired accuracy rate for the analysis. 
Given these observations, future work could involve investigating methods to further increase accuracy ,and decrease loss, such as experimenting with different architectures, further hyperparameter tuning, or increasing the dataset size and diversity. Additionally, examining model performance on a more granular level, such as analyzing the types of errors it makes, could provide insights into specific areas for improvement.








