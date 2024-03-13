# Problem:  Predict Students' Dropout and Academic Success

### INTRODUCTION

Academic performance is a vital indicator of the students' future, which includes various aspects of their lives, such as jobs, success, personal drive, and ambition. It is also a great indicator of more macroscopic aspects of a demographic, like the progress of the economy, social circumstances, and such [1]. However, academic excellence can be affected by various attributes such as family income, attendance, and regularity, student personality, so identifying these factors becomes essential to this study.
We find this problem very interesting since understanding how various factors can dictate academic performance could potentially lead to preemptive approaches to tackle students at risk before it is too late, as and when we start to notice a trend [2]. The outcome of this project can also create awareness and increase the rate of students' success.
Machine learning techniques have a wide variety of applications in prediction, among other domains. One such application is our use case - predicting Dropout students and academic success. This project aims to use various machine learning models to predict students' academic success and do an intra and inter-comparison of the models used. To ensure better accuracy, we have ensured that the same standards were followed throughout the various techniques used.
The core idea is to compare models and understand if and why one outperforms the others. As an extra initiative, we also wanted to understand if enhancing the models by making minor tweaks to the code would be possible.
We strongly believe that being able to determine factors that dictate a student's success has benefits for both - the institute and the students by not just identifying attributes but also by making predictions on future data [2].

### EXPERIMENT SETUP
#### Methodology:
In our project, we developed four distinct algorithms to compare the results and evaluate the performance of our models based on a test dataset set aside solely for result-reporting purposes. In each algorithm, relevant preprocessing schemes were used in order to optimize the training process. We also used sampling techniques to improve the imbalances of our target values. 
As discussed in the introduction, the primary focus of this project is the classification of different student academic statuses, that is: Enrolled, which indicates a student is currently enrolled in university and is pursuing higher education; Graduate, which stands for the students who successfully finished their curriculums and obtained their degree, and Dropout, showing the corresponding students terminated their student program before attaining their degree.

#### Dataset and Preprocessing:
The dataset was created by Valentim Realinho et al. [3] from several disjoint databases from higher education institutes where each instance defines a unique student. There are 4424 instances from different undergraduate streams of agronomy, design, education, nursing, journalism, management, social service, and technologies. The dataset includes information like academic path, demographics, and socioeconomic factors, along with their academic performance in the first and second semesters. There are 36 features, each represented in numeric value of each feature. For example, in Marital Status, one means single, two means married, three means widower, and so it continues. They have done an excellent job preprocessing the data in removing anomalies, unexplainable outliers, and missing values.
Upon using the dataset file, one row was all in one cell - so we made a subtle configuration to convert text to columns. The dataset file is attached among all other sources to the project submission.

#### Algorithms Used:
As mentioned in the prior section, we used three classical machine learning algorithms and one deep learning solution to experiment and evaluate different models based on their performance regarding student academic success detection. The algorithms are as follows:
Naive Bayes
Decision Tree
Logistic Regression
Artificial Neural Networks

We will discuss each algorithm or model in detail in the subsequent sections and compare the results in the final section.

1. Naive Bayes (NB): Naive Bayes is a model that heavily depends on the likelihood of target labels occurring in the training data. Additionally, Naive Bayes does not incorporate any way to check for correlation between its variables. So, we first implemented Naive Bayes as it was. We implemented an updated version of Naive Bayes, where we first ran a correlation matrix to identify attributes that had a strong correlation between them. We then removed one of the two variables that were highly correlated. By doing this, we reduced the dataset dimensions more efficiently and re-executed the Naive Bayes algorithm. The results of both variations of the Naive Bayes algorithm are shared in the subsequent sections.

2. Decision Tree: For the Decision Tree, we imported DecisionTreeClassifier from sklearn and split data into 75:15 proportions for training data and testing data. Upon running the algorithm without any modification to the data, we discovered an initial accuracy of 67%. We also noticed that the f1-score for Graduate and Dropout was significantly higher than Enrolled. It was unsurprising as Enrolled instances are only 18% of total instances. We also plotted the tree[Appendix Fig 1] using the matplot library, with a depth level of 27. After looking into the correlation of features, we dropped some of the features and reran the algorithm to find whether it improved results. Doing so should not make any significant difference in results as we know Decision trees make no assumptions on relationships between features[4]. However, we observed no variance in overall accuracy, though it did improve the f1-score by almost 9% of Enrolled, which was quite surprising.
As we all know, a Decision Tree is vulnerable to overfitting if a single decision tree is constructed[5]. Decision Tree uses variables like impurity measures Gini and entropy, which means that we do not gain much information with more depth level and tend towards more risk of being vulnerable to overfitting and decreased accuracy[6]. We decided to experiment with different levels of depth and see if we could improve the accuracy and reduce the chance of overfitting. On increasing the depth level, we found maximum accuracy with depth level 6, which is 75%. We also observed that the f1-score of not only Enrolled was increased but also of Graduate and Dropout. When we compared the gini value of each node in our first experiment and decision tree[Appendix Fig 2] with a depth limit of 6, we observed there are plenty of nodes whose gini value was 0, which have not been observed in the newer decision tree with a depth limit of 6. We also experimented again by increasing the depth limit and identified that the model is gaining little information and impurities have been increased. With this experiment, we can also see the model could be more confident in prediction with depth limit; however, we have better results and accuracy.

3. Logistic Regression (LR): Since LR is a discriminative classifier, as opposed to NB, which is a generative classifier, we wanted to run this model to check if this would give us better results than both variations of NB. DT and LR are both types of discriminative models, so comparing these two also was important to understand if any outperforms the other. Since we had more than 2 target labels (Dropout, Enrolled & Graduate), a multinomial logistic regression model was implemented.

4. Artificial Neural Networks: (hereafter, ANNs) are the modern algorithms used for their powerful embedded feature extraction advantage. They enable us to do any form of classification only through a number of minor adjustments in the model and certain preprocessing in the input layer. We investigated the performance of ANNs by proposing four models without input preprocessing and four models with preprocessing. In the following sections, we will dig into these model architectures. 
Each model has a 36-sized vector of neurons for the first layer as the input layer, in addition to 3 neuron-sized final layers with a softmax activation function. The main intent of the softmax function is to have a similar behavior to the logistic sigmoid function and represent a probability value for each output that can sum up to one and is between the 0 and 1 values. Additionally, each model uses an ADAM optimizer for gradient descent with 100 epochs, and the loss function is sparse categorical cross-entropy. 
Three activation functions were investigated in the models: relu, elu, and tanh. The reason that the relu function is so beneficial is that it provides non-linearity and linearity according to the weights learned by the network. Elu function brings the same advantages, whereas the tanh function completely changes that linearity and compresses the data into values of -1 and 1, which can mean a loss of information in the distribution of the previous neuron value.
For every model, we also separated training, validation, and test datasets by sampling without replacement, in respective order: 70%, 15%, and 15%.

Model 1:
In our primitive model, we developed a simple ANN with a total of three hidden layers following the input layer and prior to the final layer, which is the output. The schema of this model is 128 relu in the first hidden layer, 64 of the same function in the second,  and 128 in the third. This model was the most basic one among all others and had an accuracy of 51%.

Model 2: 
With the basic idea in the open, we tried to deepen our model by adding layers and decreasing the number of neurons in each layer. Even though the total number of hidden layers is increased by 4, the number of parameters is reduced by almost 45%, making the model deeper and lighter. The respective architecture is 64 neurons of relu activation function, followed by 32 neurons and then again by 64 and then four 32-neuron layers. The final accuracy of model 2 with respect to the same test dataset as model 1 is 73%.

Model 3: 
This model is identical to the second model but has elu activation functions instead of relu. It almost obtained the same accuracy of the second model, which was 70%.

Model 4:
This model has the same characteristics as the second one except that it uses the tanh activation function in the third and fifth hidden layers. The accuracy of this model after testing was 51%.
After evaluating all these models together, we also applied the best model (i.e., model 2) after doing a few more preprocessing.
We used different sampling methods, such as the undersampling and oversampling methods, and lastly, combining the two. However, the results unexpectedly didn't improve that much, and it digressed significantly in terms of accuracy (all of them acquired around 30%).

### RESULTS:

The table below shows the comparison of accuracy measures of the target labels (Dropout, Enrolled or Graduate).

|                   | Precision | Recall | f1-score | Accuracy |
|-------------------|-----------|--------|----------|----------|
| **NAIVE BAYES**   |           |        |          |          |
|                   | 0.73      | 0.67   | 0.70     | 0.66     |
|                   | 0.23      | 0.14   | 0.17     |          |
|                   | 0.70      | 0.84   | 0.77     |          |
| **NAIVE BAYES 2.0** |         |        |          |          |
|                   | 0.79      | 0.67   | 0.72     | 0.69     |
|                   | 0.33      | 0.23   | 0.27     |          |
|                   | 0.73      | 0.87   | 0.79     |          |
| **DECISION TREE** |           |        |          |          |
|                   | 0.75      | 0.65   | 0.70     | 0.67     |
|                   | 0.28      | 0.35   | 0.31     |          |
|                   | 0.78      | 0.78   | 0.78     |          |
| **DECISION TREE 2.0 (Correlation)** |        |          |          | |
|                   | 0.67      | 0.68   | 0.67     | 0.67     |
|                   | 0.39      | 0.42   | 0.40     |          |
|                   | 0.79      | 0.76   | 0.77     |          |
| **DECISION TREE 3.0 (Depth limit)** |        |          |          | |
|                   | 0.85      | 0.67   | 0.75     | 0.75     |
|                   | 0.48      | 0.35   | 0.40     |          |
|                   | 0.76      | 0.95   | 0.84     |          |
| **LOGISTIC REGRESSION** |         |        |          |          |
|                   | 0.81      | 0.76   | 0.78     | **0.78**     |
|                   | 0.55      | 0.34   | 0.42     |          |
|                   | 0.81      | 0.95   | 0.87     |          |
| **ARTIFICIAL NEURAL NETWORKS** |      |        |          |          |
|                   | 0.80      | 0.73   | 0.76     | 0.73     |
|                   | 1.00      | 0.04   | 0.08     |          |
|                   | 0.69      | 0.98   | 0.81     |          |


### DISCUSSION & CONCLUSION:

From the results we can gather that one particular class - Enrolled, under performs for all models compared to the other two classes - Dropout & Graduate. It is quite intuitive since the dataset is biased and we have only 18% of the dataset under the ‘Enrolled’ label. We observe that reducing dimensions based on correlation, for both Naive Bayes and Decision Tree, increases the performance of the model, i.e., Naive Bayes 2.0 outperforms Naive Bayes. 

Upon all three experiments of the decision tree, we found that there are lots of features in the data which do not give much information in deciding the academic success rate of students. Due to the biased dataset, we restricted the limit of the tree to 6 which maximizes the accuracy and f1-score of all the Categories with overall accuracy of 75% and f1-score of 75%, 40% and 84% for Dropout, Enrolled, and Graduate respectively. However, sampling techniques proved to be not only ineffective for the ANNs but also deficient and reported accuracy was reduced.

Decision Tree & Logistic Regression are both discriminative models and do perform better than Naive Bayes so we can also observe that for this dataset, the discriminative models have a much higher performance than the generative one. Although the ANNs performed relatively decently compared to Naive Bayes, the model’s f-score for the Enrolled class in specific, was poor in comparison to other algorithms.
