 Diabetes Prediction Using Machine Learning
 Tushar Yadav - DTU
Diabetes is a chronic disease with the potential to cause a worldwide health care crisis. According to International Diabetes Federation 382 million people are living with diabetes across the whole world. By 2035, this will be doubled as 592 million. Diabetes is a disease caused due to the increase level of blood glucose. This high blood glucose produces the symptoms of frequent urination, increased thirst, and increased hunger. Diabetes is a one of the leading cause of blindness, kidney failure, amputations, heart failure and stroke. When we eat, our body turns food into sugars, or glucose. At that point, our pancreas is supposed to release insulin. Insulin serves as a key to open our cells, to allow the glucose to enter and allow us to use the glucose for energy. But with diabetes, this system does not work. Type 1 and type 2 diabetes are the most common forms of the disease, but there are also other kinds, such as gestational diabetes, which occurs during pregnancy, as well as other forms. Machine learning is an emerging scientific field in data science dealing with the ways in which machines learn from experience. The aim of this project is to develop a system which can perform early prediction of diabetes for a patient with a higher accuracy by combining the results of different machine learning techniques. The algorithms like K nearest neighbour, Logistic Regression, Random forest, Support vector machine and Decision tree are used. The accuracy of the model using each of the algorithms is calculated. Then the one with a good accuracy is taken as the model for predicting the diabetes. Keywords : Machine Learning, Diabetes, Decision tree, K nearest neighbour, Logistic Regression,
Support vector Machine, Accuracy .
1. INTRODUCTION
Diabetes is the fast growing disease among the people even among the youngsters.
In understanding diabetes and how it develops, we need to understand what happens in the body without diabetes. Sugar (glucose) comes from the foods that we eat, specifically carbohydrate foods. Carbohydrate foods provide our body with its main
energy source everybody, even those people with diabetes, needs carbohydrate. Carbohydrate foods include bread, cereal, pasta, rice, fruit, dairy products and vegetables (especially starchy vegetables). When we eat these foods, the body breaks them down into glucose. The glucose moves around the body in the bloodstream. Some of the glucose is taken to our brain to help us think clearly and function. The remainder of the glucose is taken

to the cells of our body for energy and also to our liver, where it is stored as energy that is used later by the body. In order for the body to use glucose for energy, insulin is required. Insulin is a hormone that is produced by the beta cells in the pancreas. Insulin works like a key to a door. Insulin attaches itself to doors on the cell, opening the door to allow glucose to move from the blood stream, through the door, and into the cell. If the pancreas is not able to produce enough insulin (insulin deficiency) or if the body cannot use the insulin it produces (insulin resistance), glucose builds up in the bloodstream (hyperglycemia) and diabetes develops. Diabetes Mellitus means high levels of sugar (glucose) in the blood stream and in the urine.
Types of Diabetes
Type 1 diabetes means that the immune system is compromised and the cells fail to produce insulin in sufficient amounts. There are no eloquent studies that prove the causes of type 1 diabetes and there are currently no known methods of prevention.
Type 2 diabetes means that the cells produce a low quantity of insulin or the body can’t use the insulin correctly. This is the most common type of diabetes, thus affecting 90% of persons diagnosed with diabetes. It is caused by both genetic factors and the manner of living.
Gestational diabetes appears in pregnant women who suddenly develop high blood sugar. In two thirds of the cases, it will reappear during subsequent pregnancies. There is a great chance that type 1 or type 2 diabetes will occur after a pregnancy affected by gestational diabetes.
Symptoms of Diabetes
• Frequent Urination
• Increased thirst
• Tired/Sleepiness
• Weight loss
• Blurred vision
• Mood swings
• Confusion and difficulty concentrating • frequent infections
Causes of Diabetes
Genetic factors are the main cause of diabetes. It is caused by at least two mutant genes in the chromosome 6, the chromosome that affects the response of the body to various antigens.
Viral infection may also influence the occurrence of type 1 and type 2 diabetes. Studies have shown that infection with viruses such as rubella, Coxsackievirus, mumps, hepatitis B virus, and cytomegalovirus increase the risk of developing diabetes.
2. LITERATURE REVIEW
It uses the classification on diverse types of datasets that can be accomplished to decide if a person is diabetic or not. The diabetic patient’s data set is established by gathering data from hospital warehouse which contains two hundred instances with nine attributes. These instances of this dataset are referring to two groups i.e. blood tests and urine tests. In this study the implementation can be done by using WEKA to classify the data and the data is assessed by means of 10-fold cross validation approach, as it performs very well on small datasets, and the outcomes are compared. The naïve Bayes, J48, REP Tree and Random Tree are used. It was concluded that J48 works best showing an accuracy of 60.2% among others.( Aljumah, A 2012 [1])
It Aims to discover solutions to detect the diabetes by investigating and examining the patterns originate in the data via classification analysis by using Decision Tree and Naïve Bayes algorithms. The research hopes to propose a faster and more efficient method of identifying the disease that will help in well-timed cure of the patients. Using PIMA dataset and cross validation approach the study

concluded that J48 algorithm gives an accuracy rate of 74.8% while the naïve Bayes gives an accuracy of 79.5% by using 70:30 split. (Arora, R 2015 [2])
It aims to find and calculate the accuracy, sensitivity and specificity percentage of numerous classification methods and also tried to compare and analyse the results of several classification methods in WEKA, the study compares the performance of same classifiers when implemented on some other tools which includes Rapidminer and Matlabusing the same parameters (i.e. accuracy, sensitivity and specificity). They applied JRIP, Jgraft and BayesNet algorithms. The result shows that Jgraft shows highest accuracy i.e 81.3%, sensitivity is 59.7% and specificity is 81.4%. It was also concluded that WEKA works best than Matlab and Rapidminner.( Bamnote 2016 [3])
focus on applying a decision tree algorithm named as CART on the diabetes dataset after applying the resample filter over the data. The author emphasis on the class imbalance problem and the need to handle this problem before applying any algorithm to achieve better accuracy rates. The class imbalance is a mostly occur in a dataset having dichotomous values, which means that the class
variable have two possible outcomes and can be handled easily if observed earlier in data preprocessing stage and will help in boosting the accuracy of the predictive model.( D.K., Paul 2016 [4])
3. METHODOLOGY
In this section we shall learn about the various classifiers used in machine learning to predict diabetes. We shall also explain our proposed methodology to improve the accuracy. Five different methods were used in this paper. The different methods used are defined below. The output is the accuracy metrics of the machine learning models.
Then, the model can be used in prediction.
Dataset Description
The diabetes data set was originated from
https://www.kaggle.com/johndasilva/diabetes. [9] Diabetes dataset containing 2000 cases. The objective is to predict based on the measures to predict if the patient is diabetic or not.
  Fig-1 The diabetes data set https://www.kaggle.com/johndasilva/diabetes.
  .
Fig-2 The Mean of the data

      Preprocess Data
    Dataset
Applied Algorithm
Fig-3 Proposed Model Diagram
   Performance Evaluation on Various Measures
         Comparative Analysis Based on Accuracy
  Result
    3. RESULT & DISCUSSION
Correlation Matrix:
 Histogram:
Fig-5 Histogram of Diabetes dataset
https://www.researchgate.net/figure/Representat ion-of-Histograms-for-Diabetes- dataset_fig3_348638615
It shows how each feature and label is distributed along different ranges, which further confirms the need for scaling. Next, wherever you see discrete bars, it basically means that each of these is actually a categorical variable. We will need to handle these categorical variables before applying Machine Learning.[5]
      Fig-4 Correlation matrix of PIMA dataset
https://www.researchgate.net/figure/Correlation -matrix-of-PIMA-Dataset_fig1_343169138
It is easy to see that there is no single feature that has a very high correlation with our outcome value. Some of the features have a negative correlation with the outcome value and some have positive.
  
 Bar Plot For Outcome Class
 Fig-6
https://www.researchgate.net/publication/347091823_Di abetes_Prediction_Using_Machine_Learning
The above graph shows that the data is biased towards datapoints having outcome value as 0 where it means that diabetes was not present actually. [6]
K-Nearest Neighbors
The k-NN algorithm is arguably the simplest machine learning algorithm. Building the model consists only of storing the training data set. To make a prediction for a new data point, the algorithm finds the closest data points in the training data set, its “nearest neighbors.”
   Fig-7
https://www.researchgate.net/publication/347091823_Di abetes_Prediction_Using_Machine_Learning
   Considering if we choose one single nearest neighbor, the prediction on the training set is perfect. But when more neighbors are considered, the training accuracy drops, indicating that using the single nearest neighbor leads to a model that is too complex. The best performance is somewhere around 9 neighbors.
 Training Accuracy
 Testing Accuracy
Logistic regression:
0.81 0.78
  Table-1
Logistic Regression is one of the most common classification algorithms.
 Training Accuracy
C=1
0.779
C=0.01
0.784
   C=100
0.778 Table-2
Testing Accuracy
0.788 0.780 0.792
    In first row, the default value of C=1 provides with 77% accuracy on the training and 78% accuracy on the test set.
 In second row, using C=0.01 results are 78% accuracy on both the training and the test sets.
 Using C=100 results in a little bit lower accuracy on the training set and little bit highest accuracy on the test set, confirming that less regularization and a more complex model may not generalize better than default setting.
 Therefore, we should choose default value C=1. [7]
 
 Decision Tree:
This classifier creates a decision tree based on which, it assigns the class values to each data point. Here, we can vary the maximum number of features to be considered while creating the model.
Table-3
The accuracy on the training set is 100% and the test set accuracy is also good.
Feature Importance in Decision Trees
Feature importance rates how important each feature is for the decision a tree makes. It is a number between 0 and 1 for each feature, where 0 means “not used at all” and 1 means “perfectly predicts the target”.[8]
  Fig-8
https://www.researchgate.net/publication/347091823_Di abetes_Prediction_Using_Machine_Learning
Feature “Glucose” is by far the most important feature.
Random Forest:
This classifier takes the concept of decision trees to the next level. It creates a forest of trees where each tree is formed by a random selection of features from the total features.
    Table-4
Feature importance in Random Forest:
Fig-9
https://www.researchgate.net/publication/347091823_Di abetes_Prediction_Using_Machine_Learning
Similarly to the single decision tree, the random forest also gives a lot of importance to the “Glucose” feature, but it also chooses “BMI” to be the 2nd most informative feature overall.
Support Vector Machine:
This classifier aims at forming a hyper plane that can separate the classes as much as possible by adjusting the distance between the data points and the hyper plane. There are several kernels based on which the hyper plane is decided. I tried four kernels namely, linear, poly, rbf, and sigmoid.
Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.(DL Singh 2012 [8])
   
   Decision Tree:
This classifier creates a decision tree based on which, it assigns the class values to each data point. Here, we can vary the maximum number of features to be considered while creating the model.
1.00 Testing AccuFriagc-y10 0.99
https://www.researchgate.net/publication/347091823_Di
 Training Accuracy
   Table-3
 abetes_Prediction_Using_Machine_Learning
The accuracy on the training set is 100% and the test set accuracy is also good.
 As can be seen from the plot above, the linear kernel performed the best for this dataset and achieved a score
Feature Importance in Decision Trees
of 77%.
AccFueratcuyreCiomppoartraisnocne rates how important each feature is
for the decision a tree makes. It is a number between 0
means “not used at all”
and 1 for each feature, where 0
  Algorithms Training
and 1 means “perfectly predicts
Accuracy
 k-Nearest Neighbors
 81%
 Logistic Regression
 78%
 Decision Tree
  98%
 Random Forest
 94%
   Testing
the target”.
Accuracy 78%
      SVM
78%
99%
97%
76% 77% Table-5
   Feature “Glucose” is by far the most important feature.
  Random Forest:
4. CONCLUSIONANDFUTUREWORK
This classifier takes the concept of decision trees to the next
One of the important real-world medical problems is the
level. It creates a forest of trees where each tree is formed by
detection of diabetes at its early stage, if we are
a random selection of features from the total features.
successfully able to detect diabetes at its early stage then
we can successfully able to prevent many disease. In this
study, systematic efforts are made in designing a system
 Training Accuracy 1.00
which results in the prediction of diabetes. During this
 work, five machine learning classification algorithms are
studied and evaluated on various measures. Experiments
Testing Accuracy 0.974
are performed on john Diabetes Database. Experimental
  results determine the adequacy of the designed system
with an achieved accuracy of 99% using Decision Tree
Feature importance in Random Forest:
algorithm.
In future, the designed system with the used machine learning classification algorithms can be used to predict or diagnose other diseases. The work can be extended and improved for the automation of diabetes analysis including some other machine learning algorithms.
Similarly to the single decision tree, the random forest also gives a lot of importance to the “Glucose” feature, but it also chooses “BMI” to be the 2nd most informative feature overall.
Support Vector Machine:
This classifier aims at forming a hyper plane that can separate the classes as much as possible by adjusting the distance between the data points and the hyper plane. There are several kernels based on which the hyper plane is decided. I tried four kernels namely, linear, poly, rbf, and sigmoid.
 
5. REFERENCES
[1]. Aljumah, A.A., Ahamad, M.G., Siddiqui, M.K., 2013. Application of data mining: Diabetes health care in young and old patients. Journal of King Saud University - Computer and Information Sciences 25, 127–136. doi:10.1016/j.jksuci.2012.10.003. https://www.researchgate.net/publication/2337 53511_Application_of_data_mining_Diabetes _health_care_in_young_and_old_patients
[2]. Arora, R., Suman, 2012. Comparative Analysis of Classification Algorithms on Different Datasets using WEKA. International Journal of Computer Applications 54,(2015) 21–25. doi:10.5120/8626-2492. https://www.sciencedirect.com/science/article/ pii/S1877050918308548
[3]. Bamnote, M.P., G.R., 2014. Design of Classifier for Detection of Diabetes Mellitus Using Genetic Programming. Advances in Intelligent Systems and Computing 1,(2016) 763–770. doi:10.1007/978-3319-11933-5. https://link.springer.com/chapter/10.1007/978- 3-319-11933-5_86
[4]. Choubey, D.K., Paul, S., Kumar, S., Kumar, S., 2017. Classification of Pima indian diabetes dataset using naive bayes with genetic algorithm as an attribute selection, in: Communication and Computing Systems: Proceedings of the International Conference on Communication and Computing System (ICCCS 2016), pp. 451– 455. https://www.researchgate.net/publication/3138 06910_Classification_of_Pima_indian_diabete s_dataset_using_naive_bayes_with_genetic_al gorithm_as_an_attribute_selection
[5]. Dhomse Kanchan B., M.K.M., 2016. Study of Machine Learning Algorithms for Special Disease Prediction using Principal of Component Analysis, in: 2016 International Conference on Global Trends in Signal
Processing, Information Computing and Communication, IEEE. pp. 5–10.
https://www.researchgate.net/publication/3179 15046_Study_of_machine_learning_algorithm s_for_special_disease_prediction_using_princi pal_of_component_analysis
[6]. Sharief, A.A., Sheta, A., 2014. Developing a Mathematical Model to Detect Diabetes Using Multigene Genetic Programming. International Journal of Advanced Research in Artificial Intelligence https://www.researchgate.net/publication/2677 94039_Developing_a_Mathematical_Model_t o_Detect_Diabetes_Using_Multigene_Genetic _Programming
[7]. Sisodia, D., Shrivastava, S.K., Jain, R.C., 2010. ISVM for face recognition. Proceedings - 2010 International Conference on Computational Intelligence and Communication Networks, https://www.researchgate.net/publication/2242 15230_ISVM_for_Face_Recognition
CICN 2010 , 554– 559doi:10.1109/CICN.2010.109.
[8]. Sisodia, D., Singh, L., Sisodia, S., 2014. Fast and Accurate Face Recognition Using SVM and DCT, in: Proceedings of the Second International Conference on Soft Computing for Problem Solving (SocProS 2012), December 28-30, 2012, Springer. pp. 1027– 1038. https://www.researchgate.net/publication/2791 38750_Fast_and_accurate_face_recognition_u sing_SVM_and_DCT
[9]. For the dataset
                    https://www.kaggle.com/johndasilva/diabetes
.
     
