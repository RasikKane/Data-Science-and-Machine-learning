# Feature Selection & Analysis in ML
The objective of this exercise is to assess the impact of feature selection on training and test
datasets. Two datasets accompany this assignment, heart-train.csv and heart-test.csv .
The idea is to identify a good feature subset using the training dataset and test this subset
on the test data. In preparing your submission, you should focus on explaining discrepancies
between train and test performance rather than maximising performance.

## Requirements
* Use Gradient Boosting as your classifier.
* As a baseline, report performance (accuracy) on the train and test data using all features. The results on the training data can be based on cross validation. The results on the test data can be hold-out, i.e. train on train data and test on test data.
* Using a feature subset selection method of choice; identify a feature subset that is expected to generalise well for this task. Test the performance of this feature subset on the test data.
* At no stage the test data is used in classifier training or in feature selection.
* stability and consistency of the results is discussed in inference.

# Implementation

## Gradient Boosting algorithm - Baseline

Gradient Boosting works on principal of "Ensemble of Decision Tree". **Gradient Boosting Classifier implementeation in scikit-learn supports both binary and multi-class classification.**
* GB builds an additive model of weak decision trees in a forward stage-wise fashion
* The weak decison tree is successively Optimized using arbitrary differentiable loss functions.
* In each stage, n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. **But, addressed classification problem is Binary, hence it has induced only a single regression tree**

<br>

## Feature Selection

In Feature selection we need to find the best subset of all available features, which contains the smallest number of features which contribute most to the accuracy. Discard the remaining, unimportant features.

**Feature Selection Strategies**
* **Filter**: During Pre-processing step, filters evaluates the importance of features based only on their inherent characteristics, without incorporating any learning algorithm which is applied subsequently.
    * Evaluation is based on multiple statistical parameters like Gini Index, Chi-square index etc. 
    * They are extensively used on high dimensional data where wrapper methods have a prohibitive computational cost.
* **Embedded methods**: This is typically implemented by using a sparsity regularizer orconstraint which makes the weight of some features become zero. eg. Sparse Multinomial Logistic Regression, Automatic Relevance Determination Regression, LASSO, Ridge regression.
* **Wrapper**
    * **Sequential feature selection**: The classifier is “wrapped” in the feature selection mechanism. Feature subsets are evaluated directly based on their performance when used with that specific classifier. A key aspect of wrappers is the search strategy which generates the candidate feature subsets for evaluation.
        * **Sequential Forward Selection (SFS)**: This method starts with empty feature subset. In each step, algorithm adds  most informative feature to subset. This procedure repeats till model performance does not improve any further.
        * **Sequential Backward Eilimination (SBE)**: This method starts with whole feature space as subset. In each step, algorithm removes least informative feature. This procedure repeats till model performance does not improve any further. Hence, this is *feature selection by EXCLUSION*.  
        * **Floating Selection (FS))**: Floating selection is variant used with both above methods.
         
    * **Recurssive feature eliminator with cross validation** [[Ref](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV)]: scikit-learn implements RFECV : Feature ranking with recursive feature elimination and cross-validated selection of the best number of features. This is also a greedy - aproach algorithm which works on similar algorithm like *Sequential Backward Eilimination (SBE)*.     

### Recurssive feature eliminator with cross validation : Choosen for experiment
* Implement similar algorithm alike backward selection; BUT **AUTOMATES ADDITIONAL STEP TO LOCATE AND RETRAIN INNER MODEL WITH OPTIMAL k-features**

### Sequential Backward FLOATING Eilimination selection: Choosen for experiement
* It adds a recosideration stage to INCLUDE PREVIOUSELY REMOVED samples which may improve performance with PRESENT SAMPLE-SPACE. THIS METHOD INADVENTLY CONSIDERS POSTERIORI "Support" of features for each other. This is neglected in FILTER TYPES OF SELECTIONS.

## Inference

**FEATURE SELECTION METHOD**
* Based on the resultant graph,  resultant accuracies for training data using cross validation are consistent across all feature sets
* Accuracy of the classifier is most improved with Wrapper-based methods. 

**The discrepancies between performance on the training and test data** is due to two facts:
1. Training data is hardly twice the size of testing data. *This is reason behind smaller training accuracy compared to testing accuracy*
2. Sample size for both reaining and testing data is also relatively small [< 200]

*******
* Comparing wall times; Recursive feature selection(RFECV) is faster than Backward floating elimination (SBFS)
* **Wrapper based Sequential Forward Selection feature selection method is faster than backward elimination**. Due to option for early stopping, it can also be memory efficient.
* But, accuracy for Backward Elimination is high as it starts with large entire sample space removes less important features. As Given dataset is not huge, backward elimination would work.
* *Hence, For given dataset, Sequential Forward floating Selection is model of choice based on accuracy metric*
*******