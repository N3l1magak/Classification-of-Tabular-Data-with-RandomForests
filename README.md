# Project1: Classification of Tabular Data with RandomForests
Project for CS_6364 Machine Learning

## Description

Analyze a dataset of 12,000 instances of user history for online shopping sessions over the course of a year, in order to build a binary classifier that predicts whether or not the user ended up buying something during that time.

## Getting Started
### Part 1: Dataset download and extraction
This dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset), where you will find a description of the dataset, including its features. A [copy](https://github.com/N3l1magak/Classification-of-Tabular-Data-with-RandomForests/blob/main/online_shoppers_intention_cs4364.csv) of simplified dataset is provided. In order to compare results, we're going to use rows 10000 - 12331 as our holdout dataset. 

**1. Load csv correctly into a DataFrame and show contents in a cell**

![image](https://user-images.githubusercontent.com/105015948/169902204-69854d98-5c8e-4672-99bd-16354319ae22.png)

**2. Holdout dataset split as specified**

![image](https://user-images.githubusercontent.com/105015948/169902440-059f654a-6acd-410c-9b01-099553cd7e11.png)

**3. Explanation of generalization from such holdout split:**

The split is not a good idea from the perspective of generalization. Because when we are doing splits, it should be follow the principle of random assignment. Whereas the dataframe is not totally random placed. E.g., for the "Month" feature, "Feb" is mostly in the first 200 rows, and from 10000 to 12330 only contains "Nov" and "Dec". Thus the model trained would work poorly and predict badly on the holdout set.

### Part 2: Data cleaning

**4. Use `value_counts()` in `pandas` to print out the distributions of the categorical and ordinal numbered features (treat `SpecialDay` as categorical here). Turn on the setting to reveal missing data -- how many features, and what percent of them, were missing?**

![image](https://user-images.githubusercontent.com/105015948/169904742-377eecf5-cd20-4ee6-8b65-7192515a1759.png)

As shown above, only "ProductRelated" feature has 2.717% of data missing.

**5. Use the `describe()` method in `pandas` to print out summary statistics. Which features need to be considered more carefully, based on these results?**

![image](https://user-images.githubusercontent.com/105015948/169905070-0c1ac452-3676-4bda-9c02-17a0e639a11f.png)

"Administrative_Duration", "Informational_Duration" and "ProductRelated_Duration" should be considered more carefully as they have a larger standard deviation, which means the data is distributed more scattered.

**6. Handle any missing data in training and holdout dataset, but do not simply delete the rows.**

![image](https://user-images.githubusercontent.com/105015948/169905398-9bb874a7-497b-4ae9-a152-c0b745c5b96e.png)

Since the missing value only contains less than 3% of total data in the feature, replacing them with the most frequently exist value would not affect too much when the model is learning.

**whether or not need to scale/normalize the features, and which ones, if any.**

For numerical features like: "Administrative_Duration" has a max value of 3398; "Informational_Duration" has a max value of 2549; "ProductRelated" has a max value of 705; "ProductRelated_Duration" has a max value of 63973; "PageValues" has a max value of 361. These features need to be sacled since their range is wide.

**There are several categorical features. Discuss and implement if encode them as ordinal numbers, or one-hot encode them.**

The "Month" and "VisitorType" should be convert into numerical(binary) value, and they should be one-hot encoded. The "Revenue" and "Weekend" features should be converted into binary for the ease of calculation. (Already implemented in cell 3)

![image](https://user-images.githubusercontent.com/105015948/169905969-07d59caa-39ed-406d-9e79-852fe7d325e3.png)

The "Special_Day" feature should be categorical into a "is Special_Day" or "is_not_Special_Day". As almost 90% of the value in the data sample is 0.
Besides, the "Page_Value" feature also have a imbalanced distribution (around 77% is 0), thus I think it can also be transformed to categorical.

**Use a heatmap to show the correlation between all feature pairs.**

![image](https://user-images.githubusercontent.com/105015948/169906276-b8b8fc61-996a-4759-ade7-1a81acb4990b.png)

From the graph plotted, I would consider "TrafficType" as a feature to drop. Since it has the least correlation score with the target "Revenue", and the correlation with other features also remain low (max: 0.2 with OS, which makes sense since traffictype is highly related to OS). By dropping this feature, it might reduce nosie of data, and the model might have a better training result.

**engineer one additional feature.**

![image](https://user-images.githubusercontent.com/105015948/169906487-379893a0-b522-4e86-bc66-dc0aef40486c.png)

Since in "OperatingSystems" feature, value 2 is responsible for around 6600 examples, and value 5 ~ 8 only responsible for a relatively small number of sample, we could combine 5 to 8 into 5 only, thus the random forest would be faster as less separation exist.

**Separate training data into features and labels**

![image](https://user-images.githubusercontent.com/105015948/169906603-868d93de-404b-4476-b0cc-337bdf5e0931.png)

The labels for this dataset are highly imbalanced. Use under-sampling to randomly choose out 1440 with "Revenue" of value 0 from 8559 examples.

![image](https://user-images.githubusercontent.com/105015948/169906706-4c53749c-3178-4275-a2dd-26a360b7328a.png)

**Instantiate a `RandomForest` model. Define a grid to tune at least three different hyperparameters with at least two different values each.**

![image](https://user-images.githubusercontent.com/105015948/169906932-9816a04e-f9eb-40a9-87c4-0153cb65c742.png)

**Set up a `gridsearchCV` with 5-fold cross validation.**

![image](https://user-images.githubusercontent.com/105015948/169907043-1273d108-4970-4b03-8b80-f6914ff9e018.png)

**Train the model using `gridsearchCV`, and report the best performing hyperparameters. Calculate accuracy, precision and recall on the holdout dataset.**

![image](https://user-images.githubusercontent.com/105015948/169907246-05286927-24bd-47f2-a646-3e8dd83299b2.png)

In our case, the recall is more important, as a company, we want our profit maximized, so we do not care about "false possitive", but "false negative" (which means those who would purchase but missed by model).

As shown above, using the validation set (X_test,y_test), the recall acurracy is around 90%. for holdout set (X_holdout, y_holdout), the recall accuracy is around 80%. Since the scores different by 10% (and relatively high) ,so I think the model does overfit a little, but it still has a considerably good score (near 80%).

**Generate a confusion matrix**

![image](https://user-images.githubusercontent.com/105015948/169907358-ae3a266e-da0d-4ed9-81d7-08a60bfb2c46.png)

* For group 1 (the first vector in the matrix, represent "will not buy"), rate of correctness is around 81%
* For group 2 (the second vector in the matrix, represent "Will Buy"), rate of correctness is around 78% (which is the recall)

**Print out the feature importances**

![image](https://user-images.githubusercontent.com/105015948/169907506-b2b7b74a-cf95-4475-af70-a15e14727434.png)

* Feature 8 (PageValues) is the highest among all

**Train and tune another decision-tree based model on the training dataset. Using the best performing hyperparameters, test this model with holdout.**

![image](https://user-images.githubusercontent.com/105015948/169907695-5f5cd86c-7ab3-45a4-8291-0910b5ec29d3.png)

The Gradient Boosting Classifier results slightly lower accuracy than the Random Forest Classifier. Since the difference between the prediction based on validation set and holdout set is 10%, I think the the model does overfit (since the split of dataset is problematic), and result can not be considered as generalized.

**repeat training and tuning on the same data with a `LogisticRegression` model.**

![image](https://user-images.githubusercontent.com/105015948/169907866-9cb356ff-5d50-4f34-8b35-6593798b9228.png)

### Before executing ipynb

the following libraries are needed:
* pandas
* Seaborn
* numpy
* matplotlib.pyplot
* sklearn
* imblearn

## Author

Reynolds_Z @ 2022

