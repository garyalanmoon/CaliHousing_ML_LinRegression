# CaliHousing_ML_LinRegression
This is my first project as I seek to improve my skills and understanding in the field of ML. 

This program trains and tests a linear regression model for predicting housing prices in California. For the data, I used the well-known California Housing Prices dataset from Scikit-learn. A few words are in order abut the dataset. The dataset is based on data from the 1990 U.S. Census. Each row corresponds to a Census block group. A block group is the smallest geographical area about which the U.S. Census Bureau publishes sample data. The average population of a block group is 600 to 3,000 people. The dataset contains the following data for each block group:
MedInc = median income in the block group
HouseAge = median house age in the block group
AveRooms = average number of rooms per household in the block group
AveBedrms = average number of bedrooms per household in the block group
Population = population of the block group
AveOccup = average number of household members in the block group
Latitude = latitude of block group
Longitude = longitude of block group
MedHouseVal = median value of a house in the block group

Not understanding how the dataset was formed may make some of the data look odd. For instance, you may see a row regarding the number of bedrooms and it is not an integer. This would be odd if the data was on individual house sales. However, given that it is an average across block groups this is to be expected. In addition, the average number of rooms and/or bedrooms can sometimes take surprisingly large values. This can happen if there is a block group with a relatively low population (number of households) and a large number of houses (e.g., lots of vacant homes). Another important thing to note is that MedHouseVal is capped at $500k. So, any median value >= $500k is recorded as $500k (or 5.0 in the dataset).

Now that we've understood the dataset, let's discuss what the program does. From here, I'll outline things in steps.

# (1) Obtain and preprocess the data
The program first fetches the data and then converts the data from a numpy array to a dataframe. Note that I assigned the label 'Price' to the target variable MedHouseVal. Next, we check the dataframe for missing values and NaNs. In this case, there were neither, so we did not need to take further action there. Next, the program generates some summary statistics as well as a correlation heat map to give some understanding of how the different features and the target variable are related.

# (2) Define features/target and split the data
The next step is to define the features and the target. This is already nicely split in the dataset. The median house value (or price in the program) is the target. All other variabls are features. Once we've split into features/target, we split the dataset into training data (80% of the dataset) and testing data (20% of the dataset). 

# (3) Initialize the model, train the model and generate predictions
Now, we first initialize the model. We use a simple linear regression model. The next step is to train the model on the training data. Once the model has been trained, we generate predicted house prices using the features in the testing dataset.

# (4) Evaluation of the model's performance
At this stage, we want to evaluate the model's performance. To do so, we compute the root mean squared error and the R^2. Further, we also inlcude some visualizations, namely actual vs predicted prices and the residuals. The RMSE and R^2 were decent for a first project, but probably nothing to write home about. However, when we look at the actual vs predicted prices, we can clearly see that the model struggles pricing the most expensive houses. This is likely due, in no small part, to the artificial cap at $500k on the housing prices. If we look at a histogram of the house prices, we see a large spike at $500k where many expensive homes would be lumped together. As a result, I decided that this dataset may be more appropriate for modeling housing prices for homes with values below $500k.

# (5) Filtering the data and rerunning the model
Having seen the issues the model had at the upper end of the price ranges, I decided to repeat the above process after filtering out all houses priced at $500k. One thing to note here is that we are thus decreasing the size of the dataset. Given that ML programs tend to be rather greedy for data, this could pose problems. Doing so led to an improvement in the model:
RMSE_unfiltered = 0.74558 and RMSE_filtered = 0.64286. Given the units of the price variable ($100k), this corresponds to an improvement of about $10k in the average root-squared error. Interestingly, the R^2 coefficient actually decreased a bit. This could, in part, be true because we've decreased the variance in the data and the residual sum of squares did not decrease sufficiently to compensate. It's also possible that the smaller dataset played a role. In dropping block groups with average house price of $500k, I dropped ~1000 block groups representing about 5% of the data. Linear regression models tend to benefit from having more data and a ~5% reduction in the size of the training dataset could have had negative implications for the model.

# Conclusion

In the end, I think that filtering the dataset led to a win given the non-trivial reduction of the RMSE. In order to price a wider variety of homes, it would probably be necessary to get a different set of data (e.g., one without the artificial cap at $500k). In terms of how we might make better predictions about this dataset, I think we get some clue from the R^2. Note that the R^2 remained rather less-than-stellar, even after filtering out the problematic price-capped block groups. This suggests that there may be some nonlinear effects at play. It would thus be of interest to try a nonlinear model. I think that polynomial regression is probably the next logical step. It could be that the relationships involved are too complex or non-smooth for polynomial regression to capture effectively. If so, from there, it would be worth trying either decision tree or random forest regression, perhaps.

