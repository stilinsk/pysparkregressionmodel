# pyspark  big data
When dealing with big daa(big amaount of data) using the normal libraries can be very tiring and very time consuming may even take days to run and also the computing accuracy may not be that good that why we need to use a frameowrk known as pyspark,dont worry if you have never used pyspark before in the following notebook we will be implementing 
a begginers to advanced guide in pyspark regression ,ingesion of data,eda using pyspark  building several regression models and fine tuning such models.I will take you through all these.
### Project overview

In the following notebook we will be predicting the prices of houses in paris based on;

1.SquareMetres 

2. Number of rooms

3. Whether it has a yard or not

4.Whether has a pool or not
  
5. Number of floors

6.city code

7CityPartRange

8. Number of previous owners
   
9.is new built
 
10. Whether has a storm Protector or not

11. Whether has a basement or not

 12. Whether has a garage
          
 13. Whether has a storage room or not

 14. Has a guest room

We will be predicting the cost of our house using pyspark and the price is our target variable.

### creating a spark session and data  ingestion
for one to create a spark session we will need to configure the environment with java unless you a re using ggogle colab then you will need to configure it.then we will create a spark session using the following command
```
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('housing').getOrCreate()
house_df = spark.read.csv('ParisHousing.csv', header = True, inferSchema = True)
house_df.printSchema()
```
### EDA
We will look at the ditribution of our data this is normalluy the command we use in sklearn `df.describe` in spark we use
`house_df.describe().toPandas()`

we will then look at the **correlation** of our indipendent varibales to our dependent varibaleand look at how each individual column is related to out=r output varibale
```
import six
for i in house_df.columns:
    if not( isinstance(house_df.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to for price ", i, house_df.stat.corr('price',i))
```

*Correlation Values for Price*

- Price vs. Square Meters: 0.9999993570640745
- Price vs. Number of Rooms: 0.009590905935479128
- Price vs. Has Yard: -0.006119244882540526
- Price vs. Has Pool: -0.00507034083386251
- Price vs. Floors: 0.0016542562406504926
- Price vs. City Code: -0.001539367348580816
- Price vs. City Part Range: 0.008812911660535336
- Price vs. Number of Previous Owners: 0.016618826067943387
- Price vs. Made: -0.007209526254690673
- Price vs. Is New Built: -0.010642774359518865
- Price vs. Has Storm Protector: 0.0074959113342807394
- Price vs. Basement: -0.003967482178851144
- Price vs. Attic: -0.000599514077496332
- Price vs. Garage: -0.017229051207338166
- Price vs. Has Storage Room: -0.0034852993013792864
- Price vs. Has Guest Room: -0.0006439241048174541
- Price vs. Price: 1.0
finally we look at the **types** of data in respective columns

`house_df.dtypes`


*Dataset Column Names and Data Types*

- squareMeters: int
- numberOfRooms: int
- hasYard: int
- hasPool: int
- floors: int
- cityCode: int
- cityPartRange: int
- numPrevOwners: int
- made: int
- isNewBuilt: int
- hasStormProtector: int
- basement: int
- attic: int
- garage: int
- hasStorageRoom: int
- hasGuestRoom: int
- price: double

Next we create a scatter matrix where all numeric features are plotted againist all the other numeric feaures
```
import pandas as pd
import matplotlib.pyplot as plt

# Your code to read the data and create the DataFrame goes here

numeric_features = [t[0] for t in house_df.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = house_df.select(numeric_features).sample(False, 0.8).toPandas()

scatter_matrix = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
plt.show()
```
![a](https://github.com/stilinsk/pysparkregressionmodel/assets/113185012/b5400bd4-6b84-48ce-9b5e-8b58740f66f1)



It identifies the numeric columns in the DataFrame by checking their data types. It creates a list called numeric_features that contains the names of these numeric columns. It samples 80% of the rows from the DataFrame df using the sample method with a sampling fraction of 0.8. The sampled data is converted to a Pandas DataFrame using the
toPandas() method. It creates a scatter matrix plot using pd.plotting.scatter_matrix function. The scatter matrix plot is a grid of scatter plots where each numeric feature is plotted against all other numeric features. It provides a visual representation of the pairwise relationships between the variables.

### data preprocessing and splitting  for model building
We will create a vector assembler for our modelto handle our nueric columns

The VectorAssembler class is imported from the pyspark.ml.feature module. An instance of VectorAssembler is created with the following parameters: inputCols: A list of column names representing the input features. In this case, the list contains multiple column names such as 'squareMeters', 'numberOfRooms', 'hasYard', and so on. outputCol: The name of the output column where the assembled vector will be stored. In this case, it's set to 'features'. The transform method of the vectorAssembler object is called on the house_df DataFrame. It transforms the DataFrame by adding a new column named 'features', which contains the assembled vector of input features. The take(1) method is called on the transformed DataFrame vhouse_df. This retrieves the first row of the DataFrame as a list.

The VectorAssembler is commonly used in Spark machine learning pipelines to prepare features for model training. By combining multiple input columns into a single vector column, it enables easier processing and compatibility with Spark ML algorithms that expect input in vector form.
```
from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols = 
['squareMeters', 'numberOfRooms', 'hasYard', 'hasPool', 'floors', 'cityCode', 'cityPartRange', 'numPrevOwners', 'made', 'isNewBuilt', 'hasStormProtector', 'basement', 'attic', 'garage', 'hasStorageRoom', 'hasGuestRoom'
], outputCol = 'features')
vhouse_df = vectorAssembler.transform(house_df)

vhouse_df.take(1)
```
*[Row(squareMeters=75523, numberOfRooms=3, hasYard=0, hasPool=1, floors=63, cityCode=9373, cityPartRange=3, numPrevOwners=8, made=2005, isNewBuilt=0, hasStormProtector=1, basement=4313, attic=9005, garage=956, hasStorageRoom=0, hasGuestRoom=7, price=7559081.5, features=DenseVector([75523.0, 3.0, 0.0, 1.0, 63.0, 9373.0, 3.0, 8.0, 2005.0, 0.0, 1.0, 4313.0, 9005.0, 956.0, 0.0, 7.0]))]*

```
column_list = house_df.columns
print(column_list)
```
*['squareMeters', 'numberOfRooms', 'hasYard', 'hasPool', 'floors', 'cityCode', 'cityPartRange', 'numPrevOwners', 'made', 'isNewBuilt', 'hasStormProtector', 'basement', 'attic', 'garage', 'hasStorageRoom', 'hasGuestRoom', 'price']*

#### splitting the data
```
splits = vhouse_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]
```
### Model Building
#### Linear regression
The LinearRegression class is imported from the pyspark.ml.regression module. An instance of LinearRegression is created with the following parameters: featuresCol: The name of the input column that contains the assembled feature vector. In this case, it's set to 'features'. labelCol: The name of the column that contains the target variable or label. In this case, it's set to 'price'. maxIter: The maximum number of iterations for the optimization algorithm. It's set to 10 in this case. regParam: The regularization parameter for controlling overfitting. It's set to 0.3, which determines the strength of the regularization. elasticNetParam: The mixing parameter between L1 and L2 regularization. It's set to 0.8, indicating a higher emphasis on L1 regularization (LASSO). The fit method of the lr object is called on the train_df DataFrame. It fits the linear regression model to the training data and returns a LinearRegressionModel object. The coefficients of the trained linear regression model are printed using lr_model.coefficients. The intercept of the trained linear regression model is printed using lr_model.intercept.
```
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features', labelCol='price', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
```
*Coefficients: [99.9998604435825,0.13189783477533978,2971.0136471264227,2975.1116906751704,55.152661125550274,-0.0011270866020865143,47.13067701099789,7.6812244009179285,-2.5174377046591854,155.6078203686164,139.20167334600671,-0.00113603675428869,-0.012859433418450786,0.03508281364200009,-23.11352883743656,-6.2383816051051815]
Intercept: 5407.045327011325*

lets look at the rmse and the r2 score
```
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
```
*RMSE: 1889.040452*
*r2: 1.000000*
We will look at the r2 of our test data
```
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","price","features").show(5)

from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="price",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
```
```
R Squared (R2) on test data = 1
Our model is perfoming at its best very rare chances of this but the perfomance of the data on the test and train data is remarkable. The model is perfominga at 100%
```
We will also look at the rmse of the test data
```
test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
```

### Decison Tree Regressor
```
from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'price')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
```

*Root Mean Squared Error (RMSE) on test data = 90565.5*

Lets look athe r2 score of both the test and train data
```

from pyspark.ml.evaluation import RegressionEvaluator

# Assuming you already have the required imports and have trained the DecisionTreeRegressor

# Evaluate R2 score on train data
train_predictions = dt_model.transform(train_df)
train_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="r2")
train_r2 = train_evaluator.evaluate(train_predictions)
print("R2 score on train data = %g" % train_r2)

# Evaluate R2 score on test data
test_predictions = dt_model.transform(test_df)
test_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="r2")
test_r2 = test_evaluator.evaluate(test_predictions)
print("R2 score on test data = %g" % test_r2)
```

*R2 score on train data = 0.999016*

*R2 score on test data = 0.999027*

We can clearly see that our model is perfoming fairly better even whnen using the decision tree regressor thus we wont need to even tune our model as its perfoming in the best way possible

### Gradient Boosting Regressor
```

from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'price', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'price', 'features').show(5)
```

Lets evaluate its perfomance
```
from pyspark.ml.evaluation import RegressionEvaluator

# Assuming you have already imported necessary modules and fitted the GBTRegressor

# Evaluate R2 score on train data
train_predictions = gbt_model.transform(train_df)
train_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="r2")
train_r2 = train_evaluator.evaluate(train_predictions)
print("R2 score on train data = %g" % train_r2)

# Evaluate R2 score on test data
test_predictions = gbt_model.transform(test_df)
test_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="r2")
test_r2 = test_evaluator.evaluate(test_predictions)
print("R2 score on test data = %g" % test_r2)
```
*R2 score on train data = 0.999088*

*R2 score on test data = 0.999016*

Mostly this is not alwya]s the case it does not mean that pysprk models are the best  its perfomance is greatly enhanced due to the fact we are using pyspark on a fairly small data ,but basivcally thats how the regression tasks are done with spark
