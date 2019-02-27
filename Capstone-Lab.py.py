# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Data Science Capstone
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/airlift/AAHPtolqpaxIgZ0_m03xYBkcgBoyi6XDN2QB.png" style="width:1000px"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Airbnb has a host of public data sets for listings in cities throughout the world: <br>
# MAGIC http://insideairbnb.com/get-the-data.html
# MAGIC 
# MAGIC This challenge will be working on the data set for London.
# MAGIC 
# MAGIC Split into teams of 2 or 3 to work on completing this challenge.
# MAGIC 
# MAGIC **Metric:** We will be using RMSE to assess the effectiveness of our models.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Airbnb Price Prediction Challenge
# MAGIC 
# MAGIC 1. Configure Classroom
# MAGIC 1. Add the data set to Databricks.
# MAGIC 2. Read the Data
# MAGIC 2. Prepare the Data
# MAGIC 3. Define Preprocessing Models
# MAGIC 4. Split the Data for Model Development
# MAGIC 4. Prepare a benchmark Model
# MAGIC 5. Iterate on Benchmark Model
# MAGIC 6. Iterate on Best Model
# MAGIC 
# MAGIC A list of available regression models can be found here: https://spark.apache.org/docs/2.2.0/ml-classification-regression.html

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Configure Classroom
# MAGIC 
# MAGIC Run the following cell to configure our "classroom."

# COMMAND ----------

# MAGIC %run "../Includes/Classroom Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Read the Data
# MAGIC - Load the data as a Dataframe
# MAGIC - Prepare a basic description of the data set including:
# MAGIC    - the schema with data types
# MAGIC    - the number of rows

# COMMAND ----------

try:
  sasToken="?sv=2017-11-09&ss=bf&srt=co&sp=rl&se=2099-12-31T23:59:59Z"+\
    "&st=2018-01-01T00:00:00Z&spr=https&sig=di3x0sjVwmqIjO5ReQ%2Bwa54R9shTDZePtKHipkabqAg%3D"
  dbutils.fs.mount(
    source = "wasbs://class-453@airlift453.blob.core.windows.net/",
    mount_point = "/mnt/training-453",
    extra_configs = {"fs.azure.sas.class-453.airlift453.blob.core.windows.net": sasToken})
except Exception as e:
  if "Directory already mounted" in str(e):
    pass # Ignore error if already mounted.
  else:
    raise e
print("Success.")

# COMMAND ----------

filePath = "dbfs:/mnt/training-453/airbnb/listings/london-cleaned.csv"

initDF = (spark.read
  .option("multiline", True)
  .option("header", True)
  .option("inferSchema", True)
  .csv(filePath)
)

display(initDF)

# COMMAND ----------

initDF.printSchema()

# COMMAND ----------

initDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Prepare the Data
# MAGIC - Count rows with null values
# MAGIC - Impute missing values for numerical fields
# MAGIC - remove rows with null values for `zipcode`

# COMMAND ----------

# TODO: Count the number of rows in the `initDF` DataFrame

# COMMAND ----------

# TODO: Show the description of `initDF` DataFrame to shows the number of non-null rows

# COMMAND ----------

# TODO: Impute missing values for numerical columns

# COMMAND ----------

# TODO: Remove rows with null values for zip code

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Prepare for the Competition
# MAGIC 
# MAGIC - Using the same random seed select 20% of the data to be used as a hold out set for comparison between teams.
# MAGIC 
# MAGIC `modelingDF` will be used to prepare your model. You are free to use this data any way that you see fit in order to prepare the best possible model. **You must not expose your model to `holdOutDF`**.
# MAGIC 
# MAGIC `holdOutDF` will be used for comparison. You will submit scores for you model's performance on `holdOutDF` to the instructor for comparisonn.

# COMMAND ----------

seed = 273
(holdOutDF, modelingDF) = airbnbDF.randomSplit([0.2, 0.8], seed=seed)

print(holdOutDF.count(), modelingDF.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Define Preprocessing Models
# MAGIC 
# MAGIC Prepare the following models to be used in a modeling pipeline:
# MAGIC - Prepare a StringIndexer for `neighbourhood_cleansed`, `room_type`, `zipcode`, `property_type`, `bed_type`
# MAGIC - Prepare a OneHotEncoder for `neighbourhood_cleansed`, `room_type`, `zipcode`, `property_type`, `bed_type`

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

print(StringIndexer().explainParams())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now *StringIndex* all categorical features (`neighbourhood_cleansed`, `room_type`, `zipcode`, `property_type`, `bed_type`) and set `handleInvalid` to `skip`. Set the output columns to `cat_neighbourhood_cleansed`, `cat_room_type`, `cat_zipcode`, `cat_property_type` and `cat_bed_type`, respectively.

# COMMAND ----------

# TODO: Prepare a StringIndexer for `neighbourhood_cleansed`, `room_type`, `zipcode`, `property_type`, `bed_type`

# COMMAND ----------

# TODO: Prepare a OneHotEncoder for `neighbourhood_cleansed`, `room_type`, `zipcode`, `property_type`, `bed_type`

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Split the Data for Model Development
# MAGIC 
# MAGIC Let's keep 80% for the training set and set aside 20% of our data for the test set.
# MAGIC 
# MAGIC **NOTE:** The data is now split into three sets:
# MAGIC - `trainDF` - used for training a model
# MAGIC - `testDF` - used for internal validation of hyperparamters
# MAGIC - `holdOutDF` - used for final assessement of model and comparison to models prepared by other teams

# COMMAND ----------

# TODO: Perform a train-test split on `modelingDF`

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Prepare a Benchmark Model
# MAGIC 
# MAGIC - Define a `list` (Python) or `Array` (Scala) containing the features to be used. It is recommended to use the following features:
# MAGIC 
# MAGIC   `"host_total_listings_count"`, ` "accommodates"`, ` "bathrooms"`, ` "bedrooms"`, ` "beds"`, ` "minimum_nights"`, ` "number_of_reviews"`, ` "review_scores_rating"`, ` "review_scores_accuracy"`, ` "review_scores_cleanliness"`, ` "review_scores_checkin"`, ` "review_scores_communication"`, ` "review_scores_location"`, ` "review_scores_value"`, ` "vec_neighborhood"`, `"vec_room_type"`, `"vec_zipcode"`, `"vec_property_type"`, `"vec_bed_type"`
# MAGIC - Build a Linear Regression pipeline that contains:
# MAGIC   - each of the StringIndexers
# MAGIC   - the OneHotEncoder
# MAGIC   - a VectorAssembler
# MAGIC   - a LinearRegression Estimator
# MAGIC - Evaluate the performance of the Benchmark Model using a RegressionEvaluator

# COMMAND ----------

# TODO: Define a `list` (Python) or `Array` (Scala) containing the features to be used.

# COMMAND ----------

# TODO: Build a Linear Regression pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the performance of the Benchmark Model using a RegressionEvaluator on internal testing set, `testDF`.

# COMMAND ----------

# TODO: Evaluate the performance of the Benchmark Modelusing a RegressionEvaluator on internal testing set, `testDF`

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the performance of the Benchmark Modelusing a RegressionEvaluator on the class evaluation set, `holdOutDF`

# COMMAND ----------

# TODO: Evaluate the performance of the Benchmark Modelusing a RegressionEvaluator on the class evaluation set, `holdOutDF`

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Iterate on Benchmark Model
# MAGIC - Prepare a model to beat your benchmark model.
# MAGIC - Build a regression pipeline that contains:
# MAGIC    - each of the StringIndexers
# MAGIC    - the OneHotEncoder
# MAGIC    - a VectorAssembler
# MAGIC    - an improved Regression Estimator
# MAGIC  - Evaluate the performance of the new Model using a RegressionEvaluator on your internal testing set, `testDF`.
# MAGIC  - Use the internal testing set to adjust the hyper parameters of your model.
# MAGIC  - Evaluate the performance of the new Model using a RegressionEvaluator on the class evaluation set, `holdOutDF`
# MAGIC  - When you have beaten the benchmark, share the results with your instructor.

# COMMAND ----------

# TODO: Build a better Regression pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the performance of the better Model using a RegressionEvaluator on internal testing set, `testDF`

# COMMAND ----------

# TODO: Evaluate the performance of the better Model using a RegressionEvaluator on internal testing set, `testDF`

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Use the internal test set to adjust the hyperparameters of your model.

# COMMAND ----------

# TODO: Build a better Regression pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the performance of the tuned Model using a RegressionEvaluator on internal testing set, `testDF`

# COMMAND ----------

# TODO: Evaluate the performance of the better Model using a RegressionEvaluator on internal testing set, `testDF`

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the performance of the tuned Model using a RegressionEvaluator on the class evaluation set, `holdOutDF`

# COMMAND ----------

# TODO: Evaluate the performance of the tuned Model using a RegressionEvaluator on the class evaluation set, `holdOutDF`

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2018 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
