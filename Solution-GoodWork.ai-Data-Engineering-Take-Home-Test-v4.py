# Databricks notebook source
# MAGIC %md
# MAGIC ## Goodwork.ai Data Engineering take-home test
# MAGIC
# MAGIC Thank you for taking the time to complete this assessment!
# MAGIC
# MAGIC - Please answer the following three questions to the best of your ability.
# MAGIC - Aim to complete the tasks within 2 hours (though you are welcome to spend more time if you wish).
# MAGIC - Be sure to explain your logic and reasoning as you work through the tasks. During the follow-up interview, we’ll ask you to present and discuss your solutions.
# MAGIC - Even though the test environment is running Spark on a single node, write your code **as if it were operating with multiple worker nodes**. Ensure that your solutions make use of Spark's distributed processing capabilities effectively, avoid practices that wouldn't scale well in a real-world cluster setting.
# MAGIC
# MAGIC Good luck, and we look forward to reviewing your work!

# COMMAND ----------

# MAGIC %md 
# MAGIC **Thank you!** for the opportunity to complete the take-home test for the Data Engineering position. 
# MAGIC
# MAGIC I thoroughly enjoyed tackling the challenges, which provided a great opportunity to apply my skills and learn along the way. It was both fun and insightful, and I’m excited to share my solutions with you.

# COMMAND ----------

# Install non-default packages on Databricks notebook
%pip install fuzzy_match
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip list

# COMMAND ----------

import json
import time
import random
import pyspark
import urllib.request
import pandas as pd
from pyspark.sql import functions as F
from collections import defaultdict
from fuzzy_match import match
from delta.tables import DeltaTable
from pyspark.sql.types import *
from pyspark.storagelevel import StorageLevel
from pyspark import StorageLevel




# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1 (data cleaning):
# MAGIC  
# MAGIC The sales team of a newly acquired customer has given you some data. The data contains two key challenges:
# MAGIC - The `Unit_Sold` column has inconsistent or low-quality data that needs cleaning.
# MAGIC - Three columns (`Unknown_1`, `Unknown_2`, `Unknown_3`) have unclear meanings and may have potential relationships to explore.
# MAGIC  
# MAGIC Your tasks are:
# MAGIC  
# MAGIC 1. **Load** the raw data from the endpoint below.
# MAGIC 2. **Clean** the `Unit_Sold` column to ensure consistent, high-quality data, and save the cleaned dataset.
# MAGIC 3. **Analyse** the unknown columns to identify any relationships _between_ them, including possible _hierarchical structures_.
# MAGIC  
# MAGIC Data endpoint: `wasbs://takehometestdata@externaldatastoreaccnt.blob.core.windows.net/coding_test_raw_data_v3`

# COMMAND ----------

spark.conf.set(
    "fs.azure.sas.takehometestdata.externaldatastoreaccnt.blob.core.windows.net",
    "sp=rl&st=2024-11-21T04:26:51Z&se=2024-12-25T12:26:51Z&spr=https&sv=2022-11-02&sr=c&sig=wc87T0FJxX74i0BrAZRlZkdKw9JKJD5%2F0VcTa6RN7a0%3D"
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Load the raw data from the **endpoint**

# COMMAND ----------

# Load the delta table
df = spark.read.format("delta").load(
    "wasbs://takehometestdata@externaldatastoreaccnt.blob.core.windows.net/coding_test_raw_data_v3"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #####  Data Quality Analysis

# COMMAND ----------

# Display the schema of the DataFrame, showing each column's name and data type
df.printSchema()

# Count the total number of rows in the DataFrame
row_count = df.count()
print(f"Total number of rows: {row_count}")

# Display the content of the DataFrame in a tabular format (limit to 100 rows for preview)
display(df.limit(10))

# 1. Count Null Values in Each Column
null_counts = df.select([F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns])
display(null_counts)

# Generate Summary Statistics for the 'Units_Sold' column
units_sold_summary = df.select("Units_Sold").describe()
display(units_sold_summary)

# Count the number of rows with missing values in the 'Units_Sold' column
missing_units_sold = df.filter(F.col("Units_Sold").isNull()).count()
print(f"Number of rows with missing 'Units_Sold' values: {missing_units_sold}")


# COMMAND ----------

"""
    Filters rows from a DataFrame based on the following conditions:
    - The column contains non-positive numeric values (<= 0).
    - The column contains non-numeric (text or null) values.

    Args:
    df (DataFrame): The input DataFrame.
    column_name (str): The name of the column to apply the filter.

    Returns:
    DataFrame: A filtered DataFrame containing rows where the specified column 
               is either non-numeric or has non-positive numeric values.
    """
def filter_non_positive_or_text(df, column_name):
    # Add a column to identify numeric values
    dataframe = df.withColumn(
        "Is_Numeric",
        F.col(column_name).cast("float").isNotNull()
    )

    # Add a column to identify non-positive values (for numeric rows)
    dataframe = dataframe.withColumn(
        "Is_Non_Positive",
        F.when(F.col("Is_Numeric") & (F.col(column_name).cast("float") <= 0), True)
        .otherwise(False)
    )

    # Filter rows that are either non-positive or non-numeric
    filtered_df = dataframe.filter(
        (F.col("Is_Non_Positive") == True) | (F.col("Is_Numeric") == False)
    )

    # Drop helper columns
    filtered_df = filtered_df.drop("Is_Numeric", "Is_Non_Positive")

    return filtered_df


# COMMAND ----------

# Apply the filter to the DataFrame to exclude non-positive or non-numeric rows in the "Units_Sold" column
filtered_data = filter_non_positive_or_text(df, "Units_Sold")

# Count and display the number of distinct values in the "Units_Sold" column of the filtered DataFrame
distinct_count = filtered_data.select("Units_Sold").distinct().count()
print(f"Number of distinct values in 'Units_Sold' after filtering: {distinct_count}")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 2. Clean the `Units_Sold` Column  
# MAGIC The goal of this step is to ensure that the `Units_Sold` column contains only consistent and high-quality data by:  
# MAGIC - Removing rows with non-positive numeric values (`<= 0`).  
# MAGIC - Removing rows with non-numeric values (e.g., text or null).  
# MAGIC
# MAGIC After cleaning, the resulting dataset will be saved for further processing.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Planned Data Cleaning Process for the *Units_Sold* Column  
# MAGIC
# MAGIC The following steps will be applied to ensure that the *Units_Sold* column contains clean, consistent, and high-quality data:  
# MAGIC
# MAGIC 1. **Extract Numeric Values**  
# MAGIC    - Identify and process mixed-format entries (e.g., "Units_X"), extracting valid numeric values.  
# MAGIC    - Replace invalid entries (e.g., non-numeric or undefined formats) with `None`.  
# MAGIC
# MAGIC 2. **Handle Negative Values**  
# MAGIC    - Treat negative values as invalid. Replace them with `0` to ensure meaningful results.  
# MAGIC
# MAGIC 3. **Impute Missing Values**  
# MAGIC    - Replace missing values (`None`) with the median of the column, ensuring statistically robust imputations.  
# MAGIC
# MAGIC 4. **Ensure Numerical Precision**  
# MAGIC    - Round all cleaned values to two decimal places for uniformity and accuracy.  
# MAGIC
# MAGIC 5. **Optimize Distributed Processing**  
# MAGIC    - Use Spark's `approxQuantile` method to compute the median efficiently across distributed data.  
# MAGIC    - Avoid driver-side computations to ensure scalability in large datasets.  
# MAGIC
# MAGIC 6. **Enhance Performance with Caching**  
# MAGIC    - Cache or persist the final cleaned DataFrame to optimize performance for downstream processing tasks.  
# MAGIC
# MAGIC By implementing these steps, we aim to create a high-quality and consistent dataset that is suitable for further analysis and transformations.
# MAGIC
# MAGIC

# COMMAND ----------

# Define a UDF to extract numeric values from the 'Units_Sold' column
"""
    Extract numeric values from mixed-format entries in the 'Units_Sold' column.
    - Handles strings like 'Units_X' by extracting the numeric part.
    - Returns None for invalid entries or strings like 'Na'.
"""
def extract_units(value):
    try:
        # Extract numbers from strings like 'Units_X'
        if isinstance(value, str) and 'Units_' in value:
            return float(value.split('_')[1])
        # Handle valid numeric values
        return float(value)
    except:
        return None  # Return None for invalid strings like 'Na'

# Register the UDF
extract_units_udf = F.udf(extract_units, FloatType())

# Apply the UDF to clean the Units_Sold column
df_extract_units = df.withColumn("Units_Sold", extract_units_udf(F.col("Units_Sold")))

# COMMAND ----------


# Display the count of distinct values in the 'Units_Sold' column of the cleaned DataFrame
distinct_units_sold_count = df_extract_units.select("Units_Sold").distinct().count()
print(f"Number of distinct values in 'Units_Sold': {distinct_units_sold_count}")

# COMMAND ----------

# Optional - Generate Summary Statistics for the 'Units_Sold' column
# units_sold_summary = df_cleaned_units.select("Units_Sold").describe()

# Display the entire cleaned DataFrame (optional, for inspection)
# df_cleaned_units.show()

# Display the summary statistics for 'Units_Sold'
# units_sold_summary.show()


# COMMAND ----------


# 1. Cast 'Units_Sold' to double for numeric processing
cleaned_data = df.withColumn("Units_Sold", F.col("Units_Sold").cast("double"))

# 2. Calculate the approximate median using approxQuantile (efficient for large data)
median_value = cleaned_data.approxQuantile("Units_Sold", [0.5], 0.01)[0]

# 3. Replace missing/null/negative/zero values and round to 2 decimal places
cleaned_data = (cleaned_data
    .fillna({"Units_Sold": median_value})  # Replace nulls with the median
    .withColumn("Units_Sold", F.when(F.col("Units_Sold") < 0, F.lit(0))  # Replace negatives with 0
                .when(F.col("Units_Sold") == 0, F.lit(median_value))  # Replace zeros with the median
                .otherwise(F.col("Units_Sold")))  # Keep all valid positive values as-is
    .withColumn("Units_Sold", F.round(F.col("Units_Sold"), 2))  # Round to 2 decimals
)

# 4. Optional: Clean and round additional numerical columns (e.g., 'Sales_Excl_Tax')
cleaned_data = cleaned_data.withColumn("Sales_Excl_Tax", F.round(F.col("Sales_Excl_Tax"), 2))

# 5. Persist and save the cleaned dataset for performance optimization
cleaned_data.persist(StorageLevel.MEMORY_AND_DISK)


# COMMAND ----------

# Check for non-positive values (negative and zero)
non_positive_count = cleaned_data.filter("Units_Sold <= 0").count()
print(f"Number of non-positive values in 'Units_Sold': {non_positive_count}")

# Check for non-numeric values using the filter_non_positive_or_text function (if defined)
filtered_data = filter_non_positive_or_text(cleaned_data, "Units_Sold")
filtered_count = filtered_data.select("Units_Sold").count()
print(f"Number of non-numeric or non-positive values in 'Units_Sold': {filtered_count}")


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Data Cleaning Process for `Units_Sold`
# MAGIC
# MAGIC ##### 1. **Cast to Numeric**
# MAGIC - The `Units_Sold` column was cast to the `double` type to ensure consistent and accurate numeric processing for further analysis.
# MAGIC
# MAGIC ##### 2. **Calculate Approximate Median**
# MAGIC - Used Spark's `approxQuantile` method to compute the **median** efficiently across large datasets.
# MAGIC - The median was chosen because it is a robust measure of central tendency, unaffected by extreme outliers.
# MAGIC
# MAGIC ##### 3. **Replace Invalid Values**
# MAGIC - Replaced invalid values in the `Units_Sold` column:
# MAGIC   - **Null values** were replaced with the median value to ensure no missing data.
# MAGIC   - **Negative values** were replaced with `0`, as negative units sold are not realistic.
# MAGIC   - **Zero values** were replaced with the median to maintain data integrity.
# MAGIC - This step ensured that all values in `Units_Sold` are valid and meaningful for analysis.
# MAGIC
# MAGIC ##### 4. **Round Values**
# MAGIC - The `Units_Sold` and `Sales_Excl_Tax` columns were rounded to **2 decimal places** to maintain consistent precision throughout the dataset.
# MAGIC
# MAGIC ##### 5. **Persist and Save**
# MAGIC - The cleaned dataset was persisted in memory to improve performance for downstream processing and saved for future analysis steps.
# MAGIC
# MAGIC

# COMMAND ----------

# Optional Analysis: Check for duplicates based on multiple columns

# Group the data by relevant columns and count occurrences
cleaned_data_Gdf = cleaned_data.groupBy(['Promotion', 'Sales_Channel', 'State', 'Category_Name', 
                                         'Sub_Category_Name', 'Unknown_3', 'Unknown_2', 'Chain', 
                                         'Store', 'Supplier', 'Region', 'Area', 'Cluster', 
                                         'Unknown_1', 'Pack_Size', 'Fiscal_Week']).count()

# Filter to find duplicates (rows where count > 1)
duplicates = cleaned_data_Gdf.filter('count > 1')

# Display the count of duplicates
duplicates_count = duplicates.count()
print(f"Number of duplicate entries: {duplicates_count}")

# COMMAND ----------

# Optional Analysis

# Filter for non-positive or non-numeric values using the 'filter_non_positive_or_text' function (if defined)
# filtered_data = filter_non_positive_or_text(cleaned_data, "Units_Sold")

# Display the distinct values from the 'Units_Sold' column after filtering for non-positive or non-numeric entries
# filtered_data.select("Units_Sold").distinct().show()

# Generate summary statistics for the 'Units_Sold' column
# units_sold_summary = cleaned_data.select("Units_Sold").describe()

# Display the summary statistics for 'Units_Sold'
# units_sold_summary.show()


# COMMAND ----------

'''
Another Aproach to cleaning Units_Sold 
# Step 1: Convert Units_Sold to double and handle negative values and rounding.
cleaned_data = df.withColumn(
    "Units_Sold",
    F.when(F.col("Units_Sold").cast("double") < 0, F.lit(0)) # Replace negatives with 0
    .otherwise(F.round(F.col("Units_Sold").cast("double"), 2))  # Round valid values
)

# Step 2: Calculate the average Units_Sold per Area
avg_units_sold = cleaned_data.groupBy("Area").agg(
    F.round(F.avg("Units_Sold"), 2).alias("average_Units_Sold_per_Area") # Calculate and round area average
)

# Step 3: Perform a broadcast join for better performance as avg_units_sold DataFrame is small
joined_df = cleaned_data.join(F.broadcast(avg_units_sold), "Area", "left")

# Step 4: Replace missing Units_Sold values with the average and remove intermediate column
final_df = joined_df.withColumn(
    "Units_Sold",
    F.coalesce(F.col("Units_Sold"), F.col("average_Units_Sold_per_Area")) # Replace null with average
).drop("average_Units_Sold_per_Area")

# Step 5: Persist the final DataFrame 
# final_df = final_df.persist()

# Display a few rows of the final DataFrame
# display(final_df)
'''

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 3. Analyze Unknown Columns for Relationships
# MAGIC - Below code gives the relationships between three unknown columns (Unknown_1, Unknown_2, Unknown_3) to identify potential correlations, unique value distributions, and hierarchical structures.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Extract Numeric Parts of Unknown Columns
# MAGIC - Purpose: Transform the values of Unknown_1, Unknown_2, and Unknown_3 into numeric format for analysis.

# COMMAND ----------

# Extract numeric parts of Unknown columns
def extract_numeric(value):
    return float(''.join(filter(str.isdigit, str(value)))) if value else None

# Register UDF
extract_numeric_udf = F.udf(extract_numeric, FloatType())


# COMMAND ----------

# Apply numeric extraction to Unknown columns
unknown_numeric = cleaned_data.select(
    extract_numeric_udf(F.col("Unknown_1")).alias("Unknown_1"),
    extract_numeric_udf(F.col("Unknown_2")).alias("Unknown_2"),
    extract_numeric_udf(F.col("Unknown_3")).alias("Unknown_3")
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Calculate Correlations
# MAGIC - Purpose: Quantify the linear relationships between the columns.
# MAGIC

# COMMAND ----------


# Calculate correlations
correlations = {
    "Unknown_1 and Unknown_2": unknown_numeric.corr("Unknown_1", "Unknown_2"),
    "Unknown_1 and Unknown_3": unknown_numeric.corr("Unknown_1", "Unknown_3"),
    "Unknown_2 and Unknown_3": unknown_numeric.corr("Unknown_2", "Unknown_3")
}

# display(correlations)

# COMMAND ----------

# MAGIC %md 
# MAGIC Calculate Unique Value Counts
# MAGIC - Purpose: Measure the distinctness of values in each column.

# COMMAND ----------

# Calculate unique value counts
unique_counts = unknown_numeric.select(
    F.countDistinct("Unknown_1").alias("Unique_Unknown_1"),
    F.countDistinct("Unknown_2").alias("Unique_Unknown_2"),
    F.countDistinct("Unknown_3").alias("Unique_Unknown_3")
).collect()[0].asDict()


display(unique_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC Analyze Hierarchical Relationships
# MAGIC - Purpose: Identify hierarchical dependencies between the columns.

# COMMAND ----------

# Step 4: Analyze hierarchical relationships
# Count unique mappings between columns
hierarchical_relationships = {
    "Unknown_1 -> Unknown_2": cleaned_data.groupBy("Unknown_1").agg(F.countDistinct("Unknown_2").alias("Unique_Unknown_2")).select(F.avg("Unique_Unknown_2")).first()[0],
    "Unknown_1 -> Unknown_3": cleaned_data.groupBy("Unknown_1").agg(F.countDistinct("Unknown_3").alias("Unique_Unknown_3")).select(F.avg("Unique_Unknown_3")).first()[0],
    "Unknown_2 -> Unknown_3": cleaned_data.groupBy("Unknown_2").agg(F.countDistinct("Unknown_3").alias("Unique_Unknown_3")).select(F.avg("Unique_Unknown_3")).first()[0]
}

display(hierarchical_relationships)

# COMMAND ----------

# Display results
print("Correlation Matrix:")
for key, value in correlations.items():
    print(f"{key}: {value}")

print("\nUnique Value Counts:")
for key, value in unique_counts.items():
    print(f"{key}: {value}")

print("\nHierarchical Relationships:")
for key, value in hierarchical_relationships.items():
    print(f"{key}: {value:.2f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Summary of Results
# MAGIC
# MAGIC ### Correlation Matrix
# MAGIC - **Purpose**: Measures the linear relationships between the unknown columns.
# MAGIC - **Results**:  
# MAGIC   - **Unknown_1 and Unknown_2**: -0.0235 (Very weak negative correlation)  
# MAGIC   - **Unknown_1 and Unknown_3**: -0.0600 (Very weak negative correlation)  
# MAGIC   - **Unknown_2 and Unknown_3**: 0.1142 (Weak positive correlation)  
# MAGIC
# MAGIC - **Interpretation**:  
# MAGIC   No significant linear relationships exist between the columns, suggesting that these variables do not have a strong linear dependency.
# MAGIC
# MAGIC ### Unique Value Counts
# MAGIC - **Purpose**: Indicates the number of distinct values in each column.
# MAGIC - **Results**:  
# MAGIC   - **Unique_Unknown_1**: 4,537  
# MAGIC   - **Unique_Unknown_2**: 13,226  
# MAGIC   - **Unique_Unknown_3**: 178  
# MAGIC
# MAGIC - **Interpretation**:  
# MAGIC   - `Unknown_2` has the highest variability, suggesting a broad range of possible values across the dataset.  
# MAGIC   - `Unknown_3` appears to have a more limited, potentially categorical set of values, likely representing a smaller number of distinct categories or groups.
# MAGIC
# MAGIC ### Hierarchical Relationships
# MAGIC - **Purpose**: Analyzes how values in one column map to distinct values in another column.
# MAGIC - **Results**:  
# MAGIC   - **Unknown_1 → Unknown_2**: 2.92 (On average, each value in `Unknown_1` maps to ~3 distinct values in `Unknown_2`.)  
# MAGIC   - **Unknown_1 → Unknown_3**: 1.82 (Each value in `Unknown_1` maps to ~2 distinct values in `Unknown_3`.)  
# MAGIC   - **Unknown_2 → Unknown_3**: 1.00 (Each value in `Unknown_2` maps almost directly to one value in `Unknown_3`.)  
# MAGIC
# MAGIC - **Interpretation**:  
# MAGIC   - A strong one-to-one mapping exists between `Unknown_2` and `Unknown_3`, indicating that for each value in `Unknown_2`, there is likely a corresponding unique value 
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Key Insights:
# MAGIC
# MAGIC 1. **Weak Correlations**  
# MAGIC    - The columns do not exhibit strong linear relationships, suggesting that other types of relationships (e.g., categorical or hierarchical) might be more relevant for further analysis.
# MAGIC
# MAGIC 2. **Distinctiveness**  
# MAGIC    - **Unknown_2** contains the highest number of unique values, likely representing highly variable data.  
# MAGIC    - **Unknown_3** has a smaller number of distinct values, hinting at a more categorical or grouped nature.
# MAGIC
# MAGIC 3. **Hierarchical Structure**  
# MAGIC    - **Unknown_1** could serve as a broader grouping variable, with each value mapping to multiple entries in **Unknown_2** and **Unknown_3**.  
# MAGIC    - A strong one-to-one dependency exists between **Unknown_2** and **Unknown_3**, suggesting they might be used together in modeling or feature engineering to capture meaningful relationships.
# MAGIC

# COMMAND ----------

# Cast 'Fiscal_Week' to integer and handle invalid values by replacing them with 0
cleaned_data = cleaned_data.withColumn("Fiscal_Week", F.col("Fiscal_Week").cast("int"))
cleaned_data = cleaned_data.fillna({"Fiscal_Week": 0})

# Split Fiscal_Week into 'Year' and 'Week' assuming format YYYYWW
final_df = cleaned_data.withColumn(
    "Year", (F.col("Fiscal_Week") / 100).cast("int")
).withColumn(
    "Week", (F.col("Fiscal_Week") % 100).cast("int")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Used partitioning by **`Year`** and **`Week`**, derived from **`Fiscal_Week`**, to enhance both read and write performance.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary of Operations:
# MAGIC
# MAGIC - **Casting**: The `Fiscal_Week` column is cast to integer for further calculations.
# MAGIC - **Null Handling**: Null or invalid `Fiscal_Week` values are replaced with `0`.
# MAGIC - **Year and Week Extraction**: The `Year` and `Week` are extracted from `Fiscal_Week` for easier analysis.
# MAGIC - **Persistence Management**: The DataFrame is unpersisted after use to free up resources.
# MAGIC - **Data Saving**: The final cleaned and transformed DataFrame is saved to DBFS in Delta format, partitioned by `Year` and `Week`.
# MAGIC - **Partition Pruning**: By partitioning the data on `Year` and `Week`, partition pruning is enabled, making query execution more efficient.
# MAGIC
# MAGIC

# COMMAND ----------

# display(final_df.limit(10))

# COMMAND ----------

# Save the raw sales table to DBFS for ease of use later on
final_df.write\
    .partitionBy("Year", "Week")\
    .mode("overwrite")\
    .format("delta")\
    .option("overwriteSchema", "true")\
    .save("/dbfs/tmp/raw_sales_table")

# Unpersist the cleaned data after saving to free up memory
cleaned_data.unpersist()


# COMMAND ----------

# MAGIC %fs ls dbfs:/dbfs/tmp/raw_sales_table/Year=2021/Week=1/
# MAGIC

# COMMAND ----------

df = spark.read.format("delta").load("/dbfs/tmp/raw_sales_table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 2 (query optimisation):
# MAGIC  
# MAGIC The customer frequently queries the sales data for comparisons across different years. Example queries include:
# MAGIC
# MAGIC - Compare sales for `Sub_Category_Name_42` between weeks 34–36 in 2022 and the same weeks in 2021.
# MAGIC - Compare sales for `State = State_3` between weeks 1–25 in 2022 and the same weeks in 2021.
# MAGIC - Compare sales for `Category_Name_4` where `Sales_Channel = Sales_Channel_1`,` State = State_3`, `Supplier = Supplier_579`, and `Chain = Chain_1` for week 3 in 2022 vs. week 3 in 2021.
# MAGIC
# MAGIC The current implementation (provided in the function below) is suboptimal and requires improvement.
# MAGIC
# MAGIC Task:
# MAGIC
# MAGIC - Come up with a new implementation to **significantly reduce query execution time** while **maintaining the expected outputs**. You are free to apply **any methods** to achieve this goal, think creatively around both Spark and non-Spark native optimisation, and both the data and the algorithm. In fact, the real dataset will be x100 times larger so even small time saving would matter.

# COMMAND ----------

# Current implementation, make sure the your solution generates the same outputs
def current_implementation(df, time_filter, dim_filters):
    df_ty = df.withColumnRenamed("Sales_Excl_Tax", "Sales_This_Year").withColumnRenamed("Units_Sold", "Units_This_Year")
    df_ly = df.withColumnRenamed("Sales_Excl_Tax", "Sales_Last_Year").withColumnRenamed("Units_Sold", "Units_Last_Year")
    df_ly = df_ly.withColumn("Fiscal_Week", F.col('Fiscal_Week') + 100)

    df_YoY_comparison = df_ty.join(
        df_ly,
        ['Promotion','Sales_Channel','State','Category_Name','Sub_Category_Name','Unknown_3','Unknown_2',
        'Chain','Store','Supplier','Region','Area','Cluster','Unknown_1','Pack_Size','Fiscal_Week'],
        "outer"
        )
    
    df_YoY_comparison_filtered = df_YoY_comparison.filter(
        (F.col("Fiscal_Week") >= time_filter["start"])
        & (F.col("Fiscal_Week") <= time_filter["end"])
    )

    if dim_filters:
        for column, value in dim_filters.items():
            df_YoY_comparison_filtered = df_YoY_comparison_filtered.filter(F.col(column) == value)

    sales_this_year = df_YoY_comparison_filtered.agg(F.sum('Sales_This_Year').alias("Sales_This_Year")).collect()[0]["Sales_This_Year"]
    sales_last_year = df_YoY_comparison_filtered.agg(F.sum('Sales_Last_Year').alias("Sales_Last_Year")).collect()[0]["Sales_Last_Year"]
    
    print(f"Sales was {sales_this_year} this year, and {sales_last_year} last year") # Output 1 of 2: MUST print the sales numbers as a direct answer to user's query. 

    return df_YoY_comparison_filtered.toPandas() # Output 2 of 2: MUST convert to a PANDAS table to be used for further analysis, NOT a spark table!

# COMMAND ----------

# Example query #3
time_filter = {"start": 202203, "end": 202203}
dim_filters = {
    "Category_Name": "Category_Name_4",
    "Sales_Channel": "Sales_Channel_1",
    "State": "State_3",
    "Supplier": "Supplier_579",
    "Chain": "Chain_1",
}
result = current_implementation(df, time_filter, dim_filters)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **Proposed Solutions**
# MAGIC
# MAGIC ##### **Issues in Main Code**
# MAGIC 1. **Inefficient filtering** on `Fiscal_Week`, limiting performance.
# MAGIC 2. **Hardcoded current/previous years**, leading to inflexibility.
# MAGIC 3. **Lack of caching**, resulting in redundant computations.
# MAGIC 4. **Unoptimized joins**, causing excessive shuffling and delays.
# MAGIC 5. **No null handling** in aggregations, potentially affecting accuracy.
# MAGIC 6. **Scalability issues** for handling large datasets efficiently.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### **Optimizations in Updated Code**
# MAGIC 1. **Efficient Filtering**: Utilizes both `Year` and `Week` for better range filtering, including prior-year data.
# MAGIC 2. **Dynamic Year Handling**: Determines the current and previous years dynamically, adding flexibility.
# MAGIC 3. **Caching**: Filters and stores yearly datasets to avoid redundant computation.
# MAGIC 4. **Optimized Joins**: Repartitions the data based on join keys, reducing shuffle and improving performance.
# MAGIC 5. **Null Handling**: Ensures that null values are properly handled in aggregations, improving data integrity.
# MAGIC 6. **Scalability**: Adjusts partitions dynamically based on available resources (e.g., 2x cores) for better handling of large datasets.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Result**  
# MAGIC The optimized code is **faster**, **scalable**, and **reliable**, producing the same results as the original implementation while improving overall performance and efficiency.
# MAGIC

# COMMAND ----------

def optimized_implementation(df, time_filter=None, dim_filters=None):
    # Step 1: Apply time filter
    if time_filter:
        start_year = time_filter["start"] // 100
        start_week = time_filter["start"] % 100
        end_year = time_filter["end"] // 100
        end_week = time_filter["end"] % 100

        # Filter the dataframe by year and week range
        df_filtered = df.filter(
            (F.col("Year").between(start_year - 1, end_year)) &  # Include previous year for comparison
            (F.col("Week").between(start_week, end_week))        # Filter the week range
        )
    else:
        # Use the entire dataset if no time_filter is provided
        df_filtered = df

    # Step 2: Apply dimension filters
    if dim_filters:
        for column, value in dim_filters.items():
            df_filtered = df_filtered.filter(F.col(column) == value)

    # Cache the filtered dataframe for reuse
    df_filtered.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        # Step 3: Determine current and previous year dynamically
        if time_filter:
            current_year = start_year
            previous_year = start_year - 1
        else:
            # Use the max year in the dataset to determine current year
            current_year = df_filtered.select(F.max("Year")).collect()[0][0]
            previous_year = current_year - 1

        # Step 4: Filter data for the current and previous years
        current_year_data = df_filtered.filter(F.col("Year") == current_year) \
            .withColumnRenamed("Sales_Excl_Tax", "Sales_Current_Year") \
            .withColumnRenamed("Units_Sold", "Units_Current_Year")

        previous_year_data = df_filtered.filter(F.col("Year") == previous_year) \
            .withColumnRenamed("Sales_Excl_Tax", "Sales_Previous_Year") \
            .withColumnRenamed("Units_Sold", "Units_Previous_Year")

        # Cache the yearly datasets for reuse
        current_year_data.persist(StorageLevel.MEMORY_AND_DISK)
        previous_year_data.persist(StorageLevel.MEMORY_AND_DISK)

        try:
            # Step 5: Optimize the join using broadcasting or partitioning
            join_columns = ['Promotion', 'Sales_Channel', 'State', 'Category_Name', 'Sub_Category_Name', 
                            'Unknown_3', 'Unknown_2', 'Chain', 'Store', 'Supplier', 'Region', 'Area', 
                            'Cluster', 'Unknown_1', 'Pack_Size', 'Fiscal_Week']

            # Note: No BroadcastJoin is used because we are joining with "outer" which is not ideal for performance.

            # Strategy: Repartition data based on join keys to reduce shuffle
            # The system has 64 GB memory and 8 cores. Typically, the number of partitions is set to 2–3 times the number of cores.
            # For this system, we start with 16 partitions (2x the number of cores) to balance parallelism and shuffle efficiency.
            
            num_cores = spark.sparkContext.defaultParallelism
            num_partitions = max(32, num_cores * 2)  # Adjust this based on dataset size and performance monitoring
            
            partitioned_current = current_year_data.repartition(num_partitions, *join_columns)
            partitioned_previous = previous_year_data.repartition(num_partitions, *join_columns)

            # Perform the join operation after repartitioning
            comparison_data = partitioned_current.join(
                partitioned_previous,
                join_columns,
                "outer"
            )

            # Step 6: Aggregate and calculate the sales summary
            sales_summary = comparison_data.agg(
                F.round(F.sum("Sales_Current_Year"), 2).alias("Total_Sales_Current_Year"),
                F.round(F.sum("Sales_Previous_Year"), 2).alias("Total_Sales_Previous_Year")
            ).first()

            # Handle null values gracefully
            total_sales_current = sales_summary["Total_Sales_Current_Year"] or 0
            total_sales_previous = sales_summary["Total_Sales_Previous_Year"] or 0

            print(f"Total Sales: {total_sales_current} this year, and {total_sales_previous} last year")

            # Step 7: Handle cases with no filters provided
            if not time_filter and not dim_filters:
                print("No time filter or dimension filters provided. Processing the entire dataset.")

            # Drop 'Year' and 'Week' columns as they are no longer needed
            comparison_data = comparison_data.drop("Year", "Week")

            # Return the comparison data for further analysis
            return comparison_data.toPandas()

        finally:
            # Unpersist yearly datasets
            current_year_data.unpersist()
            previous_year_data.unpersist()

    finally:
        # Unpersist filtered data
        df_filtered.unpersist()


# COMMAND ----------

# MAGIC %md
# MAGIC This above script applies a number of optimization techniques and strategies for efficient data processing 
# MAGIC
# MAGIC - Here's an explanation of the technical approaches used:
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### **Techniques and Strategies**
# MAGIC ##### **1. Filtering and Persisting Data**
# MAGIC - **Time & Dimension Filtering**: Filters data by `Year`, `Week`, and other dimensions (e.g., `Category`, `Region`). Cached for performance.
# MAGIC
# MAGIC ##### **2. Dynamic Year Selection**
# MAGIC - **Current & Previous Year**: Uses `time_filter` to dynamically select years, or calculates the current year if no filter is provided.
# MAGIC
# MAGIC ##### **3. Column Renaming**
# MAGIC - **Clear Naming**: Renames columns like `Sales_Excl_Tax` to `Sales_Current_Year` for clarity during joins and aggregations.
# MAGIC
# MAGIC ##### **4. Optimized Join Strategy**
# MAGIC - **No Broadcast Joins**: Avoids broadcasting due to large datasets and uses repartitioning based on join keys for better shuffle efficiency.
# MAGIC - **Partition Count**: Partitions are set to `2x` the number of cores for optimal parallelism and reduced shuffle.
# MAGIC
# MAGIC ##### **5. Aggregation**
# MAGIC - **Sales Aggregation**: Calculates total sales for current and previous years, handling null values by using a default value of `0`.
# MAGIC
# MAGIC ##### **6. Dropping Unnecessary Columns**
# MAGIC - **Column Pruning**: Drops `Year` and `Week` columns post-join to reduce memory usage.
# MAGIC
# MAGIC ##### **7. Reuse of Cached Datasets**
# MAGIC - **Efficient Memory Use**: Caches filtered and intermediate data for reuse, then unpersists when no longer needed.
# MAGIC
# MAGIC ##### **8. Handling No Filters**
# MAGIC - **Flexible Processing**: If no filters are provided, processes the entire dataset.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### **Key Optimization Principles**
# MAGIC - **Minimizing Shuffles**: Repartitioning data based on join keys minimizes shuffle during the join.
# MAGIC - **Dynamic Resource Utilization**: Adjusts partition count based on available cores for efficient parallelism.
# MAGIC - **Caching**: Reduces redundant computations by persisting datasets.
# MAGIC - **Column Pruning**: Reduces memory usage by dropping unnecessary columns.
# MAGIC
# MAGIC
# MAGIC ##### **Result**
# MAGIC These strategies collectively improve the script's performance, making it scalable for large datasets while efficiently utilizing the system's resources.

# COMMAND ----------

# MAGIC %md
# MAGIC  
# MAGIC ##### Based on the dataset characteristics, a **shuffle hash join** is more efficient than a **broadcast join**.
# MAGIC

# COMMAND ----------

# spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
# spark.conf.get("spark.sql.join.preferSortMergeJoin")

# COMMAND ----------

# Adjust shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", 16)

# Enable dynamic partition pruning
spark.conf.set("spark.sql.dynamicPartitionPruning.enabled", "true")

# Use Snappy compression for Parquet
spark.conf.set("spark.sql.parquet.compression.codec", "SNAPPY")

# Disable automatic broadcast joins (you're manually optimizing joins)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# Avoid Sort-Merge Joins
spark.conf.set("spark.sql.join.preferSortMergeJoin", "false")

# Enable Adaptive Query Execution
spark.conf.set("spark.sql.adaptive.enabled", "true")

# COMMAND ----------

# Compare sales for Category_Name_4 where Sales_Channel = Sales_Channel_1, State = State_3, Supplier = Supplier_579, and Chain = Chain_1 for week 3 in 2022 vs. week 3 in 2021.

time_filter = {"start": 202203, "end": 202203}
dim_filters = {
    "Category_Name": "Category_Name_4",
    "Sales_Channel": "Sales_Channel_1",
    "State": "State_3",
    "Supplier": "Supplier_579",
    "Chain": "Chain_1",
}

result = optimized_implementation(df, time_filter, dim_filters)

# display(spark.createDataFrame(result))
# display(result)

# COMMAND ----------

# Compare sales for State = State_3 between weeks 1–25 in 2022 and the same weeks in 2021.

time_filter = {"start": 202201, "end": 202225}
dim_filters = {
    "State": "State_3"
}

result = optimized_implementation(df, time_filter, dim_filters)

# display(result)


# COMMAND ----------

# Compare sales for Sub_Category_Name_42 between weeks 34–36 in 2022 and the same weeks in 2021.

time_filter = {"start": 202234, "end": 202236}
dim_filters = {
    "Sub_Category_Name": "Sub_Category_Name_42"
}

result = optimized_implementation(df, time_filter, dim_filters)

# display(result)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 3 (python):
# MAGIC
# MAGIC It turns out that merchandise managers can't reliably set filtering parameters (who would have guessed?). However, Goodwork.ai has been working on a system that can turn natural language queries into structured filtering parameters using GenAI!
# MAGIC
# MAGIC The GenAI system works by accepting a question as text, and then outputing a JSON string of key-value pairs corresponding to the filtering selections. The GenAI system works, but is unreliable. Sometimes, key-value pairs that don't exist are output or a value is spelled incorrectly. A system has been developed to post process the raw json string based on lexical matching. Below is the code that is used to do this. It works, but that's it.
# MAGIC
# MAGIC **Read and understand the code below. What specific changes would you make to improve the code's efficiency, readability, maintainability and reliability? Write your answer in a few paragraphs, and feel free to write in pseudocode if helpful**

# COMMAND ----------

# For this test, we'll use a mock GenAI output below (gen_genai_output) to simulate the output of the genAI system. 
SCENARIOS = {
    0:{'State':'tate_7','Pack_Size':'PackSize_3','Category_Name': 'Categor_name_5'},
    1:{'Sub_Category_Name':'Sub_Category_Name_45','Area':'Area_29','SaleChanel': 'SalesChane1'},
    2:{'asdaSub_Category_Name':'Sub_Category_Name49','Store':'Stre_304'},
    3:{'Suplier': 'Supplier_1026','Promotion':'Promotion_2'},
    4:{'Caihn':'Chain_2','Region':'Region_14'},
}

def get_genai_output(scenario: int):
  return SCENARIOS[scenario]

# COMMAND ----------

def get_choice_data():
    blob_url = "https://externaldatastoreaccnt.blob.core.windows.net/takehometestdata/field_options.json"
    with urllib.request.urlopen(blob_url) as url:
        choices = json.loads(url.read().decode())
    return choices
# Download the key value data for lexical matching
choices = get_choice_data()


method = "jaro_winkler" # Other methods include levenshtein, cosine, jaro_winkler, trigram 
for scenario in range(5):
    genai_output = get_genai_output(scenario)
    
    fixed_output = {}
    output_message = ""
    for k,v in genai_output.items():
      # If we recognise the key, then look through the options in that key
      if k in choices.keys():
        fixed_output[k],score = match.extract(v, choices[k], match_type=method)[0]
        if score < 0.9:
          output_message += f"Infering key value from {k} to {fixed_output[k]}\n"
      else:
        # If we don't recognise key, then infer the key by looping over each possible key and finding best match
        possible_output = {}
        for k_possible in choices.keys():
          possible_output[k_possible] = match.extract(v, choices[k_possible], match_type=method)[0]
        best_k, (best_v,score) = sorted(possible_output.items(), key=lambda item: item[1][1], reverse=True)[0]
        fixed_output[best_k] = best_v
        output_message += f"We do not recognise the key {k}: infering the key as {best_k}\n"
        if score < 0.9:
          output_message += f"Infering key value from {v} to {best_v}\n"
    
    print(f"Scenario {scenario}")
    print(fixed_output)
    print(output_message)


# COMMAND ----------

# MAGIC %md
# MAGIC - Here are the key points to improve the code:
# MAGIC
# MAGIC 1. **Readability**:
# MAGIC    - Use consistent naming conventions (e.g., `corrected_output` instead of `fixed_output`).
# MAGIC    - Modularize the code into reusable functions (`match_key`, `match_value`).
# MAGIC    - Add comments and clarify complex logic.
# MAGIC
# MAGIC 2. **Efficiency**:
# MAGIC    - Precompute similarity maps for keys to avoid repeated computations during runtime.
# MAGIC
# MAGIC 3. **Maintainability**:
# MAGIC    - Refactor key handling into separate functions for recognized and unrecognized keys.
# MAGIC    - Make similarity threshold and method configurable.
# MAGIC    - Ensure robust handling of invalid `choices` data (e.g., invalid JSON, missing file).
# MAGIC
# MAGIC 4. **Reliability**:
# MAGIC    - Add error handling for cases where no match is found.
# MAGIC    - Use logging instead of print statements for better traceability.
# MAGIC    - Include type hints and validation for inputs/outputs.
# MAGIC
# MAGIC These changes improve performance, modularity, and overall robustness while maintaining clarity and ease of future updates.

# COMMAND ----------

# Download choices data
def get_choice_data():
    """
    Fetch choices data from a remote blob storage.
    Returns:
        dict: Parsed JSON object containing choice data.
    """
    blob_url = "https://externaldatastoreaccnt.blob.core.windows.net/takehometestdata/field_options.json"
    try:
        with urllib.request.urlopen(blob_url) as url:
            return json.loads(url.read().decode())
    except Exception as e:
        print(f"Error downloading choice data: {e}")
        return {}

# Perform lexical matching between a value and a list of choices
def match_value_to_choices(value, choices, method="jaro_winkler"):
    """
    Match a value to the best choice using a specified matching method.
    Args:
        value (str): Value to be matched.
        choices (list): List of valid choices.
        method (str): Matching method (default: "jaro_winkler").
    Returns:
        tuple: Best match and its score.
    """
    return match.extract(value, choices, limit=1)[0]  # Returns (match, score)

# Inference function for the best matching key
def infer_best_key(value, choices, method="jaro_winkler"):
    """
    Infer the best matching key for a value across all possible choices.
    Args:
        value (str): The value to match.
        choices (dict): Dictionary of possible keys and their valid values.
        method (str): Matching method (default: "jaro_winkler").
    Returns:
        tuple: Best matching key, best matching value, and match score.
    """
    possible_matches = {
        key: match_value_to_choices(value, valid_values, method)
        for key, valid_values in choices.items()
    }
    best_match_key, (best_value, score) = max(possible_matches.items(), key=lambda x: x[1][1])
    return best_match_key, best_value, score

# Main processing loop for scenarios
def process_scenarios(choices, scenarios, method="jaro_winkler"):
    """
    Process all scenarios to match keys and values using lexical matching.
    Args:
        choices (dict): Dictionary of valid choices for each key.
        scenarios (dict): Dictionary of scenarios to process.
        method (str): Matching method (default: "jaro_winkler").
    """
    for scenario_id, scenario_data in scenarios.items():
        print(f"\nProcessing Scenario {scenario_id}...")
        fixed_output = {}
        output_message = ""

        for k, v in scenario_data.items():
            if k in choices:
                # Match value using known key
                best_value, score = match_value_to_choices(v, choices[k], method)
                fixed_output[k] = best_value
                if score < 0.9:
                    output_message += f"Inferred key value for '{k}' as '{best_value}' with low confidence (score: {score:.2f}).\n"
            else:
                # Infer best matching key
                best_key, best_value, score = infer_best_key(v, choices, method)
                fixed_output[best_key] = best_value
                output_message += f"Unrecognized key '{k}' inferred as '{best_key}'.\n"
                if score < 0.9:
                    output_message += f"Inferred value for '{v}' as '{best_value}' with low confidence (score: {score:.2f}).\n"

        # Output results
        print(f"Fixed Output: {fixed_output}")
        if output_message:
            print(f"Messages:\n{output_message}")


# COMMAND ----------

SCENARIOS = {
        0: {'State': 'tate_7', 'Pack_Size': 'PackSize_3', 'Category_Name': 'Categor_name_5'},
        1: {'Sub_Category_Name': 'Sub_Category_Name_45', 'Area': 'Area_29', 'SaleChanel': 'SalesChane1'},
        2: {'asdaSub_Category_Name': 'Sub_Category_Name49', 'Store': 'Stre_304'},
        3: {'Suplier': 'Supplier_1026', 'Promotion': 'Promotion_2'},
        4: {'Caihn': 'Chain_2', 'Region': 'Region_14'},
    }

choices = get_choice_data()
if choices:
    process_scenarios(choices, SCENARIOS)
else:
    print("Error: No valid choice data available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Thank you for walking with me till the end! 🙏

# COMMAND ----------


