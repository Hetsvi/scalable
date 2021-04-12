import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR
from pyspark.sql.functions import col, when, map_keys, map_values
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, PCA
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml.stat import Summarizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import Word2Vec

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics

## mean and count of ratings
def mcr(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    review_mean = review_data.groupBy(asin_column).mean()
    review_Count = review_data.groupBy(asin_column).count()
    product_data = product_data.join(review_mean, [asin_column], how = 'left')
    product_data = product_data.join(review_Count, [asin_column], how = 'left')
    product_data = product_data.withColumnRenamed("count", "countRating").withColumnRenamed("avg(Overall)", "meanRating")

    described = product_data.describe().toPandas()
    

    # ---------------------- Put results in res dict --------------------------

    res = {
        'count_total': int(described['asin'][0]),
        'mean_meanRating': float(described['meanRating'][1]),
       'variance_meanRating':float(described['meanRating'][2]) * float(described['meanRating'][2]),
        'numNulls_meanRating':int(described['asin'][0]) - int(described['meanRating'][0]),
        'mean_countRating': float(described['countRating'][1]),
        'variance_countRating': float(described['countRating'][2]) * float(described['countRating'][2]),
        'numNulls_countRating': int(described['asin'][0]) - int(described['countRating'][0])
    }
    # Modify res:


    data_io.save(res, 'mcr')
    return res
    
## flatten categories and salesRank
def fcs(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    product_data = product_data.withColumn(category_column, product_data[categories_column].getItem(0).getItem(0))
    product_data = product_data.withColumn(category_column,\
               when(col(category_column) == '', None).otherwise(col(category_column)))
    
    product_data = product_data.withColumn(bestSalesCategory_column, map_keys(product_data[salesRank_column]).getItem(0))
    product_data = product_data.withColumn(bestSalesCategory_column,\
                when(col(bestSalesCategory_column) == '', None).otherwise(col(bestSalesCategory_column)))   

    product_data = product_data.withColumn(bestSalesRank_column, map_values(product_data[salesRank_column]).getItem(0))
    product_data = product_data.withColumn(bestSalesRank_column,\
                when(col(bestSalesRank_column) == '', None).otherwise(col(bestSalesRank_column)))   


    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': (product_data.count()),
        'mean_bestSalesRank': product_data.agg({bestSalesRank_column: 'mean'}).collect()[0][0],
        'variance_bestSalesRank': product_data.agg({bestSalesRank_column: 'variance'}).collect()[0][0],
        'numNulls_category': product_data.filter(product_data[category_column].isNull()).count(),
        'countDistinct_category': product_data.agg(F.countDistinct(col(category_column))).collect()[0][0],
        'numNulls_bestSalesCategory': product_data.filter(product_data[bestSalesCategory_column].isNull()).count(),
        'countDistinct_bestSalesCategory': product_data.agg(F.countDistinct(bestSalesCategory_column)).collect()[0][0]
    }
    # Modify res:

    data_io.save(res, 'fcs')
    return res
    # -------------------------------------------------------------------------

## flatten related
def fr(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    
    from pyspark.sql.functions import explode_outer
    df = product_data
    df3 = df.select(df.asin, explode_outer(df.related))
    df3 = df3.where((df3.key.isNull() == True) | (df3.key == attribute))

    df3 = df3.select('*',F.size(df3.value).alias(countAlsoViewed_column))


    df3 =  df3.withColumn(
        "counAlsoViewed2",
        F.when(
            df3.countAlsoViewed == -1,
            None
        ).otherwise(df3.countAlsoViewed)
    )


    df3more = df3.select(df3.asin, df3.key, df3.counAlsoViewed2, explode_outer(df3.value))
    df3more = df3more.selectExpr("asin as explodedasin", "key as key", "col as listvalue", "counAlsoViewed2 as counAlsoViewed2")
    joined_df = df3more.join(df, df.asin == df3more.listvalue, 'right_outer')
    grouped = joined_df.groupBy(joined_df.explodedasin).mean()
    grouped = grouped.withColumnRenamed("avg(price)", meanPriceAlsoViewed_column).withColumnRenamed("avg(counAlsoViewed2)", countAlsoViewed_column)
    #grouped = grouped.selectExpr("explodedasin as explodedasin",  "avg(price) as meanPriceAlsoViewed", "avg(counAlsoViewed2) as countAlsoViewed")
    described = grouped.describe()
    described = described.toPandas()
    countgroup = df3more.groupBy(df3more.explodedasin).mean()
    countgroupdescribe = countgroup.describe().toPandas()
    count = product_data.count()

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': count,
        'mean_meanPriceAlsoViewed': float(described[meanPriceAlsoViewed_column][1]),
        'variance_meanPriceAlsoViewed': float(described[meanPriceAlsoViewed_column][2]) * float(described[meanPriceAlsoViewed_column][2]),
        'numNulls_meanPriceAlsoViewed': count - int(described[meanPriceAlsoViewed_column][0]),
        'mean_countAlsoViewed': float(countgroupdescribe['avg(counAlsoViewed2)'][1]),
        'variance_countAlsoViewed': float(countgroupdescribe['avg(counAlsoViewed2)'][2]) * float(countgroupdescribe['avg(counAlsoViewed2)'][2]),
        'numNulls_countAlsoViewed': count - int(countgroupdescribe['avg(counAlsoViewed2)'][0])
    }
    
    # Modify res:

    data_io.save(res, 'fr')
    return res
    # -------------------------------------------------------------------------

## data imputation
def d(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------


    product_data = product_data.withColumn(price_column, product_data[price_column].cast('float'))
    median = product_data.approxQuantile(price_column, [0.5], 0)
    df_stats = product_data.select(F.mean(col(price_column)).alias('mean')).collect()
    mean = df_stats[0]['mean']

    product_data = product_data.withColumn(meanImputedPrice_column, product_data[price_column])
    product_data = product_data.fillna(mean, subset=[meanImputedPrice_column])
    
    product_data = product_data.withColumn(medianImputedPrice_column, product_data[price_column])
    product_data = product_data.fillna(median[0], subset=[medianImputedPrice_column])
    
    product_data = product_data.withColumn(unknownImputedTitle_column,\
            when(col(title_column) == '', 'unknown').otherwise(col(title_column)))
    product_data = product_data.fillna('unknown', subset=[unknownImputedTitle_column])


    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': product_data.count(),
        'mean_meanImputedPrice': product_data.agg({meanImputedPrice_column: 'mean'}).collect()[0][0],
        'variance_meanImputedPrice': product_data.agg({meanImputedPrice_column: 'variance'}).collect()[0][0],
        'numNulls_meanImputedPrice': product_data.filter(product_data[meanImputedPrice_column].isNull()).count(),
        'mean_medianImputedPrice': product_data.agg({medianImputedPrice_column: 'mean'}).collect()[0][0],
        'variance_medianImputedPrice': product_data.agg({medianImputedPrice_column: 'variance'}).collect()[0][0],
        'numNulls_medianImputedPrice': product_data.filter(product_data[medianImputedPrice_column].isNull()).count(),
        'numUnknowns_unknownImputedTitle': product_data.filter(product_data[unknownImputedTitle_column]== 'unknown').count(),
    }
    # Modify res:

    data_io.save(res, 'd')
    return res
    # -------------------------------------------------------------------------

## embed title with word2vec
def etw(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    
    df5 = product_processed_data
    df5 = df5.select("*", F.lower(df5.title))
    df5 = df5.selectExpr("asin as asin", "title as title", "category as category", "lower(title) as lowtitle")
    df5 = df5.withColumn("titleArray", F.split(df5.lowtitle," "))
    Seed = 102
#     word2Vec = Word2Vec().setVectorSize(16).setMinCount(100).setSeed(Seed).setNumPartitions(4).setInputCol('titleArray').setOutputCol("result")
    word2Vec = Word2Vec(vectorSize=16, minCount = 100,seed = Seed, numPartitions = 4, inputCol="titleArray", outputCol="result")
    model = word2Vec.fit(df5)
    result = model.transform(df5)
    product_processed_data_output = result
    
    synonyms = model.findSynonyms(word_0, 10) 
    pdf = synonyms.toPandas()
    synonyms1 = model.findSynonyms(word_1, 10) 
    pdf1 = synonyms.toPandas()
    synonyms2 = model.findSynonyms(word_2, 10)  
    pdf2 = synonyms.toPandas()

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': df5.count(),
        'size_vocabulary': model.getVectors().count(),
        'word_0_synonyms': list(zip(*map(pdf.get, pdf))),
        'word_1_synonyms': list(zip(*map(pdf1.get, pdf1))),
        'word_2_synonyms': list(zip(*map(pdf2.get, pdf2)))
    }
    # Modify res:
    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)

    data_io.save(res, 'etw')
    return res
    # -------------------------------------------------------------------------

## one-hot encoding category and PCA
def ohecp(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'


    indexer = StringIndexer(inputCol = category_column, outputCol = categoryIndex_column)
    indexed = indexer.fit(product_processed_data).transform(product_processed_data)
    #indexed.show()
    encoder = OneHotEncoderEstimator(inputCols = [categoryIndex_column],
                         outputCols= [categoryOneHot_column], dropLast = False)
    model = encoder.fit(indexed)
    encoded = model.transform(indexed)
    
    pca = PCA(k=15, inputCol=categoryOneHot_column, outputCol=categoryPCA_column)
    model = pca.fit(encoded)

    result = model.transform(encoded)

    summarizer = Summarizer.metrics('mean')
    oneHot = result.select(Summarizer.mean(result[categoryOneHot_column])).collect()
    pca = result.select(Summarizer.mean(result[categoryPCA_column])).collect()


    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': result.count(),
        'meanVector_categoryOneHot': oneHot[0][0],
        'meanVector_categoryPCA': pca[0][0]
    }

    data_io.save(res, 'ohecp')
    return res
    # -------------------------------------------------------------------------
    
## Train a Decision Tree Regression model
def tdtrm(data_io, train_data, test_data):


    dt = DecisionTreeRegressor(labelCol = 'overall', maxDepth = 5)
    
    model = dt.fit(train_data)
    predictions = model.transform(test_data)

    evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': rmse
    }

    data_io.save(res, 'tdtrm')
    return res
    # -------------------------------------------------------------------------
    
## Hyperparameter tuning for the Decision Tree Regression model
def htdtrm(data_io, train_data, test_data):

    train, val = train_data.randomSplit([0.75, 0.25])
    evaluator = RegressionEvaluator().setLabelCol("overall")\
    .setPredictionCol("prediction").setMetricName("rmse")
    models = []
    rmses = []
    
    # max depth 5
    dt_5 = DecisionTreeRegressor(labelCol = 'overall', maxDepth = 5)
    model_5 = dt_5.fit(train)
    models.append(model_5)
    pred_5 = model_5.transform(val)
    rmse_5 = evaluator.evaluate(pred_5)
    rmses.append(rmse_5)
    
    # max depth 7
    dt_7 = DecisionTreeRegressor(labelCol = 'overall', maxDepth = 7)
    model_7 = dt_7.fit(train)
    models.append(model_7)
    pred_7 = model_7.transform(val)
    rmse_7 = evaluator.evaluate(pred_7)
    rmses.append(rmse_7)
    
    # max depth 9
    dt_9 = DecisionTreeRegressor(labelCol = 'overall', maxDepth = 9)
    model_9 = dt_9.fit(train)
    models.append(model_9)
    pred_9 = model_9.transform(val)
    rmse_9 = evaluator.evaluate(pred_9)
    rmses.append(rmse_9)
    
    # max depth 12
    dt_12 = DecisionTreeRegressor(labelCol = 'overall', maxDepth = 12)
    model_12 = dt_12.fit(train)
    models.append(model_12)
    pred_12 = model_12.transform(val)
    rmse_12 = evaluator.evaluate(pred_12)
    rmses.append(rmse_12)

    ind = rmses.index(min(rmses))
    mod = models[ind]
    test_pred = mod.transform(test_data)
    test_rmse = evaluator.evaluate(test_pred)

    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': test_rmse,
        'valid_rmse_depth_5': rmse_5,
        'valid_rmse_depth_7': rmse_7,
        'valid_rmse_depth_9': rmse_9,
        'valid_rmse_depth_12': rmse_12,
    }
    data_io.save(res, 'htdtrm')
    return res
    # -------------------------------------------------------------------------

