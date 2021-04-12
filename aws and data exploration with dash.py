import dask.dataframe as dd
import pandas as pd
from datetime import datetime
from dask.distributed import Client
import json

def Assignment1A(user_reviews_csv):
    client = Client('127.0.0.1:8786')
    client = client.restart()    
    
    
    data = dd.read_csv(user_reviews_csv)
    num_products = data.groupby(['reviewerID']).count(split_out=4)
    num_products = num_products[['asin']]

    ratings = data.groupby(['reviewerID']).mean(split_out=4)
    ratings = ratings[['overall']]

    reviews = data[['reviewerID', 'unixReviewTime']]
    reviews['datetime'] = dd.to_datetime(reviews['unixReviewTime'], unit = 's')
    first_review = reviews.groupby('reviewerID').min(split_out=4)
    first_review = first_review[['datetime']]
    first_review['datetime'] = first_review['datetime'].dt.year
   
    votes = data[['reviewerID', 'helpful']]
    votes['helpful'] = votes['helpful'].str[1:-1]
    new = votes['helpful'].str.partition(', ')
    votes['help'] = new[0]
    votes['help'] = votes['help'].astype(int)
    votes['total_votes'] = new[2]
    votes['total_votes'] = votes['total_votes'].astype(int)
    votes = votes.groupby(['reviewerID']).sum(split_out=4)


    num_products.assign(avg_ratings=ratings['overall'])
    num_products.assign(reviewing_since= first_review['datetime'])
    num_products.assign(helpful_votes = votes['help'])
    num_products.assign(total_votes = votes['total_votes'])
    
    final = num_products
    final = final.reset_index()

    
    
    cols = {"asin": 'number_products_rated',
           "avg_ratings": 'avg_ratings',
           "reviewing_since": 'reviewing_since',
           "helpful_votes": 'helpful_votes', 
           "total_votes" : 'total_votes'}
    result = final.rename(columns = cols)

    
    submit = result.describe().compute().round(2)    
    with open('results_1A.json', 'w') as outfile: json.dump(json.loads(submit.to_json()), outfile)


def Assignment1B(user_reviews_csv,products_csv):
    client = Client('127.0.0.1:8786')
    client = client.restart()

    # Write your results to "results_1B.json" here and round your solutions to 2 decimal points
    
    reviews = dd.read_csv(user_reviews_csv)
    prods = dd.read_csv(products_csv, dtype={'brand': 'object'})
    
    reviews_miss = reviews.isna().sum()
    perc_reviews = ((reviews_miss/len(reviews))*100).round(2)
    reviews_na = list(reviews_miss)
    perc_reviews = ((reviews_miss/len(reviews))*100).round(2)
    categories = list(reviews.columns)
    reviews_missing = {categories[i]: list(perc_reviews)[i] for i in range(1, len(categories))}

    prods_miss = prods.isna().sum()
    perc_prods = ((prods_miss/len(prods))*100).round(2)
    categories = list(prods.columns)
    prods_missing = {categories[i]: list(perc_prods)[i] for i in range(1, len(categories))}
    q1 = {}
    q1['products'] = prods_missing
    q1['reviews'] = reviews_missing
    final = {}
    final['q1'] = q1
    
    result = reviews.merge(prods, how = 'inner', left_on = 'asin', right_on = 'asin').persist()
    corr = result['overall'].corr(result['price'], method = 'pearson').compute(get=dask.get)
    final['q2'] = corr.round(2)
    
    q3 = {}
    mean = prods['price'].mean().compute()
    q3['mean'] = mean.round(2)
    std = prods['price'].std().compute()
    q3['std'] = std.round(2)
    med = prods['price'].median().compute()
    q3['50%'] = med.round(2)
    minn = prods['price'].min().compute()
    q3['min'] = minn.round(2)
    maxx = prods['price'].max().compute()
    q3['max'] = maxx.round(2)
    final['q3'] = q3
    
    prods_na = prods.dropna(subset = ['categories'])
    super_cat = [x.split(', ')[0].strip('[').strip(']').strip("'").strip("\"").replace("\\", '') for x in prods_na['categories']]
    
    count = Counter(super_cat)
    final['q4'] = count
    
    
    prod_set = set(prods['asin'])
    reviews_set = set(reviews['asin'])
    overlap = prod_set - reviews_set
    if len(overlap) > 0:
        final['q5'] = 1
    else:
        final['q5'] = 0
        
    prods_na_2 = prods.dropna(subset = ['related'])
    test = [x.replace(":", ',').split(', ') for x in prods_na_2['related']]
    table = str.maketrans('', '', "[]{}''")
    numeric = [x.translate(table) for y in test for x in y if any(digit in x for digit in '0123456789') ]
    q6 = set(numeric) - prod_set
    if len(q6) > 0:
        final['q6'] = 1
    else:
        final['q6'] = 0

    with open('results_1B.json', 'w') as fp:
        json.dump(final, fp)

Assignment1A('data/user_reviews.csv')
