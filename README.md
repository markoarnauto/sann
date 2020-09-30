# SANN
## Streaming Approximate Nearest Neighbor

Approximate nearest neighbor search (ANN) is a very popular field of research
 and a lot of algorithms exist. E.g. checkout [faiss]() and [annoy](). Since most
 of these algorithms are inherently designed for static data I played around with a popular ANN algorithm called
  [LSH](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) using random projections.
 It is a pretty simply algorithm that is very well suited for a streaming settings. For a detailed explaination read the 
 [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.374.926&rep=rep1&type=pdf) or this 
 [blog article](https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23).
 
 ### Parameters


*  `num_tables`: The number of tables to lookup, the more the better (but slower)
*  `hash_size`: The hash size to encode a vector, the more the better (but slower insertion and higher memory footprint)
*  `bucket_size`: The maximum size of a bucket. This determines the memory footprint. This basically sets the amount of past vectors to search against. There is no need to delete old vectors.

### Example

Simply copy paste the *lsh.py* into your directory and import it.

```python

import lsh
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':

    ## create toy dataset
    # some random tweets from http://rasmusrasmussen.com/rtweets/
    tweets = [
        'My home is hard labor, and I want to get a yo-yo. A path towards miniscule nights, forever. #sillyslam #randomtweet',
        'My ingenuity is hard labor, and I want to be famous. Unlikely swell criminals, in a way. #jingleburger #randomtweet',
        'My sister is a joy, and I want to compete in the Olympics. A bit of clammy whipped cream, man. #dumbrun #randomtweet',
        'My ingenuity is debt free, and I want to plant a tree. A hint of perfect shoes, please. #powerfest #randomtweet',
        'My lifestyle is funky, and I want to be heard. Here is to thirsty encounters, or something. #powerpile #randomtweet',
        'My diet is a personal wish, and I want to fly away. For all detailed glamour, forever. #thiswad #randomtweet',
        'My favorite team is paid for, and I want to slow down. More sick lessons, so say we all. #megababe #randomtweet'
    ]
    # create a lot of duplicates
    tweets_with_duplicates = []
    [tweets_with_duplicates.extend(tweets) for _ in range(500)]

    ## get vectors from tweets using sklearn
    vecs = TfidfVectorizer(stop_words='english').fit_transform(tweets_with_duplicates).toarray()

    ## create lsh
    num_tables = 20
    hash_size = 13
    bucket_size = 100000
    input_dimension = vecs.shape[1]
    sann = lsh.LSH(num_tables, hash_size, input_dimension, bucket_size)

    ## simulate streaming nearest neighbor search
    for id, vec in enumerate(vecs):
        sann[vec] = id  # append tweet to the index
        ids = sann[vec]  # get the ids of the nearest vectors

        # get next tweet
        next_id = ids[0]
        next_tweet = tweets_with_duplicates[next_id]

        # pretty print current tweet and its nearest neighbor
        print(f'\n\033[1mNearest neighbor of\033[0m\n{tweets_with_duplicates[id]}\n\033[1mis\033[0m\n{next_tweet}')
```
 
 
 
  