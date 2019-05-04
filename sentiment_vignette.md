Sentiment analysis via FeatureHashing
================

This vignette demonstrates a sentiment analysis task, using the FeatureHashing package for data preparation (instead of more established text processing packages such as 'tm') and the XGBoost package to train a classifier (instead of packages such as glmnet).

With thanks to Maas et al (2011) *Learning Word Vectors for Sentiment Analysis* we make use of the 'Large Movie Review Dataset'. In their own words:

> We constructed a collection of 50,000 reviews from IMDB, allowing no more than 30 reviews per movie. The constructed dataset contains an even number of positive and negative reviews, so randomly guessing yields 50% accuracy. Following previous work on polarity classification, we consider only highly polarized reviews. A negative review has a score less than or equal to 4 out of 10, and a positive review has a score equal to or greater than 7 out of 10. Neutral reviews are not included in the dataset. In the interest of providing a benchmark for future work in this area, we release this dataset to the public.

In our case we will only use the training data of 25,000 reviews as we'll only go as far as checking our classifier via validation.

### Why FeatureHashing?

It's not essential to use FeatureHashing for this movie review dataset - a combination of the tm and glmnet packages works reasonably well here - but it's a convenient way to illustrate the benefits of FeatureHashing. For example, we will see how easily we can select the size of the hashed representations of the review texts and will understand the options FeatureHashing makes available for processing the data in subsets.

The combination of FeatureHashing and XGBoost can also be seen as a way to access some of the benefits of the Vowpal Wabbit approach to machine learning, without switching to a fully online learner. By using the 'hashing trick', FeatureHashing easily handles features of many possible categorical values. These are then stored in a sparse, low-memory format on which XGBoost can quickly train a linear classifier using a gradient descent approach. At a minimum this is a useful way to better understand how tools like Vowpal Wabbit push the same approaches to their limits. But in our case we get to benefit from these approaches without leaving the R environment.

### Package versions

This vignette uses FeatureHashing v9.0 and XGBoost v0.3-3.

``` r
library(FeatureHashing)
library(Matrix)
library(xgboost)
```

### Basic data preparation

First we read the training data and perform some simple text cleaning using gsub() to remove punctuation before converting the text to lowercase. At this stage each review is read and stored as a single continuous string.

``` r
imdb <- read.delim("labeledTrainData.tsv", quote = "", as.is = T)
imdb$review <- tolower(gsub("[^[:alnum:] ]", " ", imdb$review))
```

Which, using one of the shortest reviews as an example, leaves us with the following. At this stage, the review is still being stored as single character string and we only use strwrap() for pretty printing:

``` r
strwrap(imdb$review[457], width = 80)
```

    ## [1] "kurosawa is a proved humanitarian this movie is totally about people living in" 
    ## [2] "poverty you will see nothing but angry in this movie it makes you feel bad but" 
    ## [3] "still worth all those who s too comfortable with materialization should spend 2"
    ## [4] "5 hours with this movie"

### FeatureHashing

We can then hash each of our review texts into a document term matrix. We'll choose the simpler binary matrix representation rather than term frequency. The FeatureHashing package provides a convenient split() function to split each review into words, before then hashing each of those words/terms to an integer value to use as a column reference in a sparse matrix.

``` r
d1 <- hashed.model.matrix(~ split(review, delim = " ", type = "existence"),
                          data = imdb, hash.size = 2^16)
```

The other important choice we've made is the hash.size of 2^16. This limits the number of columns in the document term matrix and is how we convert a feature of an unknown number of categories to a binary representation of known, fixed size. For the sake of speed and to keep memory requirements to a minimum, we're using a relatively small value compared to the number of unique words in this dataset. This parameter can be seen as a hyperparameter to be tuned via validation.

The resulting 50MB dgCMatrix is the sparse format used by the Matrix package that ships with base R. A dense representation of the same data would occupy 12GB. Just out of curiosity, we can readily check the new form of our single review example:

``` r
as.integer(which(d1[457, ] != 0))
```

    ##  [1]     1  2780  6663 12570 13886 16269 18258 19164 19665 20531 22371
    ## [12] 22489 26981 28697 29324 32554 33091 35321 35778 35961 37510 38786
    ## [23] 39382 45651 46446 51516 52439 54827 57399 57784 58791 59061 60097
    ## [34] 61317 62283 62878 62906 62941 63295

The above transformation is independent of the other reviews and as long as we use the same options in hashed.model.matrix, we could process a larger volume of text in batches to incrementally construct our sparse matrix. Equally, if we are building a classifier to assess as yet unseen test cases we can independently hash the test data in the knowledge that matching terms across training and test data will be hashed to the same column index.

### Training XGBoost

For this vignette we'll train a classifier on 20,000 of the reviews and validate its performance on the other 5,000. To enable access to all of the XGBoost parameters we'll also convert the document term matrix to an xgb.DMatrix and create a watchlist to monitor both training and validation set accuracy. The matrix remains sparse throughout the process. Other R machine learning packages that accept sparse matrices include glmnet and the support vector machine function of the e1071 package.

``` r
train <- c(1:20000); valid <- c(1:nrow(imdb))[-train]
dtrain <- xgb.DMatrix(d1[train,], label = imdb$sentiment[train])
dvalid <- xgb.DMatrix(d1[valid,], label = imdb$sentiment[valid])
watch <- list(train = dtrain, valid = dvalid)
```

First we train a linear model, reducing the learning rate from the default 0.3 to 0.02 and trying out 10 rounds of gradient descent. We also specify classification error as our chosen evaluation metric for the watchlist.

``` r
m1 <- xgb.train(booster = "gblinear", nrounds = 10, eta = 0.02,
                data = dtrain, objective = "binary:logistic",
                watchlist = watch, eval_metric = "error")
```

    ## [1]  train-error:0.076000    valid-error:0.149600 
    ## [2]  train-error:0.064600    valid-error:0.141800 
    ## [3]  train-error:0.055150    valid-error:0.135600 
    ## [4]  train-error:0.048450    valid-error:0.131000 
    ## [5]  train-error:0.042700    valid-error:0.128400 
    ## [6]  train-error:0.037500    valid-error:0.126400 
    ## [7]  train-error:0.033150    valid-error:0.123000 
    ## [8]  train-error:0.029950    valid-error:0.121200 
    ## [9]  train-error:0.027700    valid-error:0.119200 
    ## [10] train-error:0.024950    valid-error:0.118000

That code chunk runs in just a few seconds and the validation error is already down to a reasonable 12%. So FeatureHashing has kept our memory requirement to about 50MB and XGBoost's efficient approach to logistic regression has ensured we can get rapid feedback on the performance of the classifier.

With this particular dataset a tree-based classifier would take far longer to train and tune than a linear model. Without attempting to run it for a realistic number of rounds, the code below shows how easily we can switch to the tree-based mode of XGBoost:

``` r
m2 <- xgb.train(data = dtrain, nrounds = 10, eta = 0.02, 
                max.depth = 10, colsample_bytree = 0.1,
                subsample = 0.95, objective = "binary:logistic",
                watchlist = watch, eval_metric = "error")
```

    ## [1]  train-error:0.351400    valid-error:0.382800 
    ## [2]  train-error:0.303050    valid-error:0.341200 
    ## [3]  train-error:0.274900    valid-error:0.324800 
    ## [4]  train-error:0.248250    valid-error:0.306800 
    ## [5]  train-error:0.227900    valid-error:0.278200 
    ## [6]  train-error:0.212950    valid-error:0.268400 
    ## [7]  train-error:0.199100    valid-error:0.245800 
    ## [8]  train-error:0.190650    valid-error:0.236200 
    ## [9]  train-error:0.177850    valid-error:0.225600 
    ## [10] train-error:0.169450    valid-error:0.215800

The above demonstration deliberately omits steps such as model tuning but hopefully it illustrates a useful workflow that makes the most of the FeatureHashing and XGBoost packages.
