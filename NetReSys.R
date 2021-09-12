#--------------------------------  NetReSys.R  ----------------------------------------------------------
# This program is the development of a Netflix Recommendation System (NetReSys)
# It is part of the final course in the HarvardX Professional Certificate in Data Science.
# The challenge is to create and evaluate prediction models that will predict the ratings some 
# Netflix users would assign to some movies
#------ Notes
# Navigate through models using the pattern "#----- "
#------
# 13.09.2021  | Submission to HarvardX  |
#---------------------------------------------------------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(lubridate)
library(data.table)
library(caret)
library(recommenderlab)
library(ggthemes)
library(ggrepel)

options(digits = 6)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Number of Users, movies and genres
movielens %>%
  summarize(n_users = n_distinct(userId),        # 69878
            n_movies = n_distinct(movieId),      # 10677
            n_genres = n_distinct(genres))       # 797
movielens <- movielens %>% select(c(userId, movieId, rating))

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove temp Data
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#-----------------------------------------------------------------
#--- edx: Train data set
#--- validation: Test data set
#-----------------------------------------------------------------
#--- RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#-----------------------------------------------------------------
#----- MODEL 1: Movies & users effect
#-----------------------------------------------------------------
#--- mu is the average ratings of all movies across all users
mu <- mean(edx$rating)  # 3.512465
#--- Movie bias: b_i
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#--- User bias: b_u
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#--- Predict using item & user bias
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# 0.2% of predicted ratings are over 5
sum(predicted_ratings > 5) / length(predicted_ratings) # 2217/1M
# Non significant predixted ratings are negative
sum(predicted_ratings < 0) /length(predicted_ratings) # 18/1M

#--- Compute and store rmse
iu_effect_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- data_frame(Method="MODEL 1 : Movie & User Effect",
                           RMSE = iu_effect_rmse )
rmse_results %>% knitr::kable()

#--- Free memory
rm(movie_avgs, user_avgs, iu_effect_rmse)  
rm(mu, predicted_ratings)
#-----------------------------------------------------------------
#----- MODEL 2: MODEL 1 + Regularisation
# Regularisation permits to 
# penalise large estimate taht comes from small sample size
#-----------------------------------------------------------------
# Get lambdas that minimise RMSE

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE) 
train_edx <- edx[-test_index,]
temp_edx <- edx[test_index,]

#--- Make sure userId and movieId in validation set are also in edx set
test_edx <- temp_edx %>% 
  semi_join(train_edx, by = "movieId") %>%
  semi_join(train_edx, by = "userId")

#--- Add rows removed from validation set back into edx set
removed <- anti_join(temp_edx, test_edx)
train_edx <- rbind(train_edx, removed)

#--- Free memory
rm(temp_edx, test_index, removed)


mu <- mean(edx$rating)  
lambdas <- seq(0,10,0.2)

#-----------------------------------------------------------------
#----- MODEL 2a: MODEL 1 + duo-Regularisation
# Regularisation permits to 
# penalise large estimate taht comes from small sample size
#-----------------------------------------------------------------
# Get lambdas that minimise RMSE

# Calculate lambda_1:
# regularization parameter that penalises low ratings movies
rmses <- sapply(lambdas, function(l1){
  b_i <- train_edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l1))
  
  predicted_ratings <- test_edx %>% 
    left_join(b_i, by = "movieId") %>%
    mutate(pred= mu + b_i) %>% .$pred
  
  RMSE(predicted_ratings, test_edx$rating)
})

qplot(lambdas,rmses)

# lambda_1 is set to penalise movie effect
lambda_1 <- lambdas[which.min(rmses)]  
lambda_1

# Calculate lambda_2:
# regularization parameter that penalises low raters users
rmses <- sapply(lambdas, function(l2){
  b_u <- train_edx %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu)/ (n() + l2))
  
  predicted_ratings <- test_edx %>% 
    left_join(b_u, by = "userId") %>%
    mutate(pred= mu + b_u) %>% .$pred
  
  RMSE(predicted_ratings, test_edx$rating)
})
qplot(lambdas,rmses)

lambda_2 <- lambdas[which.min(rmses)] 
lambda_2

# Predict using item & user bias and regularisation
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda_1))

b_u <- edx %>%
  left_join(b_i, by= "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda_2))

predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

#--- Compute and store rmse
regul_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="MODEL 2a: MODEL 1 + duo-regularisation",  
                                     RMSE = regul_rmse ))
rmse_results %>% knitr::kable()

#-----------------------------------------------------------------
#----- MODEL 2b: MODEL 1 + uni-Regularisation
# Regularisation permits to 
# penalise large estimate that comes from small sample size
#-----------------------------------------------------------------
# Calculate lambda:
# regularization parameter that penalises low ratings movies
rmses <- sapply(lambdas, function(l){
  
  b_i <- train_edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  
  b_u <- train_edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n() + l))
  
  predicted_ratings <-
    test_edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  RMSE(predicted_ratings, test_edx$rating)
})

qplot(lambdas,rmses)
lambda <- lambdas[which.min(rmses)] 
lambda

#--- Compute predicted ratings
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda))

b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))

predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

#--- Compute and store rmse
regul1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="MODEL 2b: MODEL 1 + uni-regularisation",  
                                     RMSE = regul1_rmse ))
rmse_results %>% knitr::kable()

#--- Free memory
rm(lambda_1, lambda_2, rmses, regul_rmse, predicted_ratings)


#-----------------------------------------------------------------
#----- MODEL 3 : MODEL 2b + variable average
# We use var_mu (30 values around mu) 
# Penalty lambda is adjusted for each value of the average
# Best couple (average, lambda) is kept to measure RMSE
#-----------------------------------------------------------------

#-- Define average variability
var_mu <- seq(3.4, 4.0, length.out=20)

params <- lapply(var_mu, function(avg){
  
  #-- Adjust penalty lambda  
  lambdas <- seq(0,10,0.2)
  rmses_l <- sapply(lambdas, function(l){
    
    b_i <- train_edx %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - avg) / (n() + l))
    
    b_u <- train_edx %>%
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - avg)/(n() + l))
    
    predicted_ratings <-
      test_edx %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = avg + b_i + b_u) %>%
      .$pred
    
    RMSE(predicted_ratings, test_edx$rating)
    
  })
  
  list(penalty = lambdas[which.min(rmses_l)], 
       rmse = min(rmses_l))
})

#--- Plot RMSE vs average
df <-  data.frame(average = var_mu, 
                  rmses = unlist(lapply(params, "[", 2)))
with(df, qplot(average, rmses))

df <- data.frame(average = var_mu, 
                 rmses   = unlist(lapply(params, "[", 2)),
                 lambda  = unlist(lapply(params, "[", 1))
)
df_best <- as.data.frame(df[rmse_min,])
rmse_min <- which.min(unlist(lapply(params, "[", 2)))
best_mu <- var_mu[rmse_min]
best_l  <- unlist(lapply(params, "[", 1))[rmse_min]
mu <- mean(train_edx$rating)

# Plot Avgs vs rmses
df %>%
  ggplot(aes(average,rmses, label= round(average,2))) + 
  labs(title= substitute(paste("RMSE vs ", mu[v], "=")),
       x = expression(paste(mu[v])),
       y = "RMSE") +
  geom_vline(xintercept= df_best$average, lty= 2, colour = "green",  size= 0.3) +
  geom_hline(yintercept= df_best$rmses,  lty= 2, colour = "black", size= 0.3) +
  geom_vline(xintercept= mu, lty= 2, colour = "red", size= 0.2) + 
  geom_label(aes(mu, mean(rmses)),       
             label= round(mu, 2),
             color= "red") +
  geom_label(aes(best_mu, mean(rmses)),
             label= round(best_mu, 2),
             color= "green") +
  theme(plot.title = element_text(color="steelblue", size=14,  face= "bold", hjust = 0.5),
        axis.title.x = element_text(color="steelblue", size=11, face= "bold"),
        axis.title.y = element_text(color="steelblue", size=11, face= "bold")) 
geom_line() +
  
  
#--- Compute predicted ratings
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - best_mu) / (n() + best_l))

b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - best_mu)/(n() + best_l))

predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = best_mu + b_i + b_u) %>%
  .$pred

#-- COmpute and store rmse
varmu_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="MODEL 3 : Model 2b + variable average",  
                                     RMSE = varmu_rmse ))
rmse_results %>% knitr::kable()

#--- Free memory
rm(predicted_ratings, df, df_best, 
   best_l, rmse_min, var_mu, varmu_rmse, lmbdas, lambda)

#-----------------------------------------------------------------
#----- MODEL 4: Model 3 + Factorisation (SVD)
#-----------------------------------------------------------------
mu <- best_mu # and not mean(edx$rating) any more!
rating_residuals <- edx %>%
  left_join(b_i, by='movieId') %>% 
  left_join(b_u, by='userId') %>%
  mutate(residu= rating - mu - b_i - b_u) %>%
  select(userId, movieId, residu) %>%
  spread(movieId, residu)  %>%
  as.matrix()
# as('realRatingMatrix')

# Average of matrix values
mat_mu <- mean(rating_residuals[,-1][!is.na(rating_residuals[,-1])])
# How many zeros?
nb_zeros <- sum(rating_residuals[,-1][rating_residuals[,-1]==0])

rating_residuals[is.na(rating_residuals)] <- 0
rownames(rating_residuals) <- rating_residuals[,1]
rating_residuals <- rating_residuals[,-1]

svd <- svd(rating_residuals)
d <- as.matrix(svd$d)
expl_rate <- cumsum(d^2)/sum(d^2) * 100
qplot(1:dim(d)[1], expl_rate)
k1 <- min(which(expl_rate> 80))  #k1 explains 80% 0f the variability
k2 <- min(which(expl_rate> 90))  #k2 explains 90% 0f the variability

#--- Rebuid -reduced residuals
svd_residuals <- with(svd, 
                      sweep(u[, 1:k2], 2, d[1:k2], FUN= "*") %*% t(v[, 1:k2]))  #$$$ k1 and not k2
svd_residuals  <- cbind(userId = as.numeric(rownames(rating_residuals)), svd_residuals)
colnames(svd_residuals) <- c("userId", colnames(rating_residuals))

#--- Transform to -reduced residuals data frame
df_residuals <- svd_residuals %>%
  as.data.frame() %>% 
  gather(-userId, key= movieId, value= residu, convert= TRUE) %>%
  arrange(userId, movieId) 

#--- Compute predicted ratings
predicted_ratings <- df_residuals %>% 
  semi_join(validation, by = c("userId", "movieId")) %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u + residu) %>%
  .$pred

#--- Compute aand store rmse
svd_rmse <- RMSE(predicted_ratings, validation$rating)
svd_rmse
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="MODEL 4 : MODEL 3 + factorisation (SVD)",  
                                     RMSE = svd_rmse ))
rmse_results %>% knitr::kable()

#--- Free nenory
rm(df_residuals,rating_residuals,svd, svd_residuals, predicted_ratings,expl_rate, k2, d)

#-----------------------------------------------------------------
#----- MODEL 5: MODEL 2 + recommendation algorithm
# Package recommenderlab is used as a framework to evaluate and
# compare various recommendation algorithm
# The run is very long for a desapointing result 
#-----------------------------------------------------------------
mu <- best_mu  # mean(edx$rating) 
#--- Build the rating matrix
rating_residuals <- edx %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(residu= rating - mu - b_i - b_u) %>%
  select(userId, movieId, residu) %>%
  spread(movieId, residu)  %>%
  as.matrix() %>%
  as('realRatingMatrix')

scheme_cross <- rating_residuals %>% 
  evaluationScheme(method = "split",
                   k      = 5, 
                   train  = 0.8,  
                   given  = -1)
scheme_cross

# Learns a recommender model from given data.
recom <- Recommender(getData(scheme_cross, 'train',5), 
                     method = "SVD",  
                     param = list(k= 2300))
recom

# Run predict for users 
# ... found in validation set
ind <- which((unique(edx$userId) %in% unique(validation$userId))) #$$$
# Pass the Recommender and validation realRatingMatrix    
pred_mat <- as(predict(recom, ind, data = rating_residuals, type= "ratings"), "matrix") 
pred_mat[is.na(pred_mat)] <- 3.87 #???

# Set predicted ratings
# ... for movies in validation set
pred_mat <- pred_mat[, as.character(unique(validation$movieId))]
pred_mat <- cbind(userId = unique(edx$userId)[ind], pred_mat)
pred_df <- pred_mat %>%
  as.matrix() %>%
  as.data.frame() %>% 
  gather(-userId, key= movieId, value= residu, convert= TRUE) %>%
  mutate(um= paste(userId, movieId, sep = '_')) %>%
  arrange(userId, movieId)

validation <- validation %>%
  mutate(um= paste(userId, movieId, sep = '_'))

predicted_ratings <- pred_df %>% 
  semi_join(validation, by = c("userId", "movieId")) %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u + residu) %>%
  .$pred

recom_rmse <- RMSE(predicted_ratings, validation$rating)
recom_rmse

# Update results
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="MODEL 5: Model2 + Recommendation algo",  
                                     RMSE = recom_rmse ))
rmse_results %>% knitr::kable()

rm(rating_residuals, pred_mat, pred_df, recom, scheme_cross,
   ind,predicted_ratings, recom_rmse)


