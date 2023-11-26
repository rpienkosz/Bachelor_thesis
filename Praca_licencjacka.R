rm(list = ls())

setwd("C:/Users/rafal/Desktop/Praca licencjacka")
getwd()

library(recommenderlab)
library(ggplot2)                      
library(data.table)
library(reshape2)
library(tidyverse)
library(tidytext)
library(SnowballC)
library(lsa)

#---------------------------------------------------------------------------------------------#
#                           PREPROCESING DATA                                                 #
#---------------------------------------------------------------------------------------------#

movie_data_new <- read.csv("movies_metadata.csv")
rating_data_new <- read.csv("ratings_small.csv")
links <- read.csv("links_small.csv")
keywords <- read.csv("keywords.csv")

str(movie_data_new)
str(rating_data_new)
str(links)

movie_data_new$id <- as.integer(as.character(movie_data_new$id))
movie_data_new2 <- movie_data_new %>% select(id,title, release_date, genres,overview) %>% 
  inner_join(links, by=c("id"="tmdbId")) %>% inner_join(keywords, by=c("id"="id")) %>% 
  drop_na() %>% mutate(year = str_sub(release_date,end= 4)) %>% 
  select(movieId, title, year, genres, keywords, overview)

 

movie_data_new2$genres <- movie_data_new2$genres %>% str_remove_all("name") %>% str_remove_all("id") %>% 
  str_remove_all("[1234567890]") %>% str_remove_all("Fiction") %>% str_remove_all("TV") %>% str_remove_all("Movie")
movie_data_new2$genres <- gsub("[^[:alnum:]]", " ", movie_data_new2$genres)
movie_data_new2$genres <- gsub("\\s+", "|", str_trim(movie_data_new2$genres))
movie_data_new2$genres <- movie_data_new2$genres %>% str_replace_all("Science", "SciFi")

movie_data_new2$keywords <- movie_data_new2$keywords %>% str_remove_all("name") %>% str_remove_all("id") %>% 
  str_remove_all("[1234567890]") %>% str_remove_all("TV") %>% str_remove_all("Movie") %>% str_remove_all("animation")
movie_data_new2$keywords <- gsub("[^[:alnum:]]", " ", movie_data_new2$keywords)
movie_data_new2$keywords <- gsub("\\s+", "|", str_trim(movie_data_new2$keywords))
movie_data_new2 <- unique(movie_data_new2)

#---------------------------------------------------------------------------------------------#
#                      EXPLORATORY DATA ANALYSIS                                              #
#---------------------------------------------------------------------------------------------#

a <- rating_data_new %>% group_by(rating) %>% ggplot(aes(x=rating)) + geom_bar() + theme_minimal() + 
  labs(x = 'Rating', y='Count', title='Distribution of ratings', caption='Data Source: Movielens')
a
a2 <- rating_data_new %>% group_by(userId) %>% count() %>% arrange(desc(n)) %>% ungroup() %>% summarise(max=max(n), mean=mean(n),median=median(n), min=min(n)) 
a2
a3 <- rating_data_new %>% group_by(movieId) %>% count() %>% arrange(desc(n)) #%>% ggplot(aes(x=movieId)) + geom_bar() + theme_minimal()
a3

#---------------------------------------------------------------------------------------------#
#                      1. CONTENT BASED FILTERING SYSTEM                                      #
#---------------------------------------------------------------------------------------------#

#------------------------------------------------#
# Creating item profiles (TF-IDF)

a <- movie_data_new2[,c(1,4)]
a <- as.tibble(a)

movie_genres <- a %>%
  unnest_tokens(genre, genres) %>% count(movieId, genre, sort = TRUE)
total_words <- movie_genres %>% group_by(movieId) %>% summarize(total = sum(n))
movie_genres <- left_join(movie_genres, total_words)
movie_genres <- movie_genres %>% bind_tf_idf(genre, movieId, n) %>% select(movieId, genre, tf_idf)
movie_genres <- movie_genres %>% spread(genre, tf_idf, fill=0)

a2 <- movie_data_new2[,c(1,5)]
a2 <- as.tibble(a2)

movie_keywords <- a2 %>%
  unnest_tokens(keyword, keywords) %>% anti_join(stop_words, by=c("keyword"="word")) %>% 
  mutate(keyword=wordStem(keyword)) %>% count(movieId, keyword, sort = TRUE) 
total_words2 <- movie_keywords %>% group_by(movieId) %>% summarize(total = sum(n))
movie_keywords <- left_join(movie_keywords, total_words2) %>% filter(total >5)
q <- movie_keywords %>% group_by(keyword) %>% count() %>% filter(n>4) %>% select(keyword)
movie_keywords <- movie_keywords %>% inner_join(q, by="keyword")

movie_keywords <- movie_keywords %>%
  bind_tf_idf(keyword, movieId, n) %>% select(movieId, keyword, tf_idf) %>% 
  arrange(movieId, desc(tf_idf))
movie_keywords2 <- movie_keywords %>% spread(keyword, tf_idf, fill=0)

SearchMatrix <- movie_data_new2[,1:2] %>% inner_join(movie_genres, by="movieId") %>% 
  left_join(movie_keywords2, by="movieId")
SearchMatrix[is.na(SearchMatrix)] <- 0

words <- SearchMatrix[,-c(1,2)]


#------------------------------------------------#
# Creating user profiles

binaryratings <- rating_data_new
binaryratings$binary_rating <- ifelse(binaryratings$rating > 3,1, -1)
binaryratings <- binaryratings %>% select(userId, movieId, binary_rating)

user_mat <- dcast(binaryratings, movieId~userId, value.var = "binary_rating", na.rm=FALSE)
for (i in 1:ncol(user_mat)){
  user_mat[which(is.na(user_mat[,i]) == TRUE),i] <- 0
}
all_movies_id <- SearchMatrix[,1] %>% as.data.frame()
user_mat <- user_mat %>% right_join(all_movies_id, by=c("movieId"="."))
user_mat[is.na(user_mat)] = 0

#remove movieIds col. Rows are movieIds, cols are userIds
user_mat2 = user_mat[,-1] #user-profile matrix
word_mat <- SearchMatrix[,-c(1,2)]

sum(ifelse(is.na(user_mat2),1,0))

dim(user_mat2) #movies-users matrix
dim(word_mat) #movies-words matrix
user_mat2 <- as.matrix(user_mat2)
word_mat <- as.matrix(word_mat)
user_mat3 <- t(user_mat2)
dim(user_mat3)

sum(ifelse(is.na(user_mat3),1,0))
sum(ifelse(is.na(word_mat),1,0))

#Calculate dot product for User Profiles
result <- user_mat3 %*% word_mat
dim(result)
result2 <- t(result)
result2 <- ifelse(result2 > 0,1,0)
dim(result2)

word_mat2 <- t(word_mat)
dim(word_mat2)

## the cosine measure for all document vectors of a matrix
result3 <- result2[,1]
user1_cb <- matrix(0,1,9047)

for(i in 1:ncol(word_mat2)) {
  user1_cb[1,i] <- cosine(result3, word_mat2[,i])
}
user1_cb2 <- user1_cb[1,] %>% as.data.frame()
colnames(user1_cb2)[1] <- "cos"
user1_cb2 <- cbind(user1_cb2, SearchMatrix) %>% select(cos, movieId, title) %>% 
  arrange(desc(cos)) %>% top_n(5,cos)


#---------------------------------------------------------------------------------------------#
#                      2. COLLABORATIVE FILTERING SYSTEM                                      #
#---------------------------------------------------------------------------------------------#

#------------------------------------------------#
# Creating movie ratings matrix (sparse matrix)

ratingMatrix <- dcast(rating_data_new, userId~movieId, value.var = "rating", na.rm=FALSE)
ratingMatrix <- as(as.matrix(ratingMatrix[,-1]),"realRatingMatrix")
ratingMatrix <- ratingMatrix[rowCounts(ratingMatrix) > 50,colCounts(ratingMatrix) > 50]
ratingMatrix

dim(ratingMatrix)
# similarity_users <- similarity(ratingMatrix[1:10, ], method ="cosine", which = "users")
# as.matrix(similarity_users)
# similarity_items <- similarity(ratingMatrix[, 1:10], method =
#                                  "cosine", which = "items")
# as.matrix(similarity_items)
#a <- rating_data %>% group_by(movieId) %>% summarize(avg_rating = mean(rating), count=n()) %>% arrange(desc(avg_rating)) %>% left_join(movie_data %>% select(movieId, title), by="movieId") 
#image(ratingMatrix, main = "Heatmap of the rating matrix")
#plot(a$avg_rating, a$count)

#------------------------------------------------#
# Geting train and test data
set.seed(123)
#ratingMatrix <- normalize(ratingMatrix)
sampled_data <- sample(x = c(TRUE, FALSE),
                       size = nrow(ratingMatrix),
                       replace = TRUE,
                       prob = c(0.8, 0.2))
training_data <- ratingMatrix[sampled_data, ]
testing_data <- ratingMatrix[!sampled_data, ]
training_data
testing_data

recommendation_system <- recommenderRegistry$get_entries(dataType ="realRatingMatrix")
recommendation_system

#---------------------------------------------------------------------------------------------#
#                      2.1 ITEM-BASED COLLABORATIVE FILTERING SYSTEM                          #
#---------------------------------------------------------------------------------------------#

#------------------------------------------------#
# Training model

recommen_model <- Recommender(training_data, method = "IBCF", parameter = list(k = 30))
predicted_recommendations <- predict(recommen_model, testing_data, n = 5) # the number of items to recommend to each user

#------------------------------------------------#
# Recommedning/predicting

user1_IBCF <- predicted_recommendations@items[[1]] # recommendation for the first user
movies_user1_IBCF <- predicted_recommendations@itemLabels[user1_IBCF]
for (index in 1:5){
  movies_user1_IBCF[index] <- as.character(subset(movie_data_new2,movie_data_new2$movieId == movies_user1_IBCF[index])$title)
}
movies_user1_IBCF %>% as.data.frame()

#---------------------------------------------------------------------------------------------#
#                      2.2 USER-BASED COLLABORATIVE FILTERING SYSTEM                          #
#---------------------------------------------------------------------------------------------#

#------------------------------------------------#
# Training model

recommen_model2 <- Recommender(training_data, method = "UBCF", 
                               parameter=list(method="Cosine",nn=30))
predicted_recommendations2 <- predict(recommen_model2, testing_data, n = 5) # the number of items to recommend to each user

#------------------------------------------------#
# Recommedning/predicting

user1_UBCF <- predicted_recommendations2@items[[1]] # recommendation for the first user
movies_user1_UBCF <- predicted_recommendations@itemLabels[user1_UBCF]
for (index in 1:5){
  movies_user1_UBCF[index] <- as.character(subset(movie_data_new2,movie_data_new2$movieId == movies_user1_UBCF[index])$title)
}
movies_user1_UBCF %>% as.data.frame()

#---------------------------------------------------------------------------------------------#
#                      2.3 MATRIX-FACTORIZATION WITH STOCHASTIC GRADIENT DESCENT              #
#---------------------------------------------------------------------------------------------#

#------------------------------------------------#
# Training model

recommen_model3 <- Recommender(data = training_data, method = "SVDF", parameter=list(k = 10, gamma = 0.015, lambda = 0.001,min_improvement = 1e-06, 
                                                                                     min_epochs = 50, max_epochs = 200, verbose = FALSE))
predicted_recommendations3 <- predict(recommen_model3, testing_data, n=5) # the number of items to recommend to each user

#------------------------------------------------#
# Recommedning/predicting

user1_SVDF <- predicted_recommendations3@items[[1]] # recommendation for the first user
movies_user1_SVDF <- predicted_recommendations3@itemLabels[user1_SVDF]
for (index in 1:5){
  movies_user1_SVDF[index] <- as.character(subset(movie_data_new2,movie_data_new2$movieId == movies_user1_SVDF[index])$title)
}
movies_user1_SVDF %>% as.data.frame()

#---------------------------------------------------------------------------------------------#
#                      2.4 MATRIX-FACTORIZATION WITH ALS                                      #
#---------------------------------------------------------------------------------------------#

#------------------------------------------------#
# Training model

recommen_model4 <- Recommender(data = training_data, method = "ALS")
predicted_recommendations4 <- predict(recommen_model4, testing_data, n=5) # the number of items to recommend to each user

#------------------------------------------------#
# Recommedning/predicting

user1_ALS <- predicted_recommendations4@items[[1]] # recommendation for the first user
movies_user1_ALS <- predicted_recommendations4@itemLabels[user1_ALS]
for (index in 1:5){
  movies_user1_ALS[index] <- as.character(subset(movie_data_new2,movie_data_new2$movieId == movies_user1_ALS[index])$title)
}
movies_user1_ALS %>% as.data.frame()


movies_user1_UBCF %>% as.data.frame()
movies_user1_IBCF %>% as.data.frame()
movies_user1_SVDF %>% as.data.frame()
movies_user1_ALS %>% as.data.frame()


#---------------------------------------------------------------------------------------------#
#                      3. HYBRID FILTERING SYSTEM                                             #
#---------------------------------------------------------------------------------------------#

recommen_model5 <- Recommender(data = training_data, method = "SVDF", parameter=list(k = 10, gamma = 0.015, lambda = 0.001,min_improvement = 1e-06, 
                                                                                     min_epochs = 50, max_epochs = 200, verbose = FALSE))
predicted_recommendations5 <- predict(recommen_model5, testing_data, n=100) # the number of items to recommend to each user

#------------------------------------------------#
# Recommedning/predicting

user1_Hybrid1 <- predicted_recommendations5@items[[1]] # recommendation for the first user
movies_user1_Hybrid1 <- predicted_recommendations5@itemLabels[user1_Hybrid1]
movies_user1_Hybrid1 <- movies_user1_Hybrid1 %>% as.data.frame() %>% rename("movieId"=".")
movies_user1_Hybrid1$movieId <- as.integer(as.character(movies_user1_Hybrid1$movieId))
movies_user1_Hybrid1 <- movies_user1_Hybrid1 %>% left_join(SearchMatrix, by="movieId") %>% drop_na()
movies_user1_Hybrid12 <- movies_user1_Hybrid1[,-c(1,2)]
movies_user1_Hybrid12 <- t(movies_user1_Hybrid12)
dim(movies_user1_Hybrid12)

## the cosine measure for all document vectors of a matrix
user1_hybrid <- matrix(0,1,98)

for(i in 1:ncol(movies_user1_Hybrid12)) {
  user1_hybrid[1,i] <- cosine(result3, movies_user1_Hybrid12[,i])
}

user1_hybrid2 <- user1_hybrid[1,] %>% as.data.frame() %>% rename('cos'=".")
user1_hybrid2 <- cbind(user1_hybrid2, movies_user1_Hybrid1) %>% select(cos, movieId, title) %>% 
  arrange(desc(cos)) %>% top_n(5,cos) %>% select(title)
user1_hybrid2














#---------------------------------------------------------------------------------------------#
#                      4. EVALUATION                                                          #
#---------------------------------------------------------------------------------------------#
recommendation_system$ALS_implicit_realRatingMatrix

e <- evaluationScheme(ratingMatrix, method="split", train=0.8,
                      given=5, goodRating=3, k=1)

r1 <- Recommender(getData(e, "train"), "UBCF",parameter=list(method="Cosine",nn=30))
p1 <- predict(r1, getData(e, "known"), type='ratings')

r2 <- Recommender(getData(e, "train"), "IBCF",parameter = list(k = 30))
p2 <- predict(r2, getData(e, "known"), type='ratings')

r3 <- Recommender(getData(e, "train"), "SVDF",parameter=list(k = 10, gamma = 0.015, lambda = 0.001,min_improvement = 1e-06, 
                                                             min_epochs = 50, max_epochs = 200, verbose = FALSE))
p3 <- predict(r3, getData(e, "known"), type='ratings')

r4 <- Recommender(getData(e, "train"), "ALS")
p4 <- predict(r4, getData(e, "known"), type='ratings')


error <- rbind(UBCF = calcPredictionAccuracy(p1, getData(e, "unknown")),
               IBCF = calcPredictionAccuracy(p2, getData(e, "unknown")),
               SVDF = calcPredictionAccuracy(p3, getData(e, "unknown")),
               ALS = calcPredictionAccuracy(p4, getData(e, "unknown")))
error


algorithms <- list("user-based CF" = list(name="UBCF", param=list(method="Cosine",nn=30)),
                   "item-based CF" = list(name="IBCF", param=list(k = 30)),
                   "Matrix Factorization SGD" = list(name="SVDF", param=list(k = 10, gamma = 0.015, lambda = 0.001,min_improvement = 1e-06, 
                                                                             min_epochs = 50, max_epochs = 200, verbose = FALSE)),
                   "Matrix Factorization ALS" = list(name="ALS", param=NULL))


results <- evaluate(e, algorithms)
avg(results)

legend="topleft"
plot(results, legend="topleft")
plot(results, "prec/rec",legend="bottomright")


#aac4 <- getData(e, "train") %>% as('matrix') %>% as.data.frame()
#aac3 <- getData(e, "unknown") %>% as('matrix') %>% as.data.frame()
#aac <- getData(e, "known") %>% as('matrix') %>% as.data.frame()
#aac1 <- predict(r1, getData(e, "known"), type='ratings') %>% as('matrix') %>% as.data.frame()
#aac2 <- predict(r5, getData(e, "known"), type='ratings') %>% as('matrix') %>% as.data.frame()

#hist(getRatings(normalize(ratingMatrix)), breaks=100)
#hist(getRatings(ratingMatrix), breaks=100)



