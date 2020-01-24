library(tidyverse)
library(ggplot2)
df <- read.csv("Food_Pantry_Cleaned_weather_population.csv")
df
df$Individuals

#Data Explore: County Individual Distribution
ggplot(df, aes(Individuals)) + 
  geom_density(alpha = .5,fill="#024FC3") + 
  theme_bw()+
  ggtitle("Density Graph - County X Individual")+
  facet_wrap(~County)

ggplot(df,aes(Individuals)) + 
  geom_histogram(alpha = .5,fill="#024FC3",bins=50) + 
  theme_bw()+
  ggtitle("Hist Graph - County X Individual")+
  facet_wrap(~County)  

df%>%
  group_by(County)%>%
  summarise(meadia=median(Individuals,na.rm = TRUE),mean=mean(Individuals,na.rm = TRUE),sd=sd(Individuals,na.rm = TRUE))

df%>%
  summarise(sd=sd(Individuals,na.rm = TRUE))



#Explore Individual to see segment
#overall indiv
ggplot(df)+geom_boxplot(mapping=aes(x='Individuals',y=Individuals))
ggplot(df)+geom_boxplot(mapping=aes(x='Individuals',y=Individuals))+facet_wrap(~County)

#overall 1000: outlier
#each group 200 - separate 5 group
# 6 group: extremely large
df <- df%>%
  mutate(Size_Individuals= ifelse((Individuals<=200) & (Individuals>=0), "SuperSmall", 0 ))%>%
  mutate(Size_Individuals= ifelse((Individuals<=400) & (Individuals>=201), "Small", Size_Individuals))%>%
  mutate(Size_Individuals= ifelse((Individuals<=600) & (Individuals>=400), "Medium", Size_Individuals))%>%
  mutate(Size_Individuals= ifelse((Individuals<=800) & (Individuals>=601), "Large", Size_Individuals))%>%
  mutate(Size_Individuals= ifelse((Individuals>=801), "SuperLarge", Size_Individuals))%>%
  mutate(Size_Individuals=as.factor(Size_Individuals))

#save category
write.table(df, file = "Food_Pantry_forClassification.csv", sep = ",", col.names = NA,
            qmethod = "double")


#select column and process data
foodbankML <- df %>%
  mutate(Month = as.factor(Month))%>%
  mutate(FBNI.Agency = as.factor(FBNI.Agency))%>%
  mutate(Rural.Area = as.factor(Rural.Area))%>%
  mutate(County = as.factor(County))%>%
  mutate(Wind = as.factor(Wind))%>%
  mutate(Condition = as.factor(Condition))%>%
  select(Month,County,FBNI.Agency,Rural.Area,Individuals,Households,Temperature,Humidity,Wind,Condition,Population,Size_Individuals)
foodbankML
foodbankMLwithoutna <- na.omit(foodbankML, cols='Size_Individuals')
foodbankMLwithoutna


#Explore distribution of Y
foodbankMLwithoutna%>%
  ggplot(aes(Size_Individuals)) +
  geom_histogram(alpha = .5,fill="#024FC3",stat="count")


#balance

#split train and test
foodbankMLwithoutna <- foodbankMLwithoutna%>%
  select(Month,FBNI.Agency,County,Rural.Area,Temperature,Humidity,Wind,Condition,Population,Size_Individuals)


library(caTools)
set.seed(1234)
food_set <- foodbankMLwithoutna%>%
  pull(Size_Individuals)%>%
  sample.split(SplitRatio = 0.65)
df_train <- subset(foodbankMLwithoutna, food_set==TRUE)
df_test <- subset(foodbankMLwithoutna, food_set==FALSE)

#sample_set <- sample(nrow(foodbankMLwithoutna), round(nrow(foodbankMLwithoutna)*.65), replace = FALSE)
#df_train <- foodbankMLwithoutna[sample_set, ]
#df_test <- foodbankMLwithoutna[-sample_set, ]



#Imbalance Data
library(DMwR)
df_train_t <- SMOTE(Size_Individuals~., data.frame(df_train),perc.over =800, per.under=700,k = 2)
#df_train_t <- SMOTE(Size_Individuals~., data.frame(df_train_t),perc.over =900, per.under=800,k = 2)

round(prop.table(table(select(foodbankMLwithoutna,Size_Individuals))),4)*100
round(prop.table(table(select(df_train_t,Size_Individuals))),4)*100

foodbankMLwithoutna%>%
  ggplot(aes(Size_Individuals)) +
  geom_histogram(alpha = .5,fill="#024FC3",stat="count")
df_train_t%>%
  ggplot(aes(Size_Individuals)) +
  geom_histogram(alpha = .5,fill="#024FC3",stat="count")




#################################### MODEL ##############################3

#Normal Tree
library(rpart)

tree_mod <-
  rpart(
    Size_Individuals~.,
    method = "class",
    data = df_train,
    control = rpart.control(cp = 0.001)
  )
tree_mod
library(rpart.plot)
rpart.plot(tree_mod)
tree_pred <- predict(tree_mod, df_test, type = "class")
#confusion matrix
table(df_test$Size_Individuals,tree_pred)

pred<-tree_pred
test <-df_test$Size_Individuals

library(caret)
accuracy <- mean(test == pred)
confmat <- confusionMatrix(pred, test)
kappa <- as.numeric(confmat$overall["Kappa"])
comparisons <- tibble(approach="Classification Tree", accuracy = accuracy,kappa = kappa) 





#RandomForest
grid <- expand.grid(.mtry = c(3, 6, 9,12))
ctrl <-
  trainControl(method = "cv",
               number = 5,
               selectionFunction = "best")
set.seed(1234)
rf.mod <-
  train(
    Size_Individuals ~ ., data = df_train,
    method = "rf",
    trControl = ctrl,
    tuneGrid = grid
  )

rf.mod
#Random Forest Performance
rf_pred <- predict(rf.mod, df_test, type = "raw")
#confusion matrix
table(df_test$Size_Individuals,rf_pred)
pred<-rf_pred
test <-df_test$Size_Individuals
accuracy <- mean(test == pred)
confmat <- confusionMatrix(pred, test)
kappa <- as.numeric(confmat$overall["Kappa"])
comparisons <- comparisons %>%
  add_row(approach="Random Forest", accuracy = accuracy,kappa = kappa) 




#XGB
grid <- expand.grid(
  nrounds = 20,
  max_depth = c(4, 6, 8),
  eta =  c(0.1, 0.3, 0.5),
  gamma = 0.01,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = c(0.5, 1)
  
)
ctrl <-
  trainControl(method = "cv",
               number = 5,
               selectionFunction = "best")
set.seed(1234)
xgb.mod <-
  train(
    Size_Individuals ~ ., data = df_train,
    method = "xgbTree",
    trControl = ctrl,
    tuneGrid = grid
  )

xgb.mod

#Predict
xgb.pred <- predict(xgb.mod, df_test, type = "raw")
pred <- xgb.pred
#Performance
#confusion matrix
table(df_test$Size_Individuals,pred)
accuracy <- mean(test == pred)
confmat <- confusionMatrix(pred, test)
kappa <- as.numeric(confmat$overall["Kappa"])

comparisons <- comparisons %>%
  add_row(approach="XGB", accuracy = accuracy,kappa = kappa) 

comparisons
