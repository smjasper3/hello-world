##################################################################################
#0. Load in data and split into train/test
##################################################################################
library(AmesHousing)
library(dplyr)
library(rpart)
library(ggplot2)
library(tidyverse)
set.seed(12345)
ames <- ames_raw

#Add a new binary column to indicate if the SalePrice is greater than $160,000 (median price)
ames = ames %>% 
  mutate(SalePriceBinary = ifelse(SalePrice >=160000, 1, 0))

#Split into train/test
train = ames %>% sample_frac(0.7) #70% of data points are set aside for training (can be changed)
test <- anti_join(ames, train, by = 'Order') #The remainder are assigned for testing

#Remove Order and PID columns. Remove ms.subclass variable (causes problems with predicting later)
# train = train %>% 
#   dplyr::select(-c(Order,PID,`MS SubClass`,Utilities))
# 
# test = test %>% 
#   dplyr::select(-c(Order,PID,`MS SubClass`,Utilities))

#Only select a few columns for this tree (factor levels in test that are not in train cause errors)
train = train %>% 
  dplyr::select(c(SalePriceBinary, SalePrice, Neighborhood,`Year Built`, `Overall Qual`, `Gr Liv Area`))

test = test %>% 
  dplyr::select(c(SalePriceBinary, SalePrice, Neighborhood,`Year Built`, `Overall Qual`, `Gr Liv Area`))

##################################################################################
#1. CART - Classification and Regression Trees
##################################################################################
#1.a. Create classification tree to predict SalePriceBinary
BC.tree = rpart(SalePriceBinary ~ . -SalePrice , data=train, method='class',
                parms = list(split='gini')) ## or 'information'
summary(BC.tree)

#Print the actual tree
print(BC.tree)

library(rpart.plot)
rpart.plot(BC.tree)

#Variable importance
BC.tree$variable.importance

varimp.data = data.frame(BC.tree$variable.importance)
varimp.data$names = as.character(rownames(varimp.data))

ggplot(data = varimp.data,aes(x = fct_reorder(names,BC.tree.variable.importance), y = BC.tree.variable.importance)) +
  geom_bar(stat="identity") +
  coord_flip() +
  labs(x="Variable Name",y="Variable Importance")
  
##Calculate accuracy (i.e., misclassification rate) for train and test datasets
#Use CART model to predict new values for train and test datasets
trainscores = predict(BC.tree, type='class') #Class for classification trees; vector for regression trees
testscores = predict(BC.tree, test, type='class')

#Training/testing misclassification rate:
sum(trainscores!=train$SalePriceBinary)/nrow(train)
sum(testscores!=test$SalePriceBinary)/nrow(test)


##1.b. Create regression tree to predict SalePrice
R.tree = rpart(SalePrice ~ . -SalePriceBinary , data=train
               , method='anova' #Regression tree
               , control = rpart.control(minsplit = 10)) #the minimum number of observations that must exist in a node in order for a split to be attempted.
summary(R.tree)
printcp(R.tree)

#Variable importance
library(rpart.plot)
rpart.plot(R.tree)

#Prune tree based on cp value
R2.tree<-prune(R.tree,cp=0.05)
printcp(R2.tree)

#Use CART regression model to predict new values for train and test datasets
trainscores = predict(R2.tree, type='vector') #Class for classification trees; vector for regression trees
testscores = predict(R2.tree, test, type='vector')

#Training/testing MAE and MAPE (regression):
mean(abs(trainscores-train$SalePrice))
mean(abs((trainscores-train$SalePrice)/train$SalePrice))

mean(abs(testscores-test$SalePrice))
mean(abs((testscores-test$SalePrice)/test$SalePrice))

#Visualize cp values
plotcp(R.tree)



##################################################################################
#2. KNN - K-nearest neighbors
##################################################################################

#Use same train and test as before but create separate datasets for x's and y's
#In order to run knn in R, all categorical variables must be converted into numeric
train.x=subset(train,select=-c(SalePrice, SalePriceBinary, Neighborhood))
#train.x = sapply(train, unclass)
train.y=as.factor(train$SalePriceBinary)

test.x=subset(test,select=-c(SalePrice, SalePriceBinary, Neighborhood))
test.y=as.factor(test$SalePriceBinary)

#Run knn
library(class)
predict.test=class::knn(train.x,test.x,train.y,k=3)

#Determine misclassification rate
sum(predict.test != test.y)/length(test.y)

#Tuning k using caret
library(caret)
set.seed(12345)
grid = expand.grid(k = seq(1:20))

tuned_knn = caret::train(factor(SalePriceBinary) ~., method= "knn",
                    data = subset(train, select = -c(SalePrice, Neighborhood)),
                    trControl = trainControl(method = 'cv',
                                             number = 3,
                                             search = "grid"),
                    tuneGrid = grid)
tuned_knn$results

#Plot results to determine which k maximizes accuracy
ggplot(tuned_knn$results, aes(x=k, y=Accuracy)) +
geom_line()

#The following line of code works as well
tuned_knn$results[which.max(tuned_knn$results$Accuracy),]


#Extra - Dummy coding columns

# library(fastDummies)
# train.x=subset(dummy_cols(train, select_columns = 'Neighborhood')
#                ,select=-c(SalePrice,SalePriceBinary,Neighborhood))
# train.y=as.factor(train$SalePriceBinary)
# 
# test.x=subset(dummy_cols(test, select_columns = 'Neighborhood')
#               ,select=-c(SalePrice,SalePriceBinary,Neighborhood))
# test.y=as.factor(test$SalePriceBinary)




