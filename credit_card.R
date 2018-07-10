#Import libararies
library(ggplot2)

#Import the dataset
dataset=read.csv("C:/Rise_Internship/Datasets/Xbank_Credit_Payment.csv")
dataset=dataset[-c(1,4,6)]

str(dataset)

#Converting int into factor datatype
for(i in c(2,3,4:9,22))
{    
  dataset[,i]=as.factor(dataset[,i])   }

str(dataset)


#Splitting into training and testing datasets
library(caTools)
set.seed(123)
split=sample.split(dataset$default.payment.next.month,SplitRatio = 0.8)
training_set=subset(dataset,split==TRUE)
testing_set=subset(dataset,split==FALSE)

#Feature Scaling
training_set[,c(1,10:21)]=scale(training_set[,c(1,10:21)])
testing_set[,c(1,10:21)]=scale(testing_set[,c(1,10:21)])

#-----------------------------------------------------------------------------------

#Fitting logistic regression to the training set
classifier=glm(formula=default.payment.next.month ~.,
               family = binomial,
               data=training_set)
summary(classifier)


#Predicting the test_set results 
pro_pred=predict(classifier,
                 type='response',
                 newdata = testing_set[-22])
y_pred= ifelse(pro_pred>0.5,1,0)

#Confusion Matrix
library(caret)
caret::confusionMatrix(testing_set[,22],as.factor(y_pred))
#0.8148

#Roc Curve for the logistic regression
library(InformationValue)
plotROC(actuals = testing_set$default.payment.next.month, 
        predictedScores = pro_pred,
        returnSensitivityMat = TRUE)

classifier_logi=train(form= default.payment.next.month~.,
                        data=training_set,
                        method='regLogistic')

#Predicting the test_set results 
y_pred_logi=predict.train(classifier_logi,
                 type='raw',
                 newdata = testing_set[-22])

caret::confusionMatrix(testing_set[,22],as.factor(y_pred_logi))
#0.8143

# save the model to disk
#saveRDS(classifier_logi, "./final_model.rds")

# load the model
#super_model <- readRDS("./final_model.rds")

#-----------------------------------------------------------------------------------

#K-NN Neighbour 
library(class)
y_pred_knn=knn(train=training_set[,-22],
               test=testing_set[,-22],
               cl=training_set[,22],
               k=50)

#Testing Set
caret::confusionMatrix(testing_set[,22],as.factor(y_pred_knn))
#0.8133

classifier_knn=train(form= default.payment.next.month~.,
                     data=training_set,
                     method='knn')

y_pred_knn_1=predict.train(classifier_knn,newdata = testing_set[-22])

caret::confusionMatrix(testing_set[,22],as.factor(y_pred_knn_1))



#-----------------------------------------------------------------------------------

#Svm Model
library(e1071)
classifier_svm=svm(formula=default.payment.next.month ~.,
                   data=training_set,
                   type='C-classification',
                   kernel='radial')
#Training Set
y_pred_svm_train=predict(classifier_svm,newdata=training_set[,-22])

caret::confusionMatrix(training_set[,22],as.factor(y_pred_svm_train))

#Testing Set
y_pred_svm=predict(classifier_svm,newdata=testing_set[,-22])

cm=caret::confusionMatrix(testing_set[,22],as.factor(y_pred_svm))
#0.8183

library(caret)
folds = createFolds(training_set$default.payment.next.month, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = default.payment.next.month ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-22])
  cm = table(test_fold[, 22], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})

accuracy=mean(as.numeric(cv))
#0.8183334

#Grid Search for svm model

classifier_svm_hy=train(form= default.payment.next.month~.,
                        data=training_set,
                        method='svmRadial')

#Grid search with 10- fold cross validation
set.seed(123)
ctrl=trainControl(method = "cv",number=10)
fit=train(x=training_set[,-22],
          y=as.factor(training_set[,22]),
          method="svmRadial",
          trControl=ctrl)

y_pred_2=predict.train(fit,newdata = testing_set[,-22])

caret::confusionMatrix(as.factor(testing_set[,22]),as.factor(y_pred_2))
#0.8163
#-----------------------------------------------------------------------------------

#Random Forest Model
library(randomForest)
classifier_rf=randomForest(default.payment.next.month ~.,
                 data=training_set)

y_pred_rf=predict(classifier_rf,newdata=testing_set[,-22])

caret::confusionMatrix(testing_set[,22],as.factor(y_pred_rf))
#0.812

plot(classifier_rf)

#K-Fold Cross validatation
library(caret)
folds = createFolds(training_set$default.payment.next.month, k = 10)
cv_rf = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  
  classifier_rf=randomForest(default.payment.next.month ~.,
                             data=training_set)
  
  y_pred = predict(classifier_rf, newdata = test_fold[-22])
  cm = table(test_fold[, 22], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
sum(as.numeric(cv_rf))

accuracy_rf=mean(as.numeric(cv_rf))
# 0.9890001


#-----------------------------------------------------------------------------------
#Naive Bayes 
library(e1071)
classifier_naive=naiveBayes(x=training_set[-22],
                      y=training_set$default.payment.next.month)

y_pred_naive=predict(classifier_naive,newdata=testing_set[,-22])

caret::confusionMatrix(testing_set[,22],as.factor(y_pred_naive))
#0.6862

#-----------------------------------------------------------------------------------

#Decision Trees 
library(rpart)
classifier_rpart=rpart(formula=default.payment.next.month ~.,
                       data=training_set)

y_pred_rpart=predict(classifier_rpart,newdata=testing_set[,-22],type='class')

caret::confusionMatrix(testing_set[,22],as.factor(y_pred_rpart))
#0.8185

#Plotting the decision tree
plot(classifier_rpart,margin = 0.2)
text(classifier_rpart)

#------------------------------------------------------------------------------------

