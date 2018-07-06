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

#Training set
pro_pred_train=predict(classifier,
                       type='response',
                       newdata = training_set[-22])

y_pred_train=ifelse(pro_pred_train>0.5,1,0)

#Confusion Matrix
library(caret)
#training Set
caret::confusionMatrix(training_set[,22],as.factor(y_pred_train))

#testing set
caret::confusionMatrix(testing_set[,22],as.factor(y_pred))
#0.8148

#Roc Curve for the logistic regression
#install.packages('InformationValue')
library(InformationValue)
plotROC(actuals = testing_set$default.payment.next.month, 
        predictedScores = pro_pred,
        returnSensitivityMat = TRUE)

# set.seed(123)
# ctrl=trainControl(method = "cv",number=3)
# fit=train(x=training_set[,-22],
#           y=training_set[,22],
#              method="glm",
#              trcontrol=ctrl,
#              tuneLength=5)


#K-NN Neighbour 
library(class)
y_pred_knn=knn(train=training_set[,-22],
               test=testing_set[,-22],
               cl=training_set[,22],
               k=50)

#testing set
caret::confusionMatrix(testing_set[,22],as.factor(y_pred_knn))
#0.8133

# #Lets find the best k value by using cross validation method
# set.seed(123)
# ctrl=trainControl(method = "cv",number=3)
# knnfit=train(default.payment.next.month ~.,
#              data=training_set,
#              method="knn",
#              trcontrol=ctrl,
#              tuneLength=5)
# plot(knnfit)

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

caret::confusionMatrix(testing_set[,22],as.factor(y_pred_svm))
#0.8183

#Random Forest Model
library(randomForest)
classifier_rf=randomForest(default.payment.next.month ~.,
                 data=training_set)

y_pred_rf=predict(classifier_rf,newdata=testing_set[,-22])

caret::confusionMatrix(testing_set[,22],as.factor(y_pred_rf))
#0.812

plot(classifier_rf)




