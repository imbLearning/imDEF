addpath('./imDEF_E');

data = csvread('CreditCardFraud.csv');
cv = cvpartition(data(:,end),'HoldOut',0.3);
trainData = data(cv.training,1:end-1);
trainLabel = data(cv.training,end);
testData = data(cv.test,1:end-1);
testLabel = data(cv.test,end);
[predictL,predictP] = imDEF_E(trainData, trainLabel, testData);
[ACC,SE,P,SP,G,F,FPR,AUC_ROC,AUC_PR] = getPerformance(predictL,predictP,testLabel,[0 1])


