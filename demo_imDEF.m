path = './datasets';
filename  =  dir(path); 

for d =1: numel(filename)
    if ~isempty (strfind(filename(d).name, '.csv'))
        data = csvread([path '/' filename(d).name]);
        cv = cvpartition(size(data,1),'HoldOut',0.3);
        trainData = data(cv.training,1:end-1);
        trainLabel = data(cv.training,end);
        testData = data(cv.test,1:end-1);
        testLabel = data(cv.test,end);
        [predictL,predictP] = imDEF(trainData, trainLabel, testData);
        [ACC,SE,P,SP,G,F,FPR,AUC_ROC,AUC_PR] = getPerformance(predictL,predictP,testLabel,[0 1])
    end
end

