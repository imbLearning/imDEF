% demo for running imDEF_{CV}
addpath('./imDEF_MO and imDEF_CV');
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
        [acc,se,p,sp,g,f,fpr,auc,auc_pr] = imDEF_CV(trainData,trainLabel,testData,testLabel);
    end
end