function [preLabel,preM]=LearnerPredict(Learner,model,testData,testLabel,modelLabel)
if contains(Learner, 'libsvm')
   [preLabel, ~, preM] = svmpredict(testLabel,sparse(testData), model, '-b 1');
elseif  contains(Learner,'tree')
    [preLabel,preM]= predict(model,testData); 
    preLabel = str2double(preLabel);
elseif contains(Learner,'mlp')
    model.propagate(testData');
    preM = (model.Outputs)';
    [~, ind] = max(preM,[],2);
    preLabel = modelLabel(ind);   
elseif isequal(Learner, 'lr')
       pro_val = glmval(model,testData,'logit');
       preLabel = zeros(size(pro_val)); 
       preLabel(pro_val>=0.5) = 1;preLabel(pro_val<0.5) = 0;
       preM = zeros(size(pro_val,1),2);
       preM(pro_val>=0.5,2) = pro_val;preM(pro_val>=0.5,1) = 1- pro_val;
       preM(pro_val<0.5,2) = 1 - pro_val;preM(pro_val<0.5,1) = pro_val;
elseif isequal(Learner, 'knn')
     [preLabel, preM] = predict(model,testData);
else
    error('your input learner does not supported!');
end
end