function [preTrainL,preTrainM,model,modelLabel,preTestL,preTestM]=runLearner_RM(Learner,trainData,trainLabel,predict_way,testData)
if isequal(Learner,'libsvm')    % libsvm should be installed in advance
   bestcv = -inf;
   for log2c = -5:3:14,
       for log2g = -16:3:8
           cmd = ['-v 5 -c ', num2str(2^log2c),' -g ',num2str(2^log2g),' -t ',num2str(2)];
           cv = svmtrain(trainLabel, sparse(trainData), cmd);
           if (cv > bestcv),
              bestcv = cv; bestc = log2c;bestg = log2g;
           end
       end
   end       %need '-b 1' for training and testing
   tc=bestc;tg=bestg;
   for log2c=[tc-2:tc-1 tc+1:tc+2]
       for log2g = [tg-2:tg-1 tg+1:tg+2]
           cmd = ['-v 5 -c ', num2str(2^log2c),' -g ',num2str(2^log2g),' -t ',num2str(2)];
           cv = svmtrain(trainLabel, sparse(trainData), cmd);
           if (cv > bestcv),
              bestcv = cv; bestc = log2c;bestg = log2g;
           end
       end
   end
   model = svmtrain(trainLabel, sparse(trainData), ['-b 1 -t 2 ','-c ', num2str(2^bestc),' -g ',num2str(2^bestg)]);
   modelLabel = model.Label;  

   if predict_way==1
       [preTrainL,acc,preTrainM]= svmpredict(ones(size(trainData,1),1),sparse(trainData), model, '-b 1');
        preTestL=[];preTestM=[];
   elseif predict_way==2
       [preTestL, acc, preTestM] = svmpredict(ones(size(testData,1),1),sparse(testData), model, '-b 1');
        preTrainL=[];preTrainM=[];
   elseif predict_way==3
       [preTrainL,acc,preTrainM]= svmpredict(ones(size(trainData,1),1),sparse(trainData), model, '-b 1');
       [preTestL, acc, preTestM] = svmpredict(ones(size(testData,1),1),sparse(testData), model, '-b 1');
   else   
       preTrainL=[];preTrainM=[];
       preTestL=[];preTestM=[];
   end
elseif isequal(Learner,'tree')
%     [boostData, boostLabel] = boost_data(trainData, trainLabel,weight./sum(weight),numel(trainLabel));
    list_cate = [];%indices  of categorical predictors
%     classes = unique(boostLabel);
%     for i=1:numel(classes)
%         ClassNames=num2str(classes);
%     end
%     cate_label = cellstr(num2str(boostLabel));
    model = ClassificationTree.fit(trainData,cellstr(num2str(trainLabel)),'CategoricalPredictors',list_cate,'MinParent',2,'MinLeaf',2,'Prune','off');
    modelLabel = [];
    classes = unique(trainLabel);
       for i=1:numel(model.ClassNames)   %obtain the model label
           for j = 1:numel(classes)
               if isequal(model.ClassNames{i},num2str(classes(j)))
                  modelLabel(i)=classes(j);
               end
           end
       end
   if predict_way==1
       [preTrainL,preTrainM]= predict(model,trainData);
       preTrainL = str2double(preTrainL);
       preTestL=[];preTestM=[];
   elseif predict_way==2
       [preTestL,preTestM]= predict(model,testData);
       preTestL = str2double(preTestL);
       preTrainL=[];preTrainM=[];
   elseif predict_way==3
       [preTrainL,preTrainM]= predict(model,trainData);
       preTrainL = str2double(preTrainL);
       [preTestL,preTestM]= predict(model,testData);
       preTestL = str2double(preTestL);
   else   
       preTrainL=[];preTrainM=[];
       preTestL=[];preTestM=[];
   end
elseif isequal(Learner,'lr') %logistic regression
    modelLabel = [0 1];
    model = glmfit(trainData,trainLabel,'binomial','link','logit');
    if predict_way==1
       pro_val = glmval(model,trainData,'logit');
       preTrainL = zeros(size(pro_val)); 
       preTrainL(pro_val>=0.5) = 1;preTrainL(pro_val<0.5) = 0;
       preTrainM = zeros(size(pro_val,1),2);
       preTrainM(pro_val>=0.5,2) = pro_val;preTrainM(pro_val>=0.5,1) = 1- pro_val;
       preTrainM(pro_val<0.5,2) = 1 - pro_val;preTrainM(pro_val<0.5,1) = pro_val;
       preTestL=[];preTestM=[];
   elseif predict_way==2
       pro_val = glmval(model,testData,'logit');
       preTestL = zeros(size(pro_val)); 
       preTestL(pro_val>=0.5) = 1;preTestL(pro_val<0.5) = 0;
       preTestM = zeros(size(pro_val,1),2);
       preTestM(pro_val>=0.5,2) = pro_val;preTestM(pro_val>=0.5,1) = 1- pro_val;
       preTestM(pro_val<0.5,2) = 1 - pro_val;preTestM(pro_val<0.5,1) = pro_val;
       preTrainL=[];preTrainM=[];
   elseif predict_way==3
       pro_val = glmval(model,trainData,'logit');
       preTrainL = zeros(size(pro_val)); 
       preTrainL(pro_val>=0.5) = 1;preTrainL(pro_val<0.5) = 0;
       preTrainM = zeros(size(pro_val,1),2);
       preTrainM(pro_val>=0.5,2) = pro_val;preTrainM(pro_val>=0.5,1) = 1- pro_val;
       preTrainM(pro_val<0.5,2) = 1 - pro_val;preTrainM(pro_val<0.5,1) = pro_val;
       
       pro_val = glmval(model,testData,'logit');
       preTestL = zeros(size(pro_val)); 
       preTestL(pro_val>=0.5) = 1;preTestL(pro_val<0.5) = 0;
       preTestM = zeros(size(pro_val,1),2);
       preTestM(pro_val>=0.5,2) = pro_val;preTestM(pro_val>=0.5,1) = 1- pro_val;
       preTestM(pro_val<0.5,2) = 1 - pro_val;preTestM(pro_val<0.5,1) = pro_val;
   else   
       preTrainL=[];preTrainM=[];
       preTestL=[];preTestM=[];
    end
elseif isequal(Learner,'mlp')
    classes = unique(trainLabel);
    class_n = numel(classes);
    modelLabel = classes;
    n = numel(trainLabel);
    hot_label = zeros(class_n,n);
    for i = 1:class_n
        hot_label(i,trainLabel'==classes(i)) = 1;
    end
if class_n==2
    model = ...
    MultiLayerPerceptron('LengthsOfLayers', [size(trainData,2) 10 class_n],...
                         'HiddenActFcn',     'ReLU',...
                         'OutputActFcn',     'logistic',...
                         'UpdateFcn','none');
else
        model = ...
    MultiLayerPerceptron('LengthsOfLayers', [size(trainData,2) 10 class_n class_n],...
                         'HiddenActFcn',     'ReLU',...
                         'OutputActFcn',     'softmax',...
                         'UpdateFcn','none');
end
% Options                     
    Options = ...
    struct('TrainingAlgorithm', 'GD',...
           'NumberOfEpochs',    100,...
           'MinimumMSE',        1e-3,...
           'SizeOfBatches',     32,...
           'SplitRatio',        1,...
           'LearningRate', 0.1);
     % Training       UpdateFcn none
    model.train(trainData',hot_label,Options);
   if predict_way==1
       model.propagate(trainData');
       preTrainM = (model.Outputs)';
       [~, ind] = max(preTrainM,[],2);
       preTrainL = modelLabel(ind);  
       preTestL=[];preTestM=[];
   elseif predict_way==2
       model.propagate(testData');
       preTestM = (model.Outputs)';
       [~, ind] = max(preTestM,[],2);
       preTestL = modelLabel(ind);  
       preTrainL=[];preTrainM=[];
   elseif predict_way==3
       model.propagate(trainData');
       preTrainM = (model.Outputs)';
       [~, ind] = max(preTrainM,[],2);
       preTrainL = modelLabel(ind); 
       model.propagate(testData');
       preTestM = (model.Outputs)';
       [~, ind] = max(preTestM,[],2);
       preTestL = modelLabel(ind);
   else   
       preTrainL=[];preTrainM=[];
       preTestL=[];preTestM=[];
   end

 elseif isequal(Learner,'treeBagger100')
    T = 100;
%    model  = TreeBagger(T,trainData,trainLabel,'OOBPredictorImportance','On');
   model  = TreeBagger(T,trainData,trainLabel); % random forest
   modelLabel = [];
    classes = unique(trainLabel);
       for i=1:numel(model.ClassNames)   %obtain the model label
           for j = 1:numel(classes)
               if isequal(model.ClassNames{i},num2str(classes(j)))
                  modelLabel(i)=classes(j);
               end
           end
       end
   if predict_way==1
       [preTrainL,preTrainM]= predict(model,trainData);
       preTrainL = str2double(preTrainL);
%        preTrainL=str2num(preTrainL);
       preTestL=[];preTestM=[];
   elseif predict_way==2
       [preTestL,preTestM]= predict(model,testData);
       preTestL = str2double(preTestL);
%        preTestL=str2num(preTestL);%change to the numberic label
       preTrainL=[];preTrainM=[];
   elseif predict_way==3
       [preTrainL,preTrainM]= predict(model,trainData);
       preTrainL = str2double(preTrainL);
%        preTrainL=str2num(preTrainL);
       [preTestL,preTestM]= predict(model,testData);
       preTestL = str2double(preTestL);
%        preTestL=str2num(preTestL);%change to the numberic label
   else   
       preTrainL=[];preTrainM=[];
       preTestL=[];preTestM=[];
   end
 
elseif isequal(Learner,'knn')
    modelLabel = unique(trainLabel);
    model=fitcknn(trainData,trainLabel,'NumNeighbors',5);
    if predict_way==1
       [preTrainL, preTrainM] = predict(model,trainData);
%         preTrainL = str2num(char(preTrainL));
       preTestL=[];preTestM=[];
   elseif predict_way==2
       [preTestL, preTestM] = predict(model,testData);
%        preTestL = str2num(char(preTestL));
       preTrainL=[];preTrainM=[];
   elseif predict_way==3
       [preTrainL, preTrainM] = predict(model,trainData);
%        preTrainL = str2num(char(preTrainL));
       [preTestL, preTestM] = predict(model,testData);
%        preTestL = str2num(char(preTestL));
   else   
       preTrainL=[];preTrainM=[];
       preTestL=[];preTestM=[];
    end
else
    error('your input learner does not supported!');
end
end

