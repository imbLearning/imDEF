function [Y,P] = imDEF_E(Train, Label, TestData, varargin)
% [Y,P] = imDEF(Train, Label, TestData, Learner, n, Tp, b, alpha2, beta)
% output
% Y: the predicted label
% P: the predicted probability 
% input
% n: number of component classifiers
% b: number of bins
% alpha1,alpha2: self-paced adjusting parameters
% beta: number of generations
%--------------------------parameter settings------------------------------
Learner = 'tree'; n =100; Tp=0.8; b=20;alpha2=1.4; beta=10; 
classes = unique(Label);class_n = numel(classes);
if class_n~=2
   error('This efficient variant of imDEF can only deal with two-class imbalanced problems!')
end

if nargin <3
    error('wrong number of paramters! The training data, training label, and test data are required!')
end
if nargin >=4
   Learner = varargin{1};
end
if nargin >=5
   n = varargin{2};
end
if nargin >=6
   Tp = varargin{3};
end
if nargin >=7
   b = varargin{4};
end
if nargin >=8
   alpha2 = varargin{5};
end
if nargin >=9
   beta = varargin{6};
end
%----------------------------------end-------------------------------------    
disp('Generating classifier pool')
[model,modelLabel] = SPE(Train,Label,Learner,n);
disp('Finished!')

disp('Classifying the train data and test data by the component classifiers')
preTrL = zeros(size(Train,1),n);  
preTrM = zeros(size(Train,1),n*class_n);
preTeL = zeros(size(TestData,1),n);  
preTeM = zeros(size(TestData,1),n*class_n);
for j = 1:n                          
    [preTrL(:,j),preTrM_j]=LearnerPredict(Learner,model{j},Train,Label,modelLabel{j});    
    mapind = modelLabel2classes(modelLabel{j},classes); preTrM_j(:,mapind) = preTrM_j;
    preTrM(:,(j-1)*class_n+1:j*class_n) = preTrM_j;
    
    [preTeL(:,j),preTeM_j]=LearnerPredict(Learner,model{j},TestData,ones(size(TestData,1),1),modelLabel{j});
    preTeM_j(:,mapind) = preTeM_j; preTeM(:,(j-1)*class_n+1:j*class_n) = preTeM_j;
end
disp('Finished!')

disp('construting referee system')
CL = (preTrL==Label).*1;
% competentLevel = zeros(size(TestData,1),n);  % the competent levels of the component classifiers for all test samples
hc = sum(CL,2);
valid_ind = find(hc<=n*Tp);
if isempty(valid_ind)
   competentLevel = ones(size(TestData,1),n);
else
CL = CL(valid_ind,:);
Train = Train(valid_ind,:);
Label = Label(valid_ind,:);
[competentLevel]...
    = SPSEem(Train,CL,Label,TestData,[0, 1],Learner,alpha2,b,beta); % call SPSEem
end
disp('Finished!')

disp('Dynamic prediction')
Y = zeros(size(TestData,1),1);  
P = zeros(size(TestData,1),class_n);  
for j = 1:n
    P  = P + preTeM(:,(j-1)*class_n+1:j*class_n).*competentLevel(:,j);
end
for i = 1:size(P,1)
    [~,ind]=max(P(i,:));
    Y(i) = classes(ind);
end
P = P./sum(P,2);
disp('Finished!')
end






   
    
 











   
    
 








