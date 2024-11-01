function [Y,P] = imDEF(Train, Label, TestData, varargin)
% [Y,P] = imDEF(Train, Label, TestData, Learner, n, q, b, alpha1, alpha2, beta)
% output
% Y: the predicted label
% P: the predicted probability 
% input
% n: number of component classifiers
% q: counting parameter
% b: number of bins
% alpha1,alpha2: self-paced adjusting parameters
% beta: number of generations
%--------------------------parameter settings------------------------------
Learner = 'tree'; n =100; q=5; b=20;
classes = unique(Label);class_n = numel(classes);
if class_n==2
    alpha1=1; %(note:1 for two-class imbalance, 0.8 for multiclass imbalance)
else
    alpha1 = 0.8;
end
alpha2=0.7; beta=25; 
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
   q = varargin{3};
end
if nargin >=7
   b = varargin{4};
end
if nargin >=8
   alpha1 = varargin{5};
end
if nargin >=9
   alpha2 = varargin{6};
end
if nargin >=10
   beta = varargin{7};
end
%----------------------------------end-------------------------------------  
%------------------------------------------------generating classifier pool
model = cell(1,n);
modelLabel = cell(1,n);
disp('Generating classifier pool')
[candidateDirection,class_distribution]  = OREMpre(Train,Label,q);  % OREM_G consists of  OREMpre and OREMgen
powers = linspace(1,-1,floor(sqrt(n)));
count = 0;
for t=1:floor(sqrt(n))
    if t <= n-floor(sqrt(n))*(floor(n/floor(sqrt(n))))
        nt = floor(n/floor(sqrt(n)))+1;
    else
        nt = floor(n/floor(sqrt(n)));
    end
    
    [Data_t,Label_t] = OREMgen(Train,Label,candidateDirection,class_distribution,powers(t));
    [model(count+1:count+nt),modelLabel(count+1:count+nt)] = SPSEce(Data_t,Label_t,classes,nt,alpha1,b,Learner);
    disp (['SPSEce #' int2str(count+1) ' : ' int2str(count+nt)]);
    count = count + nt;
end
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
valid_ind = find(hc<=n*0.95);
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






   
    
 











   
    
 








