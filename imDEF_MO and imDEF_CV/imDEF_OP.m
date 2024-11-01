function [acc,se,p,sp,g,f,fpr,auc,auc_pr]...
    = imDEF_OP(Train,Label,para,TestData,TestLabel,q,b,alpha1,alpha2,beta)
% imDEF with outputing performance values
%---------------------------------------------parameter setting
T = para.n;
Learner = para.Learner;
classes = unique(Label);class_n = numel(classes);
%------------------------------------------------generating classifier pool
model = cell(1,T);
modelLabel = cell(1,T);
disp('Generating classifier pool')
[candidateDirection,class_distribution]  = OREMpre(Train,Label,q);  % OREM_G consists of  OREMpre and OREMgen
powers = linspace(1,-1,floor(sqrt(T)));
count = 0;
for t=1:floor(sqrt(T))
    if t <= T-floor(sqrt(T))*(floor(T/floor(sqrt(T))))
        nt = floor(T/floor(sqrt(T)))+1;
    else
        nt = floor(T/floor(sqrt(T)));
    end
    
    [Data_t,Label_t] = OREMgen(Train,Label,candidateDirection,class_distribution,powers(t));
    [model(count+1:count+nt),modelLabel(count+1:count+nt)] = SPSEce(Data_t,Label_t,classes,nt,alpha1,b,Learner);
    disp (['SPSEce #' int2str(count+1) ' : ' int2str(count+nt)]);
    count = count + nt;
end
disp('Finished!')
disp('Classifying the train data and test data by the component classifiers')
preTrL = zeros(size(Train,1),T);  
preTrM = zeros(size(Train,1),T*class_n);
preTeL = zeros(size(TestData,1),T);  
preTeM = zeros(size(TestData,1),T*class_n);
for j = 1:T                          
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
valid_ind = find(hc<=T*0.95);
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
for j = 1:T
    P  = P + preTeM(:,(j-1)*class_n+1:j*class_n).*competentLevel(:,j);
end
for i = 1:size(P,1)
    [~,ind]=max(P(i,:));
    Y(i) = classes(ind);
end
P = P./sum(P,2);
disp('Finished!')

[acc,se,p,sp,g,f,fpr,auc,auc_pr] = getPerformance(Y,P,TestLabel,classes);
end