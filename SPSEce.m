function [model,modelLabel] = SPSEce(Train,Label,classes,nt,alpha1,b,Learner)
% classification-error-aware self-paced sampling ensemble
class_n = numel(classes);
model = cell(1,nt);  % the component classifiers
modelLabel = cell(1,nt);
hardness = [];
sr = 1-(0:nt-1)./(alpha1*(nt-1)); % self-paced factor

flag=1;
while flag
     ind_t = randsample(1:size(Train,1),size(Train,1),true); %random sample replacedment
     if numel(unique(Label(ind_t)))==numel(classes)
        flag=0;
        Data_t = Train(ind_t,:);
        Label_t = Label(ind_t);
     end
end
[~,~,~,modelLabel_0,~,preTrM_0]=runLearner_RM(Learner,Data_t,Label_t,2,Train); % train a clasifier
mapind = modelLabel2classes(modelLabel_0,classes); preTrM_0(:,mapind) = preTrM_0;
hardness(:,1) = computeDifficulty(preTrM_0,Label,classes);

for t = 1:nt
    H = (1/t)*sum(abs(hardness),2);
    [B,AHard] = cutToBins(H,b);
    
    pl = (1./(1e-2+AHard)).^sr(t);
    pl = pl./nansum(pl);
    count = 0;
    while 1  % acquire a sample set by randomly selecting the samples from Bins
%     for j = 1:class_n
       re_ind = [];
       
       for i = 1:b
           if ~isnan(pl(i))
               if numel(B{i})==1
                  ind_i = repmat(B{i},round(pl(i)*size(Train,1)),1);
               else
                  ind_i = randsample(B{i},round(pl(i)*size(Train,1)),true);
               end
              re_ind = [re_ind; ind_i];
           end
        end
%     end
       if numel(unique(Label(re_ind)))==class_n
           break;
       else
           count = count + 1;
       end
       if count > 100
           error('Too many cycles to obtain a sampling subset!');
       end
    end
    [~,~,model{t},modelLabel{t},~,preTrM_t]=runLearner_RM(Learner,Train(re_ind,:),Label(re_ind),2,Train);  
    hardness(:,t+1) = computeDifficulty(preTrM_t,Label,classes);
end
end

function hardness = computeDifficulty(P,Label,classes)
        n = size(P,1);hardness = zeros(n,1);
        for i = 1:n
            hardness(i) = sum(P(i,classes~=Label(i)));
        end
end

function [B,AHard] = cutToBins(H,b)
% H is the computed hardness by F
B = cell(b,1);
AHard = zeros(b,1);
    thre = linspace(min(H),max(H),b+1);
    thre(end) = thre(end)+0.01;
    for i = 1:b
        cur_ind = find(H<thre(i+1) & H >= thre(i));
        B{i} = cur_ind; 
        AHard(i) = sum(H(cur_ind))/numel(cur_ind);
    end
end

