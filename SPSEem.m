function [competentLevel] = SPSEem(TrainData,CL,Label,TestData,classes,Learner,alpha2,b,beta)
% ensemble-margin-aware self-paced sampling ensemble
[n,T] = size(CL);
competentLevel = zeros(size(TestData,1),T);
cur_ind = 1:T;
o_ind = (sum(CL(:,cur_ind))./n)>=0.99;
competentLevel(:,cur_ind(o_ind)) = 1;
o_ind = (sum(CL(:,cur_ind))./n)<=0.01;
competentLevel(:,cur_ind(o_ind)) = 0;
o_ind = ((sum(CL(:,cur_ind))./n)<0.99) & ((sum(CL(:,cur_ind))./n)>0.01);
if sum(o_ind)~=0
    tr_ind = cur_ind(o_ind);
    [competentLevel(:,tr_ind)]  = doSPSEem(TrainData,CL(:,tr_ind),Label,TestData,classes,Learner,alpha2,b,beta);
end
end

function [competentLevel] = doSPSEem(TrainData,CL,Label,TestData,classes,Learner,alpha2,b,beta)
class_n = numel(classes);
T = size(CL,2);
model = cell(beta,T);
modelLabel = cell(beta,T);
temp_label = CL;
temp_label(temp_label==1)=-1;
temp_label(temp_label==0)=1;
hardness = [];
hardness(:,1) = sum(temp_label,2);  % initial negative dynamic ensemble margin
sg = 1-(0:beta-1)./(alpha2*(beta-1));  % self-paced factor
oc = unique(Label);
w = zeros(1,numel(oc));                % the class weights
for i = 1:numel(oc)
    w(i) = sum(Label==oc(i));
end
w = max(sqrt(w))./sqrt(w);
sw = zeros(numel(Label),1);
for i = 1:numel(Label)
    sw(i) = w(oc==Label(i));
end

tr_n = ceil(size(TrainData,1));
for r = 1:beta
    H = mean(hardness,2);   % obtain CNEM
    if max(H)~=0 || min(H)~=0
       H = (H-min(H))./(max(H)-min(H));    % normalize negative dynamic ensemble margin
    end
    [B,AHard] = cutToBins(H,b);         % cut the training data into Bins
    pl = (1./(1e-2+AHard)).^sg(r);      % acquire the sampling weights
    pl = pl./nansum(pl);
    predict_pro = [];
    for t = 1:T
    label_t = CL(:,t);
    count = 0;
    while 1
       
       re_ind = [];
%     for j = 1:class_n
       for i = 1:b
           if ~isnan(pl(i))
               if numel(B{i})==1
                  ind_i = repmat(B{i},ceil(pl(i)*tr_n),1);
               else
                  ind_i = randsample(B{i},ceil(pl(i)*tr_n),true,sw(B{i})); % weighted sampling within each Bin
               end
              re_ind = [re_ind; ind_i];
           end
        end
%     end
       if numel(unique(label_t(re_ind)))==class_n
           break;
       else
           count = count + 1;
       end
       if count>100
           error('Too many cycles to obtain a sampling subset!');
       end
    end
    [~,~,model{r,t},modelLabel{r,t},~,preTrM_t]=runLearner_RM(Learner,TrainData(re_ind,:),label_t(re_ind),2,TrainData);  
    mapind = modelLabel2classes(modelLabel{r,t},classes); preTrM_t(:,mapind) = preTrM_t;
    predict_pro(:,t) = preTrM_t(:,classes==1);
    end
    hardness(:,r+1) = sum(predict_pro.*temp_label,2);    % compute negative dynamic ensemble margin w.r.t r-th generational referee collective
    disp([int2str(r) '-th generational referee collective!'])
end
for r = 1:beta+1
   disp (['1-' int2str(r) ' generational referee collectives, avg-CNEM: ' int2str(mean(hardness(:,r))) ', min-CNEM: ' int2str(max(hardness(:,r)))]);  
end

competentLevel = zeros(size(TestData,1),T);
for r = 1:beta
for t = 1:T
    [~,preTestM] = LearnerPredict(Learner,model{r,t},TestData,ones(size(TestData,1),1),modelLabel{r,t});
    competentLevel(:,t) = competentLevel(:,t) + preTestM(:,modelLabel{r,t}==1);
end
end
competentLevel = competentLevel./beta;
end




function [B,AHard] = cutToBins(H,bin_n)
% H is the computed hardness by F
B = cell(bin_n,1);
AHard = zeros(bin_n,1);
% for j = 1:numel(classes)
%     ind_j = CL==classes(j);
%     max_hard = max(H); min_hard = min(H);
%     thre =  (0:bin_n)./bin_n;
    thre = linspace(min(H),max(H),bin_n+1);
    thre(end) = thre(end)+0.01;
    for i = 1:bin_n
        cur_ind = find(H<thre(i+1) & H >= thre(i));
%         if ~isempty(cur_ind)
        B{i} = cur_ind; 
        AHard(i) = sum(H(cur_ind))/numel(cur_ind);
%         end
    end
% end
end