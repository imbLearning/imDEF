function [model,modelLabel]  = SPE(Train,Label,Learner,T)
% self-paced ensemble
bin_n = 20;
model = cell(1,T);
modelLabel = cell(1,T);
classes = unique(Label);
class_n = numel(classes);
class_size = zeros(1,class_n);
for i = 1:class_n
    class_size(i)=numel(find(Label==classes(i)));
end
[~,maj]=max(class_size);
majClass = classes(maj);
[~,mino]=min(class_size);
minClass = classes(mino);

POS_DATA = Train(Label==minClass,:);
NEG_DATA = Train(Label==majClass,:);
pos_size = size(POS_DATA,1);
neg_size = size(NEG_DATA,1);
TRAIN = [POS_DATA;NEG_DATA];
clear Train Label

hardness = [];

u_neg_ind = randsample(1:neg_size,pos_size,true);
[~,~,~,modelLabel_i,~,preTrM]=runLearner_RM(Learner,[POS_DATA;NEG_DATA(u_neg_ind,:)],[ones(pos_size,1).*minClass;ones(pos_size,1).*majClass],2,TRAIN);  
hardness = [hardness preTrM(pos_size+1:end,modelLabel_i==minClass)];
thre = (1/bin_n)*(0:bin_n);
for t = 1:T
    F = (1/t)*sum(abs(hardness),2);
    B = cell(1,bin_n); avg_hardness = zeros(1,bin_n);
    for i = 1:bin_n
        cur_i = find(F<thre(i+1) & F>=thre(i));
        B{i} = cur_i; 
        avg_hardness(i) = sum(F(cur_i))/numel(cur_i);
    end
    sp_f = tan(t*pi/(2*bin_n));
    pl = 1./(sp_f+avg_hardness);
    pl = pl./nansum(pl);  
    u_neg_ind = [];
    for i = 1:bin_n
        if ~isnan(pl(i))
            if numel(B{i}) > round(pl(i)*pos_size)
               ind_i = randsample(B{i},round(pl(i)*pos_size),false);
            else
               ind_i = randsample(B{i},round(pl(i)*pos_size),true);
            end
           u_neg_ind = [u_neg_ind; ind_i];
        end
    end
    [~,~,model{t},modelLabel{t},preTrL,preTrM]=runLearner_RM(Learner,[POS_DATA;NEG_DATA(u_neg_ind,:)],[ones(pos_size,1).*minClass;ones(numel(u_neg_ind),1).*majClass],2,TRAIN);  
    hardness = [hardness preTrM(pos_size+1:end,modelLabel{t}==minClass)];
end
end
