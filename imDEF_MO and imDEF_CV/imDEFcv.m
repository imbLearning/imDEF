function POS_fit...
    = imDEFcv(Train,Label,cv,para_learning,POS)
POS(:,1) = round(POS(:,1)); POS(:,2) = round(POS(:,2)); POS(:,5) = round(POS(:,5));
[u_POS,~,u_i]=unique(POS,'rows');
n = size(u_POS,1);
% n_fold = 3;
% cv = cvpartition(Label,'KFold',n_fold,'Stratify',true);
% g=[];f=[];auc=[];
for j = 1:n
    q = u_POS(j,1); b = u_POS(j,2); a1 = u_POS(j,3); a2 = u_POS(j,4); beta = u_POS(j,5);
for i = 1:cv.NumTestSets
    [acc(j,i),se(j,i),p(j,i),sp(j,i),g(j,i),f(j,i),fpr(j,i),auc(j,i),auc_pr(j,i)]...
    = imDEF_OP(Train(~cv.test(i),:),Label(~cv.test(i)),para_learning,Train(cv.test(i),:),Label(cv.test(i)),q,b,a1,a2,beta);
end
end

for i = 1:size(POS,1)
    t_acc(i,:) = acc(u_i(i),:);t_se(i,:) = se(u_i(i),:);t_p(i,:) = p(u_i(i),:);t_sp(i,:) = sp(u_i(i),:);
    t_g(i,:) = g(u_i(i),:);t_f(i,:) = f(u_i(i),:);t_fpr(i,:) = fpr(u_i(i),:);t_auc(i,:) = auc(u_i(i),:);
    t_auc_pr(i,:) = auc_pr(u_i(i),:);
end
acc = mean(t_acc,2); se = mean(t_se,2); p=mean(t_p,2); sp = mean(t_sp,2); g = mean(t_g,2); f = mean(t_f,2); fpr = mean(t_fpr,2); auc = mean(t_auc,2); auc_pr = mean(t_auc_pr,2);

POS_fit = [f g auc];
end

