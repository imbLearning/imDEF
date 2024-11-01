function [acc,se,p,sp,g,f,fpr,auc,auc_pr]...
    = CVForOptimizingPara(Train,Label,para,TestData,TestLabel)
% cross validation for optimizing parameters
n_fold = 5;
cv = cvpartition(Label,'KFold',n_fold,'Stratify',true);
best_g = 0; best_f = 0; best_auc = 0;
for q=5:2:7
    for b=15:5:25
        for a1=0.5:0.2:0.9
            for a2=0.8:0.2:1.2
                for beta=25:10:45
                    for i = 1:n_fold
                    [~,~,~,~,g,f,~,auc,~]...
    = imDEF_OP(Train(~cv.test(i),:),Label(~cv.test(i)),para,Train(cv.test(i),:),Label(cv.test(i)),q,b,a1,a2,beta);
                    if sum([best_g<g best_f<f best_auc<auc])>=2
                       best_g=g; best_f=f; best_auc=auc;
                       best_q=q;best_b=b;best_a1=a1;best_a2=a2;best_beta=beta;
                    end
                    end
                end
            end
        end
    end
end

[acc,se,p,sp,g,f,fpr,auc,auc_pr]...
    = imDEF_OP(Train,Label,para,TestData,TestLabel,best_q,best_b,best_a1,best_a2,best_beta);
end