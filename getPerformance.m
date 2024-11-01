function [ACC,SE,P,SP,G,F,FPR,auc_ROC,auc_PR] = getPerformance(predict_label,classProbs,test_label,labelOrder)
         [~,~,~,auc_PR] = perfcurve(test_label, classProbs(:,labelOrder==1),1,'xCrit','reca','yCrit','prec');
         [~,~,~,auc_ROC] = perfcurve(test_label, classProbs(:,labelOrder==1),1,'xCrit','FPR','yCrit','reca');       
         CPSI=predict_label == test_label;        
         TP = length(CPSI(CPSI==1 & test_label==1));
         TN = length(CPSI(CPSI==1 & test_label==-1));
         FP = length(CPSI(CPSI==0 & test_label==-1));
         FN = length(CPSI(CPSI==0 & test_label==1));
         CM = [TP FN;FP TN];       
         ACC = (TP+TN)/(TP+TN+FP+FN);
         SE = TP/(TP+FN);
         Recall = SE;
         P = TP/(TP+FP);   %Precision
         SP = TN/(TN+FP);
         G = (SE*SP)^(1/2);
         F = (2*Recall*P/(Recall+P));
         FPR = FP/(FP+TN);
end