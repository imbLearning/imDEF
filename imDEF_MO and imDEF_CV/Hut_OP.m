function [] =Hut_OP(Data,SaveFileName,wekaobject,Evaluate,Dataproce,Inducer) 
% finding optimal paramter for individual dataset
para.T = 100;
para.Learner = 'tree';
runClassifier = str2func(Inducer.type);

label = Data(:,1);

classes = unique(label);classes = classes(~isnan(classes));

option.r = 1;

option.f = Evaluate.f;

option.m = Evaluate.m;
[~,~,~,~,~,~,splitingstr] = regexpi(SaveFileName,'_');
dataname = splitingstr{1};classifier = splitingstr{2};oversamplingMethod=splitingstr{3};
SaveFileName = strcat(dataname,'_',classifier,'_',para.Learner,'_',num2str(para.T));
cur_dir_files = dir; flag=0;
 for j=1:numel(cur_dir_files)
     if isequal(cur_dir_files(j).name,[SaveFileName '.mat'])
        flag=1;
        break;
     end
 end
if flag==1
   return;
end
if numel(classes)==2

Accarray = [];Searray = [];Parray = [];
Sparray = [];Garray = [];Farray = [];FPRarray = [];
auc_ROCarray = [];  auc_PRarray = [];  
CM = cell(Evaluate.r,Evaluate.f);
DATr={};
DATe={};
% p = parpool(10);
% parfor i = 1 : Evaluate.r
      for i = 1 : Evaluate.r
          
          [Itrain, Itest] = SECP(label,option);
          
          for j = 1: 5
             
              indtr =   cell2mat(Itrain(1,j));
              
              indte =   cell2mat(Itest(1,j));
              
              disp(['processing ' dataname ' using ' oversamplingMethod ', running ' num2str(i) ' times and ' num2str(j) ' fold with the classifier ' classifier]);
             
%               [TRAIN, TEST] = standarizeData(Data(indtr,2:end),Data(indte,2:end));        %standarize data
              [TRAIN, TEST, delete_ind] = standarizeData(Data(indtr,2:end),Data(indte,2:end),wekaobject.featuresFlag);
%               [TRAIN, TEST, delete_ind] = standarizeData(Data(indtr,2:end),Data(indte,2:end));
              
              curObject=updateWekaOject(wekaobject,delete_ind);  %对连续性属性标准化（删除无用的属性），然后过采样数据，然后对离散性属性离散化
              
              newData = Dataproce(TRAIN,Data(indtr,1),curObject);
              
              discrnewdata = nominal2C(newData(:,2:end),curObject);
              
              discrTEST = nominal2C(TEST,curObject);
              
              [Accarray(i,j),Searray(i,j),Parray(i,j),Sparray(i,j),Garray(i,j),Farray(i,j),FPRarray(i,j),auc_ROCarray(i,j),auc_PRarray(i,j)]...
    = runClassifier(discrnewdata,full(newData(:,1)),para,discrTEST,full(Data(indte,1)));
              
%               [preTestL,preTestM,labelOrder,DATr{i,j},DATe{i,j}]=runClassifier(discrnewdata,full(newData(:,1)),para,discrTEST,full(Data(indte,1)));
             
%               [Accarray(i,j),Searray(i,j),Parray(i,j),Sparray(i,j),Garray(i,j),Farray(i,j),FPRarray(i,j),auc_ROCarray(i,j),auc_PRarray(i,j)] = TestPerf(preTestL,preTestM,full(data(indte,1)),labelOrder,classes(minClassInd));
              
%               [preTestL,preTestM,labelOrder]=runSVM(Inducer.type,discrnewdata,newData(:,1),discrTEST);
             
%               [Accarray(i,j),Searray(i,j),Parray(i,j),Sparray(i,j),Garray(i,j),Farray(i,j),FPRarray(i,j),auc_ROCarray(i,j),auc_PRarray(i,j),CM{i,j}] = TestPerf(preTestL,preTestM,full(Data(indte,1)),labelOrder);
       
          end  
      end
% delete(p)   
class_n = numel(classes);
over_n = computeNumberOfPatternstoOversample(label);
minClassInd = find(over_n>0);
majClassInd = find(over_n<=0);
classSize=zeros(1,class_n);
for i=1:class_n
    classSize(i)=numel(find(label==classes(i)));
end

 ACC =nanmean(Accarray(:));stdACC = nanstd(Accarray(:));
 
 SE = nanmean(Searray(:));stdSE = nanstd(Searray(:));
 avgMinRecall=SE;stdavgMinRecall=stdSE;
 
 P =  nanmean(Parray(:));  stdP =  nanstd(Parray(:));
 avgMinPrecision=P;stdavgMinPrecision=stdP;
 
 F = nanmean(Farray(:));stdF =  nanstd(Farray(:));
 MF = F;stdMF = stdF;
 
 SP =nanmean(Sparray(:));stdSP =  nanstd(Sparray(:));
avgMajRecall=SP;stdavgMajRecall=stdSP;

 G = nanmean(Garray(:));stdG =  nanstd(Garray(:));
 MG = G;stdMG = stdG;
 
 FPR = nanmean(FPRarray(:));stdFPR =  nanstd(FPRarray(:));
 
 auc_ROC = nanmean(auc_ROCarray(:));stdauc_ROC =  nanstd(auc_ROCarray(:));
 MAUC = auc_ROC;stdMAUC=stdauc_ROC;
  
 auc_PR =nanmean(auc_PRarray(:));stdauc_PR =  nanstd(auc_PRarray(:));
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%perform t检验%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
 SaveFileName = strcat(SaveFileName,'.mat');
 
 save(SaveFileName,'-regexp','array$');
 
     
save(SaveFileName,'MG','MAUC','ACC','SE','P','F','SP','G','FPR','auc_ROC','auc_PR','classSize','-append');
     
else

    class_n = numel(classes);

    over_n = computeNumberOfPatternstoOversample(label);
    minClassInd = find(over_n>0);
    majClassInd = find(over_n<=0);
    Recallarry=nan(Evaluate.r,Evaluate.f,class_n);
    AUCarray=zeros( Evaluate.r,Evaluate.f,class_n*(class_n-1)/2);
    Precisionarry=nan(Evaluate.r,Evaluate.f,class_n);
    DATr={};
    DATe={};
    ACCarry=nan(Evaluate.r,Evaluate.f);
    CM = cell(Evaluate.r,Evaluate.f);
    p =parpool(10);
    parfor i = 1 : Evaluate.r
%        for i = 1 : Evaluate.r 
          
          [Itrain, Itest] = SECP(label,option);
          
          for j = 1:5
             
              indtr =   cell2mat(Itrain(1,j));
              
              indte =   cell2mat(Itest(1,j));
              
              disp(['processing ' dataname ' using ' oversamplingMethod ', running ' num2str(i) ' times and ' num2str(j) ' fold with the classifier ' classifier]);
              
%               [TRAIN, TEST, delete_ind] = standarizeData(Data(indtr,2:end),Data(indte,2:end));
              [TRAIN, TEST, delete_ind] = standarizeData(Data(indtr,2:end),Data(indte,2:end),wekaobject.featuresFlag);
              
              curObject=updateWekaOject(wekaobject,delete_ind);
              
              newData = Dataproce(TRAIN,Data(indtr,1),curObject);
              
              discrnewdata = nominal2C(newData(:,2:end),curObject);
              
              discrTEST = nominal2C(TEST,curObject);
              
              [preTestL,preTestM,DATr{i,j},DATe{i,j}]=runClassifier(discrnewdata,full(newData(:,1)),para,discrTEST,classes,full(Data(indte,1)));
              
%               [preTestL,preTestM,labelOrder]=runSVM(Inducer.type,discrnewdata,newData(:,1),discrTEST);
             
              [ACCarry(i,j),Recallarry(i,j,:),AUCarray(i,j,:),Precisionarry(i,j,:),CM{i,j}] = TestPerf_mc(preTestL,preTestM,full(Data(indte,1)),classes); 
          end  
    end
    delete(p)
    classSize=zeros(1,class_n);
    for i=1:class_n
        classSize(i)=numel(find(label==classes(i)));
    end
    
    for k=1:class_n
        temp_k=Recallarry(:,:,k);
        Recallclasses(k) = nanmean(temp_k(:));   %Recall for sigle class
        stdRecallclasses(k)=nanstd(temp_k(:));   %std Recall for sigle class
        temp_k=Precisionarry(:,:,k);
        Precisionclasses(k) = nanmean(temp_k(:)); %Precision for sigle class
        stdPrecisionclasses(k)=nanstd(temp_k(:)); %std Precision for sigle class
    end
    
     for k=1:(class_n-1)*class_n/2
         temp_k=AUCarray(:,:,k);
         AUCclasses(k)=nanmean(temp_k(:));      %AUC for each pair classes
         stdAUCclasses(k)=nanstd(temp_k(:));
     end
     MGarry=nan(Evaluate.r,Evaluate.f);MFarry=nan(Evaluate.r,Evaluate.f);
     MAUCarry = nan(Evaluate.r,Evaluate.f);avgRecallarry=nan(Evaluate.r,Evaluate.f);%Recall across all classes on each running time and fold
     avgMinRecallarry=nan(Evaluate.r,Evaluate.f);avgMajRecallarry=nan(Evaluate.r,Evaluate.f);%Recall across all minority classes and majority classes on each running time and fold
     avgPrecisionarry=nan(Evaluate.r,Evaluate.f);
     avgMinPrecisionarry=nan(Evaluate.r,Evaluate.f);avgMajPrecisionarry=nan(Evaluate.r,Evaluate.f);
     for i=1:Evaluate.r
         for j=1:Evaluate.f
             ind = find(~isnan(Recallarry(i,j,:)));
             MGarry(i,j)=prod(Recallarry(i,j,ind))^(1/numel(ind));
             MFarry(i,j)=nanmean((2*Recallarry(i,j,:).*Precisionarry(i,j,:))./(Recallarry(i,j,:)+Precisionarry(i,j,:)));
             MAUCarry(i,j)=nanmean(AUCarray(i,j,:));
             avgRecallarry(i,j)=nanmean(Recallarry(i,j,:));avgMinRecallarry(i,j)=nanmean(Recallarry(i,j,minClassInd));
             avgMajRecallarry(i,j)=nanmean(Recallarry(i,j,majClassInd));
             avgPrecisionarry(i,j)=nanmean(Precisionarry(i,j,:));avgMinPrecisionarry(i,j)=nanmean(Precisionarry(i,j,minClassInd));
             avgMajPrecisionarry(i,j)=nanmean(Precisionarry(i,j,majClassInd));
         end
     end
     MG = nanmean(MGarry(:));MF=nanmean(MFarry(:)); MAUC = nanmean(MAUCarry(:));
     stdMG = nanstd(MGarry(:));stdMF=nanstd(MFarry(:)); stdMAUC=nanstd(MAUCarry(:));%std for MG, MF and MAUC
     ACC =  nanmean(ACCarry(:));stdACC = nanstd(ACCarry(:));
     avgRecall=nanmean(avgRecallarry(:));avgMinRecall=nanmean(avgMinRecallarry(:));avgMajRecall=nanmean(avgMajRecallarry(:));
     stdavgRecall=nanstd(avgRecallarry(:));stdavgMinRecall=nanstd(avgMinRecallarry(:));stdavgMajRecall=nanstd(avgMajRecallarry(:));
     avgPrecision=nanmean(avgPrecisionarry(:));avgMinPrecision=nanmean(avgMinPrecisionarry(:));avgMajPrecision=nanmean(avgMajPrecisionarry(:));
     stdavgPrecision=nanstd(avgPrecisionarry(:));stdavgMinPrecision=nanstd(avgMinPrecisionarry(:));stdavgMajPrecision=nanstd(avgMajPrecisionarry(:));

 %---------------------------------for traning set
 TrQstatistics=[];Trgmean=[];Trfmeasure=[];Trauc=[];
 TrminQstatistics=[];TrmajQstatistics=[];Trminaccuracymin=[];Trminaccuracymax=[];
 Trminaccuracymean=[];Trmajaccuracymin=[];Trmajaccuracymax=[];Trmajaccuracymean=[];
if ~isempty(DATr{1,1})
 for i = 1 : Evaluate.r
     for j = 1: Evaluate.f
         TrQstatistics = [TrQstatistics DATr{i,j}.Qstatistics];
         TrminQstatistics = [TrminQstatistics DATr{i,j}.minQstatistics];
         TrmajQstatistics = [TrmajQstatistics DATr{i,j}.majQstatistics];
         
         Trminaccuracymin =[Trminaccuracymin DATr{i,j}.minaccuracymin];
         Trminaccuracymax =[Trminaccuracymax DATr{i,j}.minaccuracymax];
         Trminaccuracymean =[Trminaccuracymean DATr{i,j}.minaccuracymean];
         
         Trmajaccuracymin =[Trmajaccuracymin DATr{i,j}.majaccuracymin];
         Trmajaccuracymax =[Trmajaccuracymax DATr{i,j}.majaccuracymax];
         Trmajaccuracymean =[Trmajaccuracymean DATr{i,j}.majaccuracymean];

         Trgmean = [Trgmean DATr{i,j}.gmean];
         Trfmeasure = [Trfmeasure DATr{i,j}.fmeasure];
         Trauc = [Trauc DATr{i,j}.auc];
     end
 end
TrQstatistics = nanmean(TrQstatistics);
TrminQstatistics = nanmean(TrminQstatistics);
TrmajQstatistics = nanmean(TrmajQstatistics);
Trminaccuracymin = nanmean(Trminaccuracymin);
Trminaccuracymax = nanmean(Trminaccuracymax);
Trminaccuracymean = nanmean(Trminaccuracymean);
Trmajaccuracymin = nanmean(Trmajaccuracymin);
Trmajaccuracymax = nanmean(Trmajaccuracymax);
Trmajaccuracymean = nanmean(Trmajaccuracymean);

Trgmean = nanmean(Trgmean);
Trfmeasure = nanmean(Trfmeasure);
Trauc = nanmean(Trauc);
end
%-----------------------------------for testing set
 TeQstatistics=[];Tegmean=[];Tefmeasure=[];Teauc=[];
 TeminQstatistics=[];TemajQstatistics=[];Teminaccuracymin=[];Teminaccuracymax=[];
 Teminaccuracymean=[];Temajaccuracymin=[];Temajaccuracymax=[];Temajaccuracymean=[];
 if ~isempty(DATe{1,1})
 for i = 1 : Evaluate.r
     for j = 1: Evaluate.f
         TeQstatistics = [TeQstatistics DATe{i,j}.Qstatistics];
         TeminQstatistics = [TeminQstatistics DATe{i,j}.minQstatistics];
         TemajQstatistics = [TemajQstatistics DATe{i,j}.majQstatistics];
         
         Teminaccuracymin =[Teminaccuracymin DATe{i,j}.minaccuracymin];
         Teminaccuracymax =[Teminaccuracymax DATe{i,j}.minaccuracymax];
         Teminaccuracymean =[Teminaccuracymean DATe{i,j}.minaccuracymean];
         
         Temajaccuracymin =[Temajaccuracymin DATe{i,j}.majaccuracymin];
         Temajaccuracymax =[Temajaccuracymax DATe{i,j}.majaccuracymax];
         Temajaccuracymean =[Temajaccuracymean DATe{i,j}.majaccuracymean];

         Tegmean = [Tegmean DATe{i,j}.gmean];
         Tefmeasure = [Tefmeasure DATe{i,j}.fmeasure];
         Teauc = [Teauc DATe{i,j}.auc];
     end
 end
TeQstatistics = nanmean(TeQstatistics);
TeminQstatistics = nanmean(TeminQstatistics);
TemajQstatistics = nanmean(TemajQstatistics);
Teminaccuracymin = nanmean(Teminaccuracymin);
Teminaccuracymax = nanmean(Teminaccuracymax);
Teminaccuracymean = nanmean(Teminaccuracymean);
Temajaccuracymin = nanmean(Temajaccuracymin);
Temajaccuracymax = nanmean(Temajaccuracymax);
Temajaccuracymean = nanmean(Temajaccuracymean);

Tegmean = nanmean(Tegmean);
Tefmeasure = nanmean(Tefmeasure);
Teauc = nanmean(Teauc);
 end
    
    classPair = nchoosek(classes,2);
    minVsminAUC = [];minVsmaxAUC=[];maxVsmaxAUC=[];
    for i=1:size(classPair,1)
        if ~isempty(find(classes(minClassInd)==classPair(i,1))) && ~isempty(find(classes(minClassInd)==classPair(i,2)))
            minVsminAUC=[minVsminAUC AUCclasses(i)];
        elseif ~isempty(find(classes(majClassInd)==classPair(i,1))) && ~isempty(find(classes(majClassInd)==classPair(i,2)))
            maxVsmaxAUC= [maxVsmaxAUC AUCclasses(i)];
        else
            minVsmaxAUC=[minVsmaxAUC AUCclasses(i)];
        end
    end
    
    avgMinVsMinAUC = nanmean(minVsminAUC);avgMaxVsMaxAUC = nanmean(maxVsmaxAUC);avgMinVsMaxAUC = nanmean(minVsmaxAUC);
    
    SaveFileName = strcat(SaveFileName,'.mat');
     
    save(SaveFileName,'MG','MAUC','MF','ACC','CM','classSize','-regexp','arry$','^Tr','^Te','classes$','^avg','^std','Ind$');
    
end
end

function [ACC,SE,P,SP,G,F,FPR,auc_ROC,auc_PR] = TestPerf(predict_label,classProbs,test_label,labelOrder)


         [~,~,~,auc_PR] = perfcurve(test_label, classProbs(:,labelOrder==1),1,'xCrit','reca','yCrit','prec')
         [~,~,~,auc_ROC] = perfcurve(test_label, classProbs(:,labelOrder==1),1,'xCrit','FPR','yCrit','reca')
         
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

function [ACC,Recall,AUC,Precision,CM] = TestPerf_mc(predict_label,classProbs,test_label,class)
%can deal with the situation same classes missed in test set with respect
%to train set
%nota that some classes in training set may not be in the test set.
         
         class_pred = unique(test_label);
         
         classPair_pred = nchoosek(class_pred,2);
         
         class_n = numel(class);
         
         Recall = nan(1,class_n); AUC =nan(1,class_n*(class_n-1)/2);
         Precision = nan(1,class_n); ACC = nan;
         k=0;
         for i=1:class_n-1
             for j=i+1:class_n
                 k = k +1;
                 ind = find(ismember(classPair_pred,[class(i) class(j)],'rows'));
                 if ~isempty(ind)
                     ind_i = find(test_label==class(i));
                     ind_j = find(test_label==class(j));
                     try
%                      AUC(k)=mean(colAUC(classProbs([ind_i;ind_j],[i j]),test_label([ind_i;ind_j]),'plot',false,'ROC')); 
                     [~,~,~,A_ij] = perfcurve(test_label([ind_i;ind_j]),classProbs([ind_i;ind_j],i),class(i),'xCrit','FPR','yCrit','reca');
                     [~,~,~,A_ji] = perfcurve(test_label([ind_i;ind_j]),classProbs([ind_i;ind_j],j),class(j),'xCrit','FPR','yCrit','reca');
                     catch
                     disp('error!');
                     end
               
                     AUC(k)=mean([A_ij A_ji]);
                 end
             end
         end
  
%          colAUC(classProbs,test_label,'ROC');
         CM = zeros(class_n,class_n);
         for i=1:class_n
             ind_ir=find(test_label==class(i));
             Recall(i)=(numel(find(predict_label(ind_ir)==class(i))))/numel(ind_ir);
             ind_ip=find(predict_label==class(i));
             Precision(i)=(numel(find(test_label(ind_ip)==class(i))))/numel(ind_ip);
             
            for j = 1:class_n
                 CM(i,j) = numel(intersect(ind_ir,find(predict_label==class(j))));
             end
         end
         
         CPSI=predict_label == test_label;
         
         ACC = numel(CPSI(CPSI==1))/numel(CPSI);
end


