function [newdata,newlabel] = OREMgen(data,label,candidateDirection,class_distribution,powers)
classes = unique(label);
class_n = numel(classes);
class_distribution = (class_distribution).^powers;
class_distribution = class_distribution./sum(class_distribution);
sampleSizeVector = round(size(data,1).*class_distribution);
% oversampleSize=AllocateEffectiveNumber(sampleSizeVector,gen_n);
newdata=[];
newlabel = [];
for i=1:class_n
    if sampleSizeVector(i)>0
       temp_i=doOREM(data,label,classes(i),candidateDirection{i},sampleSizeVector(i));
       newdata = [newdata;temp_i];
       label_i = ones(size(temp_i,1),1).*classes(i);
       newlabel= [newlabel;label_i];
    end
end
end

function SI=doOREM(data,label,curMinLabel,candidateDirection,Ns)
min_ind = find(label==curMinLabel); 
maj_ind = find(label~=curMinLabel);
np = numel(min_ind);
nn = numel(maj_ind);
data_p = data(min_ind,:);
data_n = data(maj_ind,:);
% Np = numel(curMinInd);

if Ns<=0
    SI = [];
    retubeta;
end

os_ind = randsample(1:np,Ns,true);
SI = zeros(Ns,size(data,2)); 
  
% NN_ind = computeNN( data,K,curMinInd,curMinInd);                    %compute KNN for each minroity class               

for i=1:Ns      
    SI(i,:) = FocusOverSamp(data_p(os_ind(i),:),data_p,data_n,candidateDirection{os_ind(i)},'euclidean'); 
end
end

function syn = FocusOverSamp(sample,data_p,data_n,candidateDirection,distance)
         try_n = 1;
         data = [data_p;data_n];
         if isempty(candidateDirection)
             syn = sample;
             retubeta;
         end
         ind = randsample(1:numel(candidateDirection),try_n,true);
         gap = rand(try_n,size(data,2)); 
%          if candidateDirection(ind)>size(data_p,1)
%             gap = gap./2;
%          end
         syn=repmat(sample,try_n,1)+gap.*(data(candidateDirection(ind),:)-repmat(sample,try_n,1));    
end
