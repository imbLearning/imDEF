function [candidateDirection,class_distribution] = OREMpre(data,label,q)
classes = unique(label);
class_n = numel(classes);
sampleSizeVector = [];
candidateDirection = cell(1,class_n);
for i = 1:class_n
    min_ind = find(label==classes(i)); 
    maj_ind = find(label~=classes(i));
    sampleSizeVector(i)=numel(min_ind); 
    np = numel(min_ind);
    nn = numel(maj_ind);
    data_p = data(min_ind,:);
    data_n = data(maj_ind,:);
    [NNI_p,NND_p] = computeDis( data_p,data_p,np-1,'euclidean','true');    % compute the sample distances
    [NNI_n,NND_n] = computeDis(data_p,data_n,nn,'euclidean'); 
    % finding the candidate assistant seeds, i.e., exploring candidate generation region
    [~,radius_ind] = computeGenerationRadius(NND_p,NND_n,NNI_p,NNI_n,q);  
    % finding the reliable assistant seeds, i.e., identifying the clean subregions within candidate generation region
    candidateDirection{i}=candidateGenerationDirection(data_p,data_n,radius_ind,'euclidean');
end
class_distribution = sampleSizeVector./sum(sampleSizeVector);
end

function [ NNInd,NNDis] = computeDis( varargin)
%compute distance and find nearest neighbors
%data_u: the considered data
%data_s: the search data
%in_self: whether the samples in data_u are included in data_s
data_u = varargin{1};
data_s = varargin{2};
K = varargin{3};
distance = varargin{4};
if nargin==5&&isequal(varargin{5},'true')
   n = size(data_u,1);
   [IDX,Xdis]=knnsearch(data_s, data_u, 'K', K+1, 'Distance', distance);
   NNInd=[];
   NNDis=[];
   for i=1:n  %remove the itself index 
       it_index=find(IDX(i,:)==i, 1);
       if ~isempty(it_index)
           IDX(i,it_index)=-1;
       else
           IDX(i,K+1)=-1;
       end
       NNInd(i,:)=IDX(i,IDX(i,:)~=-1);
       NNDis(i,:)=Xdis(i,IDX(i,:)~=-1);
   end
   return;
end
if nargin==4||(nargin==5&&isequal(varargin{5},'false'))
   [NNInd,NNDis]=knnsearch(data_s, data_u, 'K', K, 'Distance', distance);
   return;
end
end



function candidateDirection=candidateGenerationDirection(data_p,data_n,radius_ind,distance)
np = size(data_p,1); 
data = [data_p;data_n];
candidateDirection=cell(np,1);
for i=1:np
    for j=1:numel(radius_ind{i})
        if sum(radius_ind{i}(1:j-1)>np)==0
           candidateDirection{i} = [candidateDirection{i} radius_ind{i}(j)];  
           continue;
        end
        try
        mean_i = mean([data_p(i,:);data(radius_ind{i}(j),:)],1);
        maj_ind = radius_ind{i}(1:j-1)>np;
        catch
            disp('error!')
        end
        thre_dis_ij = pdist2(data_p(i,:), mean_i, distance);
        dis_i=pdist2(data(radius_ind{i}(maj_ind),:), mean_i, distance);
        smaller_dis_ij_ind = find(dis_i-thre_dis_ij<1e-5, 1);
%         min_count = sum(radius_ind{i}(smaller_dis_ij_ind)<=np)+1;
%         maj_count = sum(radius_ind{i}(smaller_dis_ij_ind)>np);
        if isempty(smaller_dis_ij_ind)
           candidateDirection{i} = [candidateDirection{i} radius_ind{i}(j)]; 
        end
    end
end
end
function [radius,radius_ind] = computeGenerationRadius(NND_p,NND_n,NNI_p,NNI_n,q)
np = size(NND_p,1);
radius=zeros(np,1);
% thre_ind = size(NND_p,2);
radius_ind = cell(np,1);
% min_radius_ind = cell(np,1);
for i=1:np
    dis_i = [NND_p(i,:) NND_n(i,:)];
    ind_i = [NNI_p(i,:) NNI_n(i,:)+np];        %在[data_p;data_n]当中的索�?
    [~,sorted_ind]=sort(dis_i);  %排序�?有的距离
    count_break=0;
    radius_ind{i}=ind_i(sorted_ind);
    for j=1:size(sorted_ind,2)
        if sorted_ind(j) > size(NND_p,2)
           count_break = count_break + 1;
        else
           count_break = 0;
        end
        if count_break >= q  
            radius_ind{i}=ind_i(sorted_ind(1:max(j-5,1)));
            break;
        end
    end
end
end