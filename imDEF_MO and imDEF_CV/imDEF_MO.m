function [acc,se,p,sp,g,f,fpr,auc,auc_pr] = imDEF_MO(Train,Label,TestData,TestLabel,varargin)
    % Parameters
    para_learn.n = 100; para_learn.Learner = 'tree';
    if nargin<4
       error('Some parameters are not input!')
    end
    if nargin>4
        para_learn.n = varargin{1};
    end
    if nargin>5
        para_learn.Learner = varargin{2};
    end
    Np      = 30; %30
    Nr      = 60; %60
    maxgen  = 10; %15
    W       = 0.4;
    C1      = 2;
    C2      = 2;
    ngrid   = 10;
    maxvel  = 5;
    u_mut   = 0.5;
  
    n_fold = 5;
    cv = cvpartition(Label,'KFold',n_fold,'Stratify',true);
%     q = para_imDEF(1); b = para_imDEF(2); a1 = para_imDEF(3); a2 = para_imDEF(4); beta = para_imDEF(5);  
    
    fun     = @imDEFcv;
    nVar    = 5;
    var_min = [5;15;0.5;0.8;25];
    var_max = [7;25;0.9;1.2;45];
    
    % Initialization
    POS = repmat((var_max-var_min)',Np,1).*rand(Np,nVar) + repmat(var_min',Np,1);
    VEL = zeros(Np,nVar);
%     POS_fit  = fun(POS);
    POS_fit = fun(Train,Label,cv,para_learn,POS);
    if size(POS,1) ~= size(POS_fit,1)
        warning(['The objective function is badly programmed. It is not returning' ...
            'a value for each particle, please check it.']);
    end
    PBEST    = POS;
    PBEST_fit= POS_fit;
    DOMINATED= checkDomination(POS_fit);
    REP.pos  = POS(~DOMINATED,:);
    REP.pos_fit = POS_fit(~DOMINATED,:);
    REP      = updateGrid(REP,ngrid);
    maxvel   = (var_max-var_min).*maxvel./100;
    gen      = 1;
    
    stopCondition = false;
    while ~stopCondition
        
        % Select leader
        h = selectLeader(REP);
        
        % Update speeds and positions
        VEL = W.*VEL + C1*rand(Np,nVar).*(PBEST-POS) ...
                     + C2*rand(Np,nVar).*(repmat(REP.pos(h,:),Np,1)-POS);
        POS = POS + VEL;
        
        % Perform mutation
        POS = mutation(POS,gen,maxgen,Np,var_max,var_min,nVar,u_mut);
        
        % Check boundaries
        [POS,VEL] = checkBoundaries(POS,VEL,maxvel,var_max,var_min);       
        
        % Evaluate the population
        POS_fit = fun(Train,Label,cv,para_learning,POS);
        
        % Update the repository
        REP = updateRepository(REP,POS,POS_fit,ngrid);
        if(size(REP.pos,1)>Nr)
             REP = deleteFromRepository(REP,size(REP.pos,1)-Nr,ngrid);
        end
        
        % Update the best positions found so far for each particle
        pos_best = dominates(POS_fit, PBEST_fit);
        best_pos = ~dominates(PBEST_fit, POS_fit);
        best_pos(rand(Np,1)>=0.5) = 0;
        if(sum(pos_best)>1)
            PBEST_fit(pos_best,:) = POS_fit(pos_best,:);
            PBEST(pos_best,:) = POS(pos_best,:);
        end
        if(sum(best_pos)>1)
            PBEST_fit(best_pos,:) = POS_fit(best_pos,:);
            PBEST(best_pos,:) = POS(best_pos,:);
        end
       
        gen = gen + 1;
        if(gen>maxgen), stopCondition = true; end
    end
%     hold off;
    h = selectLeader(REP);
    Best_soultion = REP.pos(h,:);
    Best_soultion([1 2 5]) = round(Best_soultion([1 2 5]));
%     try
    [acc,se,p,sp,g,f,fpr,auc,auc_pr]...
    = imDEF_OP(Train,Label,para_learning,TestData, TestLabel, Best_soultion(1), Best_soultion(2), Best_soultion(3), Best_soultion(4), Best_soultion(5));
%     catch
%         disp('error!')
%     end
end
% Function that updates the repository given a new population and its
% fitness
function REP = updateRepository(REP,POS,POS_fit,ngrid)
    % Domination between particles
    DOMINATED  = checkDomination(POS_fit);
    REP.pos    = [REP.pos; POS(~DOMINATED,:)];
    REP.pos_fit= [REP.pos_fit; POS_fit(~DOMINATED,:)];
    % Domination between nondominated particles and the last repository
    DOMINATED  = checkDomination(REP.pos_fit);
    REP.pos_fit= REP.pos_fit(~DOMINATED,:);
    REP.pos    = REP.pos(~DOMINATED,:);
    % Updating the grid
    REP        = updateGrid(REP,ngrid);
end
% Function that corrects the positions and velocities of the particles that
% exceed the boundaries
function [POS,VEL] = checkBoundaries(POS,VEL,maxvel,var_max,var_min)
    % Useful matrices
    Np = size(POS,1);
    MAXLIM   = repmat(var_max(:)',Np,1);
    MINLIM   = repmat(var_min(:)',Np,1);
    MAXVEL   = repmat(maxvel(:)',Np,1);
    MINVEL   = repmat(-maxvel(:)',Np,1);
    
    % Correct positions and velocities
    VEL(VEL>MAXVEL) = MAXVEL(VEL>MAXVEL);
    VEL(VEL<MINVEL) = MINVEL(VEL<MINVEL);
    VEL(POS>MAXLIM) = (-1).*VEL(POS>MAXLIM);
    POS(POS>MAXLIM) = MAXLIM(POS>MAXLIM);
    VEL(POS<MINLIM) = (-1).*VEL(POS<MINLIM);
    POS(POS<MINLIM) = MINLIM(POS<MINLIM);
end
% Function for checking the domination between the population. It
% returns a vector that indicates if each particle is dominated (1) or not
function dom_vector = checkDomination(fitness)
    Np = size(fitness,1);
    dom_vector = zeros(Np,1);
    all_perm = nchoosek(1:Np,2);    % Possible permutations
    all_perm = [all_perm; [all_perm(:,2) all_perm(:,1)]];
    
    d = dominates(fitness(all_perm(:,1),:),fitness(all_perm(:,2),:));
    dominated_particles = unique(all_perm(d==1,2));
    dom_vector(dominated_particles) = 1;
end
% Function that returns 1 if x dominates y and 0 otherwise
function d = dominates(x,y)
    d = all(x<=y,2) & any(x<y,2);
end
% Function that updates the hypercube grid, the hypercube where belongs
% each particle and its quality based on the number of particles inside it
function REP = updateGrid(REP,ngrid)
    % Computing the limits of each hypercube
    ndim = size(REP.pos_fit,2);
    REP.hypercube_limits = zeros(ngrid+1,ndim);
    for dim = 1:1:ndim
        REP.hypercube_limits(:,dim) = linspace(min(REP.pos_fit(:,dim)),max(REP.pos_fit(:,dim)),ngrid+1)';
    end
    
    % Computing where belongs each particle
    npar = size(REP.pos_fit,1);
    REP.grid_idx = zeros(npar,1);
    REP.grid_subidx = zeros(npar,ndim);
    for n = 1:1:npar
        idnames = [];
        for d = 1:1:ndim
            REP.grid_subidx(n,d) = find(REP.pos_fit(n,d)<=REP.hypercube_limits(:,d)',1,'first')-1;
            if(REP.grid_subidx(n,d)==0), REP.grid_subidx(n,d) = 1; end
            idnames = [idnames ',' num2str(REP.grid_subidx(n,d))];
        end
        REP.grid_idx(n) = eval(['sub2ind(ngrid.*ones(1,ndim)' idnames ');']);
    end
    
    % Quality based on the number of particles in each hypercube
    REP.quality = zeros(ngrid,2);
    ids = unique(REP.grid_idx);
    for i = 1:length(ids)
        REP.quality(i,1) = ids(i);  % First, the hypercube's identifier
        REP.quality(i,2) = 10/sum(REP.grid_idx==ids(i)); % Next, its quality
    end
end
% Function that selects the leader performing a roulette wheel selection
% based on the quality of each hypercube
function selected = selectLeader(REP)
    % Roulette wheel
    prob    = cumsum(REP.quality(:,2));     % Cumulated probs
    sel_hyp = REP.quality(find(rand(1,1)*max(prob)<=prob,1,'first'),1); % Selected hypercube
    
    % Select the index leader as a random selection inside that hypercube
    idx      = 1:1:length(REP.grid_idx);
    selected = idx(REP.grid_idx==sel_hyp);
    selected = selected(randi(length(selected)));
end
% Function that deletes an excess of particles inside the repository using
% crowding distances
function REP = deleteFromRepository(REP,n_extra,ngrid)
    % Compute the crowding distances
    crowding = zeros(size(REP.pos,1),1);
    for m = 1:1:size(REP.pos_fit,2)
        [m_fit,idx] = sort(REP.pos_fit(:,m),'ascend');
        m_up     = [m_fit(2:end); Inf];
        m_down   = [Inf; m_fit(1:end-1)];
        distance = (m_up-m_down)./(max(m_fit)-min(m_fit));
        [~,idx]  = sort(idx,'ascend');
        crowding = crowding + distance(idx);
    end
    crowding(isnan(crowding)) = Inf;
    
    % Delete the extra particles with the smallest crowding distances
    [~,del_idx] = sort(crowding,'ascend');
    del_idx = del_idx(1:n_extra);
    REP.pos(del_idx,:) = [];
    REP.pos_fit(del_idx,:) = [];
    REP = updateGrid(REP,ngrid); 
end
% Function that performs the mutation of the particles depending on the
% current generation
function POS = mutation(POS,gen,maxgen,Np,var_max,var_min,nVar,u_mut)
    % Sub-divide the swarm in three parts [2]
    fract     = Np/3 - floor(Np/3);
    if(fract<0.5), sub_sizes =[ceil(Np/3) round(Np/3) round(Np/3)];
    else           sub_sizes =[round(Np/3) round(Np/3) floor(Np/3)];
    end
    cum_sizes = cumsum(sub_sizes);
    
    % First part: no mutation
    % Second part: uniform mutation
    nmut = round(u_mut*sub_sizes(2));
    if(nmut>0)
        idx = cum_sizes(1) + randperm(sub_sizes(2),nmut);
        POS(idx,:) = repmat((var_max-var_min)',nmut,1).*rand(nmut,nVar) + repmat(var_min',nmut,1);
    end
    
    % Third part: non-uniform mutation
    per_mut = (1-gen/maxgen)^(5*nVar);     % Percentage of mutation
    nmut    = round(per_mut*sub_sizes(3));
    if(nmut>0)
        idx = cum_sizes(2) + randperm(sub_sizes(3),nmut);
        POS(idx,:) = repmat((var_max-var_min)',nmut,1).*rand(nmut,nVar) + repmat(var_min',nmut,1);
    end
end