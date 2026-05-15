
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: EMG.m
% Input: x - a nxd matrix (nx3 if using RGB)
%        k - the number of clusters
%        epochs - number of iterations (epochs) to run the algorithm for
%        flag - flag to use improved EM to avoid singular covariance matrix
% Output: h - a nxk matrix, the expectation of the hidden variable z given the data set and distribution params
%         m - a kxd matrix, the maximum likelihood estimate of the mean
%         Q - vector of values of the complete data log-likelihood function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h, m, Q] = EMG(x, k, epochs, flag)
    
    % variables
    num_clusters = k; % number of clusters
    eps = 1e-15; % small value that can be used to avoid obtaining 0's
    lambda = 1e-3; % value for improved version of EM
    [num_data, dim] = size(x);
    h = zeros(num_data, num_clusters); % expectation of data point being part of a cluster
    S = zeros(dim, dim, num_clusters); % covariance matrix for each cluster
    b = zeros(num_data,num_clusters); % cluster assignments, only used for intialization of pi and S 
    Q = zeros(epochs*2,1); % vector that can hold complete data log-likelihood after each E and M step
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: Initialise cluster means using k-means
    %       use a maximum of 10 iterations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [idx,m] = kmeans(x,num_clusters,"MaxIter",10); %idx: vector of cluster indices for each observation x; m: a kxd matrix of cluster means (kx3 if using RGB)
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: Determine the b values for all data points
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for xi = 1:num_data
        ci = idx(xi);
        b(xi,ci) = 1;
    end
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: Initialize pi's (mixing coefficients)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pi = zeros(1,num_clusters);
    for ci = 1:num_clusters
        count = sum(b(:,ci));
        pi(ci) = count/num_data;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: Initialize the covariance matrix estimate
    %       further modifications will need to be made when doing 2(d)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ci = 1:num_clusters
        assignments = b(:,ci);
        cx = x(assignments == 1,:);
        S(:,:,ci) = cov(cx);
    end
    
    
    % Main EM loop
    for n=1:epochs
        %%%%%%%%%%%%%%%% 
        % E-step
        %%%%%%%%%%%%%%%%
        fprintf('E-step, epoch #%d\n', n);
        [h] =  E_step(x, h, pi, m, S, k);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % TODO: Store the value of the complete log-likelihood function
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Q(2*n - 1) = getCompleteLog(x, h, pi, m, S, flag);

        %%%%%%%%%%%%%%%%
        % M-step
        %%%%%%%%%%%%%%%%
        fprintf('M-step, epoch #%d\n', n);
        [S, m, pi] = M_step(x, h, S, k, flag);            
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % TODO: Store the value of the complete log-likelihood function
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Q(2*n) = getCompleteLog(x, h, pi, m, S, flag);
    end
end

function ln = getCompleteLog(x, h, pi, m, S, flag)
    num_data= size(x,1);
    num_clusters = size(pi,2);

    p = zeros(num_data,num_clusters);

    for i = 1:num_data
        p(i,:) = mvnpdf(x(i,:), m, S) + eps;
    end
    
    a = log(pi) + log(p);
    b = h .* a;
    ln = sum(b, "all");
end