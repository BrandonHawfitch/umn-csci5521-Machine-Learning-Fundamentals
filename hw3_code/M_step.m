%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: M_step.m
% Input: x - a nxd matrix (nx3 if using RGB)
%        Q - vector of values from the complete data log-likelihood function
%        h - a nxk matrix, the expectation of the hidden variable z given the data set and distribution params
%        S - cluster covariance matrices
%        k - the number of clusters
%        flag - flag to use improved EM to avoid singular covariance matrix
% Output: S - cluster covariance matrices
%         m - cluster means
%         pi - mixing coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [S, m, pi] = M_step(x, h, S, k, flag)
     
    % get size of data
    [num_data, dim] = size(x);
    eps = 1e-15;
    lambda = 1e-3; % value for improved version of EM
    ni = sum(h,1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: update mixing coefficients
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pi = ni / num_data;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: update cluster means
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a = 1./ni;
    b = h' * x;
    m = a' .* b;
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: Calculate the covariance matrix estimate 
    %       further modifications will need to be made when doing 2(d)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ci = 1:k
        c = x - m(ci,:);
        d = h(:,ci);
        e = c .* d;
        f = e' * c;

        S(:,:,ci) = a(ci) .* f;

        if flag
            d = eye(dim);
            d = d * eps;
            S(:,:,ci) = S(:,:,ci) + d;
        end        
    end

    
end