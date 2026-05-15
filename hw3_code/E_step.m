%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: E_step.m
% Input: x - a nxd matrix (nx3 if using RGB)
%        Q - vector of values from the complete data log-likelihood function
%        h - a nxk matrix, the expectation of the hidden variable z given the data set and distribution params
%        pi - vector of mixing coefficients 
%        m - cluster means
%        S - cluster covariance matrices
%        k - the number of clusters
% Output: h - a nxk matrix, the expectation of the hidden variable z given the data set and distribution params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h] = E_step(x, h, pi, m, S, k)

    [num_data, ~] = size(x);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: perform E-step of EM algorithm
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for i = 1:num_data
        x_ = x(i,:);
        row = zeros(k,1);
        for ci = 1:k
            mu = m(ci,:);
            sigma = S(:,:,ci);
            prior = pi(ci);
            dist = mvnpdf(x_,mu,sigma);
            row(ci) = dist * prior;
        end
        row = (row / norm(row, 1))';
        h(i,:) = row;
    end

end