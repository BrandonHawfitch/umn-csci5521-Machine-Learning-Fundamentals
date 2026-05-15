% implements Classify, return the predicted class for each row (we'll call each row x) in data
% by computing the posterior probability that x is in class 1 vs. class 2 then 
% compare the posterior probabilities for classification.
function [predictions] = Classify(data, m1, m2, S1, S2, pc1, pc2)
  % TODO: calculate the discriminants log P(x|C) + log P(C) for both classes 
    v1 = mvndis(data,m1,S1,pc1);
    v2 = mvndis(data,m2,S2,pc2);
  
  % TODO: compare the discriminants to classify the data into two classes
    predictions = zeros(length(data),1);
    for i=1:length(data)
        if v1(i) > v2(i)
            predictions(i) = 1;
        else
            predictions(i) = 2;
        end
    end
end % Function end


function v = mvndis(X,mu,Sigma,prior)
    N = size(X,1);
    Dimensions = size(X,2);
    v = zeros(N,1);
    for i = 1:N
        x = X(i,:);
        a = (-1/2)*Dimensions*log(2*pi);
        b = (-1/2)*(log(det(Sigma)));
        dif = x-mu;
        c = (-1/2) * dif *inv(Sigma) * transpose(dif);
        v(i) = a + b + c + log(prior);
    end
end