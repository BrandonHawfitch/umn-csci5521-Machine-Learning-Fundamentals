% implements Param_Est, returns the parameters for each Multivariate Gaussian
% (m1: learned mean of features for class 1, m2: learned mean of features
% for class 2, S1: learned covariance matrix for features of class 1, 
% S2: learned covariance matrix for features of class 2)
function [m1 m2 S1 S2] = Param_Est(training_data, training_labels, part)
    [m,n] = size(training_data);

    class_1 = [];
    class_2 = [];

  % Means are computed the same way for each model type, covariance is
  % calculated differently
    for i=1:length(training_labels)
      label = training_labels(i);
      if label==1
        class_1 = [class_1;training_data(i,:)];
      else
        class_2 = [class_2;training_data(i,:)];
      end
    end
    m1 = mean(class_1);
    m2 = mean(class_2);
    
  % Parameter estimation for 3 different models described in homework
    
    % TODO: compute parameters for model 1
    S1 = samcov(class_1);
    S2 = samcov(class_2);
    if(strcmp(part, '1'))
        return
    end

    if(strcmp(part, '2'))
        % TODO: compute parameters for model 2
        pc1 = sum(training_labels==1)/size(training_labels,1);
        pc2 = 1-pc1;
        S = (pc1*S1) + (pc2*S2);
        S1 = S;
        S2 = S;
    elseif(strcmp(part,'3'))
        % TODO: compute parameters for model 3
        S1 = diag(diag(S1));
        S2 = diag(diag(S2));
    end
end % Function end



function S = samcov(data)
    [m,n] = size(data);
    mu = mean(data);
    S = zeros(n);
    for i=1:length(data)
        x = data(i,:);
        y = transpose(x-mu) * (x-mu);
        S = S + y;
    end
    S = S/m;
end


