% implements Param_Est, returns the number of eigenvectors needed to
% explain 90% of the variance in the data, and displays a plot where the
% x-axis is the number of eigenvectors and the y-axis is the percentage of
% variance explained.
function [neigenvectors] = ProportionOfVariance(training_data)

   % stack data
    data = vertcat(training_data);

  % TODO: perform PCA
    [evalues,evectors] = myPCA(data);

  % TODO: compute proportion of variance explained
    explained = zeros(length(evalues),1);
    vsum = 0;
    vtotal = sum(evalues);
    k=1;
    neigenvectors = 0;
    while k <= length(evalues)
    %for k=1:length(evalues)
        vsum = vsum + evalues(k);

        explained(k) = vsum / vtotal;
        if explained(k) > 0.9 && neigenvectors == 0
            neigenvectors = k;
        end

        k = k + 1;
    end

  % TODO: show figure of proportion of variance explained where the x-axis is the number of eigenvectors and the y-axis is the percentage of
  % variance explained
    plot((1:neigenvectors), explained(1:neigenvectors));
 
end % Function end