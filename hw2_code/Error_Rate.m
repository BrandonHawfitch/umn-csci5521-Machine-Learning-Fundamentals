% implements Error_Rate, returns nothing but prints the percentage of
% predicted labels that are incorrrect.
function [] = Error_Rate(predictions, labels)

    % TODO: compute error rate and print it out
    incorrect = 0;
    for i=1:length(predictions)
        if predictions(i) ~= labels(i)
            incorrect = incorrect + 1;
        end
    end
    disp(incorrect/length(predictions));
  
end % Function end