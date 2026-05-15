% implements myPCA. Principal Component Analysis (PCA) should
% return the coefficients and principal commponents for some data.
function [latent, coeff] = myPCA(data)
    
    % TODO: implement PCA (!hint! this takes at most 3 lines of code)
    covariance = cov(data);
    [vectors,values] = eig(covariance,"vector");
    [latent,ind] = sort(values,'descend');
    coeff = vectors(:,ind);
    
    %[coef,score,laten] = pca(data,"Economy",false,"Rows","all","Algorithm","eig");

end % end function
