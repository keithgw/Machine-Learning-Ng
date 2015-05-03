function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

params = [0.01 0.03 0.1 0.3 1 3 10 30]'; % candidate values for C and sigma

% initialize error matrices for validation and training sets
error_val = zeros(length(params),length(params));
error_train = zeros(length(params), length(params));
for i = 1:length(params) 
    for j = 1:length(params) % try every combination of C and sigma
      model= svmTrain(X, y, params(i), @(x1, x2) gaussianKernel(x1, x2, params(j))); 
      predictions = svmPredict(model, Xval);
      error_val(i,j) = mean(double(predictions ~= yval)); % calculate error for specific combination
    end
end

[min_error, ind] = min(error_val(:));   % 0.03
[C_opt, sigma_opt] = ind2sub([size(error_val, 1) size(error_val, 2)], ind);
C = params(C_opt)          % 1
sigma = params(sigma_opt)  % 0.10

% =========================================================================

end
