function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

admit = find(y == 1)
decline = find(y == 0)

plot(X(admit, 1), X(admit, 2), 'k+', 'Color', 'c')
plot(X(decline, 1), X(decline, 2), 'ko', 'Color', 'm')









% =========================================================================



hold off;

end
