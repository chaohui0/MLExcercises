function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J=sum((X*theta-y).^2)/(2*m)+sum(theta(2:end).^2)*lambda/(2*m);

%grad(1)=(X*theta-y)'*X(:,1)/m;
temp = X*theta-y;

%sum = sum(temp.*X(:,1))
%pro = temp.'*X(:,1)

%if sum==pro
%	temp
%	X(:,1)
%end

grad(1)=temp'*X(:,1)/m;

for i=2:size(grad)
	grad(i)=(temp'*X(:,i))/m+lambda/m*theta(i);
end

%h = X * theta; 
%J = (X * theta - y).' * (X * theta - y) / (2*m) +(lambda/(2*m)) * sum(theta(2:end).^2); 
%grad = grad(:); 
%grad(1) = (X(:, 1).' * (h - y)) /m;

%grad(2:end) = (X(:, 2:end).' * (h - y)) /m + (lambda/m) * theta(2:end);







% =========================================================================

grad = grad(:);

end
