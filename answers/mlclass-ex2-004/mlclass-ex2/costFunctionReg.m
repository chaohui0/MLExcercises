function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%以下是很挫的写法,效率超低，速度超慢，代码超丑
for i=1:m
	J -= y(i)*log(sigmoid(X(i,:)*theta))+(1-y(i))*log(1-sigmoid(X(i,:)*theta));%'
end
J+=-lambda/2*(theta'*theta);
J/=m;


for i=1:m
    grad+=(sigmoid(X(i,:)*theta)-y(i)).*X(i,:)';
end

grad.+=lambda.*theta;
grad(1)-=lambda*theta(1);
grad./=m;

%以下是牛逼的写法

%h=sigmoid(X*theta);
%J=(-log(h.')*y-log(ones(1,m)-h.')*(ones(m,1)-y))/m+(lambda/(2*m))*sum(theta(2:end).^2);

%grad(1)=(X(:,1).'*(h-y))/m;
%grad(2:end)=(X(:,2:end).'*(h-y))/m+(lambda/m)*theta(2:end);


% =============================================================

end
