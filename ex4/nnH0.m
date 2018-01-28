function h0 = nnH0(Theta1, Theta2, X)

% Setup some useful variables
m = size(X, 1);

a1 = [ones(m, 1) X];
z1 = Theta1 * a1';

a2 = sigmoid(z1);
a2 = [ones(m, 1) a2'];
z2 = Theta2 * a2';

h0 = sigmoid(z2);

end
