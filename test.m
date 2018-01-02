X = [1, 29, 1, 1; 1, 32, 1, 1; 1, 45, 2, 1; 1, 47, 2, 1; 1, 45, 3, 1; 1, 56, 3, 1; 1, 72, 4, 2];
disp ('X matrix: '), disp (X);

Y = [210; 250; 270; 280; 300; 320; 450];
disp ('Y matrix: '), disp (Y);

Theta = pinv(X'*X)*X'*Y;
disp ('Theta matrix: '), disp (Theta);

X1 = [1; 50; 2; 1];
predictedPrice = Theta'*X1;

disp ('X1 matrix: '), disp (X1), disp ('predicted price:'), disp (predictedPrice);
