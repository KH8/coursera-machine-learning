function test (X1)

X = [1, 29, 1, 1; 1, 32, 1, 1; 1, 45, 2, 1; 1, 47, 2, 1; 1, 45, 3, 1; 1, 56, 3, 1; 1, 72, 4, 2];
Y = [210; 250; 270; 280; 300; 320; 450];
Theta = pinv(X'*X)*X'*Y;
predictedPrice = Theta'*X1;

disp ('X1 matrix: '), disp (X1), disp ('predicted price:'), disp (predictedPrice);

endfunction
