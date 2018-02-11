function [C, sigma] = findOptimalParams(X, y, Xval, yval)

cCandidates = [1, 3, 10, 30, 100, 300, 1000, 3000];
sigmaCandidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

eMin = 1000;

for i=1:8
  for j=1:8
    cTest = cCandidates(i);
    sigmaTest = sigmaCandidates(j);

    fprintf('Testing C: %f, sigma: %f\n', cTest, sigmaTest);

    model= svmTrain(X, y, cTest, @(x1, x2) gaussianKernel(x1, x2, sigmaTest));
    predictions = svmPredict(model, Xval);

    e = mean(double(predictions ~= yval));
    fprintf('Error: %f\n', e);

    if e < eMin
      eMin = e;
      C = cTest;
      sigma = sigmaTest;
    endif

    fprintf('Current C: %f, sigma: %f, error: %f\n', C, sigma, eMin);
    fprintf('\n......................\n');
  end
end

end
