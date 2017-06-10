function err = testParams(start, stop, steps)
disp('Testing C and sigma ');
load('ex6data3.mat');
errDisp = zeros(steps,steps);
  cC = 1;
  cS = 1;
  for C = logspace(start,stop,steps)
    
    for sigma = logspace(start,stop,steps)


      % We set the tolerance and max_passes lower here so that the code will run
      % faster. However, in practice, you will want to run the training to
      % convergence.
      model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
      figure(1); 
      visualizeBoundary(X, y, model);
      for i = 1:size(Xval, 2)
        %this_X = [X1(:, i), X2(:, i)];
        predictions(:, i) = svmPredict(model, Xval);
      end
      
      %cross-validate
      figure(2); 
      visualizeBoundary(Xval, yval, model);
      errDisp(cC,cS) =  mean(mean(double(predictions ~= yval)))
      err = errDisp(cC,cS);
      disp(strcat('error for sigma = ', num2str(sigma), 'and C = ',num2str(C),'is ',num2str(err,5)));%,'/n Paused until press'));
      %pause();
      %errDisp(cC,cS) = err;
      if cS < steps
        cS = cS+1;
      else
        cS = 1;
      end
    end
    cC = cC+1;
  end
  %visualize the errors as a surface`
  figure(3);
  err = errDisp;
  
  surf(logspace(start,stop,steps),logspace(start,stop,steps),errDisp);%errDisp);
  
end