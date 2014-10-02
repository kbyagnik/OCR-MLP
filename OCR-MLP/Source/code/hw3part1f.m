% This function is the primary driver for homework 3 part 1
function hw3part1
close all;
clear all;
clc;
% we will experiment with a simple 2d dataset to visualize the decision
% boundaries learned by a MLP. Our goal is to study the changes to the
% decision boundary and the training error with respect to the following
% parameters
% - increasing overlap between the data points of the different classes
% - increasing the number of training iterations
% - increase the number of hidden layer neurons
% - see the effect of learning rate on the convergence of the network


% centroid for the three classes
c1=[1 1];
c2=[3 1];
c3=[2 3];

% standard deviation for the three classes
% "increase this quantity to increase the overlap between the classes"
% change this quantity to 0.75 when solving 1(f).
sd=0.75;

% number of data points per class
N=100;

rand('seed', 1);

% generate data points for the three classes
x1=randn(N,2)*sd+ones(N,1)*c1 ;
x2=randn(N,2)*sd+ones(N,1)*c2 ;
x3=randn(N,2)*sd+ones(N,1)*c3 ;

% generate the labels for the three classes in the binary notation
y1= repmat([1 0 0],N,1) ;
y2= repmat([0 1 0],N,1) ;
y3= repmat([0 0 1],N,1) ;

% creating the test data points
a1min = min([x1(:,1);x2(:,1);x3(:,1)]);
a1max = max([x1(:,1);x2(:,1);x3(:,1)]);

a2min = min([x1(:,2);x2(:,2);x3(:,2)]);
a2max = max([x1(:,2);x2(:,2);x3(:,2)]);

[a1, a2] = meshgrid(a1min:0.1:a1max, a2min:0.1:a2max);

testX=[a1(:) a2(:)];

% Experimenting with MLP

% number of epochs for training
tempEpo = [1000 5000 10000];

% learning rate
eta = 0.01;
H = 16 ;
tic ;

for knt=1:3
    nEpochs = tempEpo(knt) ;
    
trainerror_to_plot = [] ;
for epoch=1:nEpochs
    
    % train the MLP using the generated sample dataset
    
    [w, v, trainerror] = mlptrain([x1;x2;x3],[y1;y2;y3], H, eta, nEpochs);
    trainerror_to_plot = [trainerror_to_plot ; trainerror ] ;
    fprintf('Weights calculated after --- ') ;
    toc ;
    % plot the train error againt the number of epochs
    %figure; plot(1:nEpochs, trainerror, 'b:', 'LineWidth', 2);
    
    
    ydash = mlptest(testX, w, v) ;
    fprintf('Ydash calculated after --- ') ;
    toc ;
    
    [val, idx] = max(ydash, [], 2) ;
    label = reshape(idx, size(a1));
    
    % ploting the approximate decision boundary
    % -------------------------------------------
    
    figure;
    imagesc([a1min a1max], [a2min a2max], label), hold on,
    set(gca, 'ydir', 'normal'),
    
    % colormap for the classes:
    %class 1 = light red, 2 = light green, 3 = light blue
    cmap = [1 0.8 0.8; 0.9 1 0.9; 0.9 0.9 1];
    colormap(cmap);
    
    % plot the training data
    plot(x1(:,1),x1(:,2),'r.', 'LineWidth', 2),
    plot(x2(:,1),x2(:,2),'g+', 'LineWidth', 2),
    plot(x3(:,1),x3(:,2),'bo', 'LineWidth', 2),
    
    legend('Class 1', 'Class 2', 'Class 3', 'Location', 'NorthOutside', ...
        'Orientation', 'horizontal');
    
    %viewing the decision surface for the three classes
    ydash1 = reshape(ydash(:,1), size(a1));
    ydash2 = reshape(ydash(:,2), size(a1));
    ydash3 = reshape(ydash(:,3), size(a1));
    
    figure;
    surf(a1, a2, ydash1, 'FaceColor', [1 0 0], 'FaceAlpha', 0.5), hold on,...
    surf(a1, a2, ydash2, 'FaceColor', [0 1 0], 'FaceAlpha', 0.5), hold on,...
    surf(a1, a2, ydash3, 'FaceColor', [0 0 1], 'FaceAlpha', 0.5);
    
    fprintf('Graphs plotted after --- ') ;
    toc
    
end

figure;
    plot(1:nEpochs, trainerror_to_plot, ':', 'LineWidth', 2);
    xlim([0 nEpochs]);
    %ylim([0 1.2]);
    frac = strcat('Epochs = ',num2str(nEpochs));
        title(frac);

end




function [w, v, trainerror] = mlptrain(X, Y, H, eta, nEpochs)
    % X - training data of size NxD
    % Y - training labels of size NxK
    % H - the number of hiffe
    % eta - the learning rate
    % nEpochs - the number of training epochs
    % define and initialize the neural network parameters
    
    % number of training data points
    N = size(X,1);
    % number of inputs
    D = size(X,2); % excluding the bias term
    % number of outputs
    K = size(Y,2);

    % weights for the connections between input and hidden layer
    % random values from the interval [-0.3 0.3]
    % w is a Hx(D+1) matrix
    
    w = -0.3+(0.6)*rand(H,(D+1)) ;

    % weights for the connections between input and hidden layer
    % random values from the interval [-0.3 0.3]
    % v is a Kx(H+1) matrix
    v = -0.3+(0.6)*rand(K,(H+1)) ;
    
    % randomize the order in which the input data points are presented to the
    % MLP
    iporder = randperm(N);
    
    %[m n] = size(ydash) ;
    Nyno = size(Y,2) ;
    trainerror = zeros(1,nEpochs) ;
    
    % mlp training through stochastic gradient descent
    for epoch = 1:nEpochs
        
        for n = 1:N
            % the current training point is X(iporder(n), :)
            % forward pass
            % --------------
            
            % input to hidden layer
            % calculate the output of the hidden layer units - z
            % ---------
            
            z = zeros(H,1) ;
            for h=1:H
                z(h,1) = 1 / (1 + exp(- (w(h,:) *  [1, X(iporder(n),:)]') ) );
            end
            
            z = [1 ;z] ;

            %'TO DO'%
            % ---------

            % hidden to output layer
            % calculate the output of the output layer units - ydash
            % ---------
            
            T =  exp(v * z)' ;
            T = T ./ sum(T) ;
            
            %'TO DO'%
            % ---------
            
            % backward pass
            % ---------------
            % update the weights for the connections between hidden and
            % outlayer units
            % ---------

            deltav = zeros(K,H+1) ;
            for i=1:Nyno
                for h=1:H+1
                    %( Y(iporder(n),i) - ydash(1,i) )
                    deltav(i,h) =  (eta .* ( Y(iporder(n),i) - T(1,i) ) .* z(h,1) )  ;
                end
            end 
            v = v + deltav ;
%             for i=1:Nyno
%                 for h=1:H+1
%                         %( Y(iporder(n),i) - ydash(1,i) ) 
%                         v(i,h) =  v(i,h) +  (eta .* ( Y(iporder(n),i) - T(1,i) ) .* z(h,1) )  ;
%                 end
%             end
            
            %'TO DO'%
            % ---------


            % update the weights for the connections between the input and
            % hidden later units
            % ---------  
            deltaw = zeros(H,D+1) ;
            tot = zeros(H,1) ;
            for h=1:H
                tot(h,1) = sum(( Y(iporder(n),:) - T(1,:) ) .* v(:,h+1)' ) ;
                coeff = ( eta .* tot(h,1) .* z(h+1,1) .* (1 - z(h+1,1)) ) ;

                deltaw(h,:) = ( coeff .* [1 , X(iporder(n),:) ] ) ;
            end
            w = w + deltaw ;
            
%             tot = zeros(H,1) ;
%             for h=1:H
%                 tot(h,1) = sum(( Y(iporder(n),:) - T(1,:) ) .* v(:,h+1)' ) ;
%                 coeff = ( eta .* tot(h,1) .* z(h+1,1) .* (1 - z(h+1,1)) ) ;
%                 
%                 w(h,:) = w(h,:) + ( coeff .* [1 , X(iporder(n),:) ] ) ;
%             end
            
            %'TO DO'%
            % ---------
        end
        ydash = mlptest(X, w, v) ;
        % compute the training error
        % ---------

        temp_error = Y .* abs( log(ydash) ) ;
        trainerror(epoch) = sum(temp_error(:)) ;

        %'TO DO'% uncomment the next line after adding the necessary code
        trainerror(epoch) = trainerror(epoch) ./ N;
        % ---------

%        fprintf('training error after epoch %d: %f\n',epoch,trainerror(epoch)) ;
    end
    
    return;
    
function ydash = mlptest(X, w, v)
    % forward pass of the network

    % number of inputs
    N = size(X,1) ;

    % number of outputs
    K = size(v,1) ;
    
    H = size(w,1) ;

    % forward pass to estimate the outputs
    % --------------------------------------
    % input to hidden for all the data points
    % calculate the output of the hidden layer units
    % ---------
    output = zeros(N,K) ;
    z = zeros(H,1) ;
    for n=1:N
        for h=1:H
            z(h,1) = 1 / (1 + exp(- (w(h,:) * horzcat(1, X(n,:) )') ) ) ;
        end
        
        %'TO DO'%
        % ---------% hidden to output for all the data points
        
        % calculate the output of the output layer units
        % ---------
        ydash =  exp((v * vertcat(1 , z) )') ;
        ydash = ydash ./ sum(ydash) ;
        output(n,:)=ydash ;
    end
    
    ydash = output ;
    %'TO DO'%
    % ---------
    
    
return;