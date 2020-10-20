%% kNN Aprroach 

% By Yanglet
% Reference papers:
%      [1] Bahl P, Padmanabhan V N. RADAR: An in-building RF-based user location and tracking system. INFOCOM 2000. (citation: 6266)

% If for realistic scenarios, the SNR is lower, than we may add spatial
% filters to improve its accuracy

% For spatial correlated scenarios, one may need a neighborhood for distance
% calculations.


%% Localization using kNN
function [error_opt, error_recovered] = kNN_performance(True_Tensor, Completed_Tensor)
X=True_Tensor;
X_recovered = Completed_Tensor;

% get the dimension
K=5;
m = size(True_Tensor,1);
n = size(True_Tensor,2);
AP_num = size(True_Tensor,3);

num_points = 200; % or other numbers; % num of testing points, which we use to test localization performance

%% Generate num_points points with uniform random positions
% positions (An indicator matrix of same size, indicating where the testing points locate)

rPerm = randperm(m*n);
omega = sort(rPerm(1:num_points)); 
positions = zeros(m,n);
for i = 1:length(omega)
     row = mod(omega(i),m);
     if row==0
        row = m;
     end
        col = (omega(i) - row)/m + 1;
    positions(row,col)=1;
end

% get the measurement at num_points points from the True_Tensor = X
% noise can be added to model measurement noise
% points_RSS
points_RSS = ones(m,n,AP_num) * (-100);  % ground is -100 dB here.
temp_RSS=[];
for no = 1:AP_num
    temp_X = X(:,(no-1)*n+1:no*n);
    % some noise can be added to model measurement noise;
    temp_RSS = temp_X .* positions + (ones(m,n) * (-100)) .* (ones(m,n) - positions);
    % remember the ground is -100.
    points_RSS(:,:,no) = temp_RSS;
end

% here, we use points_RSS to test localization

%error_opt = zeros(1,num_points);
error_opt = zeros(1,num_points);
error_recovered = zeros(1,num_points);

distance_opt = ones(m,n) * 5000; % a large value
distance = ones(m,n) * 5000;

RSS_dif__opt = zeros(m,n,AP_num);
RSS_dif = zeros(m,n,AP_num);

for i=1:num_points
    
    row = mod(omega(i),m);
        if row==0
            row = m;
        end
     col = (omega(i) - row)/m + 1;
     fprintf('Localization: for the %d th point at (%d, %d)\n', i,row,col);
  
    % Calculate the distance matrix
    for ii=2:m-1
        for jj=2:n-1
            RSS_dif_vector_opt =X(ii,jj,:) - points_RSS(row,col,:);
            RSS_dif_vector = X_recovered(ii,jj,:) - points_RSS(row,col,:);
            
            distance_opt(ii,jj) = norm(squeeze(RSS_dif_vector_opt));
            distance(ii,jj) = norm(squeeze(RSS_dif_vector));
        end
    end  
    
    
    index = zeros(1,K);
    for kkk=1:K
        min_distance = min(min(distance_opt));
        temp_index = find(distance_opt == min_distance);
        index(kkk) = temp_index(1);
        distance_opt(temp_index(1))=5000;
    end

    error = zeros(1,length(index));
    for iter = 1:length(index)
        x = mod(index(iter),m);
        if x ==0
            x = m;
        end
        y = (index(iter)- row)/m + 1;
        fprintf('~~~~~opt: %d -th neighbor at (%d, %d)\n', iter,x,y);
        error(i) = norm([x-row,y-col]);

    end
    fprintf('Localization: estimation error is %f \n', mean(error));
    error_opt(i) = mean(error);
       
    
    index_NN =zeros(1,K);
    for kkk=1:K
        min_distance = min(min(distance));
        temp_index = find(distance == min_distance);
        index_NN(kkk)=temp_index(1);
        distance(temp_index(1))=5000;
    end

    error = zeros(1,length(index)); 
    for iter = 1:length(index_NN)
        x = mod(index_NN(iter),m);
        if x ==0
            x = m;
        end
        y = (index_NN(iter)- row)/m+1;
        fprintf('~~~~~Recovered: %d -th neighbor at (%d, %d)\n', iter,x,y);
        error(i) = norm([x-row,y-col]);
      
    end
    fprintf('~~~~~Recovered: estimation error is %f \n\n', mean(error));
    error_recovered(i) = mean(error);
    
end
