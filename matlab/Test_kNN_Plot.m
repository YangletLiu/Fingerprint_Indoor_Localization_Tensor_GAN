%% Test_kNN
% By Yanglet, June 2014
% Reference papers:
%      [1] Bahl P, Padmanabhan V N. RADAR: An in-building RF-based user location and tracking system. INFOCOM 2000. (citation: 6266)

% Aim: Testing the performance of kNN using full radio map
%      the performance of kNN using partial radio map
% Therefore, we verify that with subsampling, we nedd to recovery the
% original radio map to get more accurate localization.

% Procedures:
% 1. we load the map, stack it into 3D matrix for easy using matrix
% operation.
% 2. Generate a subsampled radio map with SampleRate
% 3. kNN on full radio map
% 4. kNN on partial radio map
% 5. output the localization error (vector, cdf map)
% 

%% Parameter setting
%Radio_map = RadioMap_Generation_TMC2011();
%load Radio_map;
%X=Radio_map; % size: m x 250*AP_num, the matrix for each AP is cancatened into a 2D matrix in the RadioMap_Generation_TMC2011

% Our scenario:
% a 250m x 250m area, with walls separating it into rooms of size 20m x 20m
%m = 250; n=250; N=250;
%AP_num = 169;
% we test two algorithms:
%   1. Optimal: full map + kNN
%   1. partial radio map + kNN

% New data: 64 * 256 * 21
load tensor;
X = new_tensor;
[m,n, AP_num] = size(new_tensor);
SampRate = 0.9;
N=1639;

%% Sampling the full radio map 

% for sampling
%X_sample = ones(m,n,AP_num) * (-100); % keep in mind that the gound level
%is (-100)   -104 in real data
index_sample = rand(m,n) < SampRate; % 2D dimensional, mask for indicator 
% Actually, X_sample is uniquely determined by: X(3D) and index_smple
% In the following, we perform kNN on X (calculating the distance between X and a RSS_vector_sample)
%   use index_sample, we can get distance between X_sample and the
%   RSS_vector_sample

% Therefore, we can omit the generation of X_sample



% Tensor
% for matrix operation, we stack them into a 3D matrix: m x n x AP_num
X = zeros(m,n,AP_num)* (-100); % keep in mind that the gound
for no = 1:AP_num      
    temp_X = X(:,(no-1)*n+1:no*n); %Radio_map: m x n*AP_num
    X(:,:,no) = temp_X;
    % Sampling
    %X_sample(:,:,no) = temp_X .* index_sample + (ones(m,n) * (-100)) .* (ones(m,n) - index_sample);
    % the second part is to set the ground to (-100)
end


[error_opt, error_recovered] = kNN_performance(X, X)



%% KNN

% Note that the grid size is 0.3m x 0.3m

cdfplot(error_recovered * 0.3)


