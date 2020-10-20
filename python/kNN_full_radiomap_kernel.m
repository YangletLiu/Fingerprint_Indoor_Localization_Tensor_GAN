function [error] = kNN_full_radiomap(tensor, points_rss, positions)
% tensor is a full radio map
% points_rss is a tensor: the sampled rss at the points  (positions)
% points_indicator is a matrix indicating where the testing points locate
% positions are a num_testing_points x 2  (x,y coordinates)

k = 50; % num of nearest neighbor used.

num_testing_points = length(positions);

m = size(tensor, 1);
n = size(tensor, 2);
num_ap = size(tensor, 3);


error = zeros(1, num_testing_points);
distance = ones(m,n) * 100000;
rss_vector = zeros(1,num_ap);
tensor_point_rss = zeros(m,n,num_ap);
tensor_rss_dif = zeros(m,n,num_ap);

for no_testing_point = 1:num_testing_points
    % its coordinates (row, col)
    % estimated coordinates (x,y)
    row = positions(no_testing_point, 1);
    col = positions(no_testing_point, 2);
    fprintf('Locating the ** %d **th point at (%d, %d)\n', no_testing_point,row,col);
    
    rss_vector = points_rss(row,col,:);
    % Calculate the distance matrix
    % consturcting a tensor for accelerated calculation
    for ii = 1:m
        for jj = 1:n
            tensor_point_rss(ii,jj,:) = rss_vector ;
        end
    end
    
    tensor_rss_dif = tensor - tensor_point_rss;
    distance = sum(tensor_rss_dif .^2, 3);    %sqrt(sum(tensor_rss_dif .^2, 3));
    
    % find out the nearest k neighbors
    index = zeros(1,k);
    for kkk=1:k
        min_distance = min(min(distance));
        temp_index = find(distance == min_distance);
        index(kkk) = temp_index(1);
        distance(temp_index(1))=100000;
    end

    weight = 0;
    x_total = 0;
    y_total = 0;
    weight_total = 0;
    x_est = 0;
    y_est = 0;
    for iter = 1:length(index)
        x = mod(index(iter),m);
        if x ==0
            x = m;
        end
        y = (index(iter)- x)/m + 1;
        fprintf('~~~~~: %d -th neighbor at (%d, %d)\n', iter,x,y);
        weight = 1 / (0.01 + norm([row - x, col - y]))^2;
        x_est = x_est + x;
        y_est = y_est + y;
        x_total = x_total + weight * x;
        y_total = y_total + weight * y;
        weight_total = weight_total + weight;

    end
    error(no_testing_point) = norm([row - x_total/ weight_total, col - y_total/ weight_total]);

    fprintf('error ~~~weighted: estimation error is %f \n\n', error(no_testing_point));
    fprintf('error ~~~mean: estimation error is %f \n\n',  norm([row - x_est/ k, col - y_est/k]));
     
end
