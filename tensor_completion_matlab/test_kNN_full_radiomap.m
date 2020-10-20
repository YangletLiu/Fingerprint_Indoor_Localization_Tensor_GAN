% To get data in the variablese
clear all;

addpath ./TFOCS-1.3.1/

load result_4;% or result_4 based on which experiment is being performed
true_tensor = M;
tensor_completed_tensor = X;  % recovered later
indicator = Omega(:,:,1);
clear M X Omega;

load tensor_2D_completed_30percent;
completed_tensor = tensor_2D_completed_30percent;
clear tensor_mc_completed_50percent;

m = size(true_tensor, 1);
n = size(true_tensor, 2);
num_ap = size(true_tensor, 3);
num_testing_points = 100;
%% Generate num_points points with uniform random positions
% points_rss is a tensor: the sampled rss at the points  (positions)
% points_indicator is a matrix indicating where the testing points locate
%     for getting points_rss
% positions are a num_testing_points x 2  (x,y coordinates)

rPerm = randperm(m*n);
omega = sort(rPerm(1:num_testing_points)); 
positions = zeros(num_testing_points,2);

points_indicator = zeros(m,n);

for i = 1:num_testing_points
     row = mod(omega(i),m);
     if row==0
        row = m;
     end
     col = (omega(i) - row)/m + 1;
     points_indicator(row,col) = 1;
     positions(i,1) = row;
     positions(i,2) = col;
end

% Get the rss at the testing points
points_rss = zeros(m,n,num_ap);

for no = 1:num_ap
    temp_x = true_tensor(:,:,no);
    % some noise can be added later;
    temp_rss = temp_x .* points_indicator;
    points_rss(:,:,no) = temp_rss;
end

%% generating the partial radio map 
% rss at the anchor points

SampRate = 0.3;
anchor_indicator = zeros(m,n);
nSamples = round(SampRate*m*n);
% Observation
rPerm   = randperm(m*n); % use "randsample" if you have the stats toolbox
omega   = sort( rPerm(1:nSamples));
    

for i = 1:length(omega)
   row = mod(omega(i),m);
   if row==0
      row = m;
   end
   col = (omega(i) - row)/m + 1;
   anchor_indicator(row,col)=1;
end

anchor_indicator_out = ones(m,n) - anchor_indicator;

%% Matrix Completion 
% It is too slow, we have to do it elsewhere, and then save it
% Here, we load it and use it directly.



%% Localization testing
error_opt = zeros(1,num_testing_points);
error_completed = zeros(1,num_testing_points);
error_incomplete = zeros(1,num_testing_points);
%[error_opt] = kNN_full_radiomap(true_tensor, points_rss, positions);
[error_completed] = kNN_full_radiomap(completed_tensor, points_rss, positions);
%[error_tensor_completed] = kNN_full_radiomap(tensor_completed_tensor, points_rss, positions); 
[error_incomplete] = kNN_partial_radiomap(true_tensor, anchor_indicator, points_rss, positions);

 
%% Draw cdf
% Create figure
figure1 = figure;
axes1 = axes('Parent',figure1,'YTick',[0 0.2 0.4 0.6 0.8 1],...
    'XTick',[0 1 2 3 4 5 6 7 8 9 10],...
    'FontSize',14);
xlim(axes1,[0 10]);
box(axes1,'on');
grid(axes1,'on');
hold(axes1,'all');


%cdfplot(error_opt*.1);

h=cdfplot(error_completed*.1);
hold on;
set(h,'color','r');
h=cdfplot(error_tensor_completed*.1);
set(h,'color','b');
h=cdfplot(error_incomplete*.1);
set(h,'color','g');

% Create xlabel
xlabel('Location Error','FontSize',14);
% Create ylabel
ylabel('CDF','FontSize',14);
% Create title
title('Empirical CDF','FontSize',14);
% Create legend
legend1 = legend(axes1,'show');
