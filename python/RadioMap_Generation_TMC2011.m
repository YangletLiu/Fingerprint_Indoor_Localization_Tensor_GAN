%% Generate a radio map
%function [Radio_map]=RadioMap_Generation_TMC2011()
% Radio map generation according to: 
% Paper: Discriminant minimization seach for large-scale RF-based localization systems, TMC 2011. 

% physical size: m x n, Grid: floor(m/stepsize) x floor(n/stepsize)
% AP_position: a (AP_num x 2) matrix (x,y) for each AP.   

% Output: a matrix Raido_map for signal strength (size: m x n*AP_num).

clear all;
%% Parameter setting
m = 250; n=250;  % note that: stepsize = 5m
AP_num = 169;
% AP_position

% sequential style code
%AP_position = zeros(AP_num,2); % specific for AP_num=169
%   for i=1:sqrt(AP_num)
%       for j=1:sqrt(AP_num)
%       % AP_num = 13 * 13
%       ap_no = (i-1)*sqrt(AP_num) + j;
%       AP_position(ap_no,1) = 20*(i-1)+1;
%       AP_position(ap_no,2) = 20*(j-1)+1;
%       end
%   end

% using matrix operation
AP_position =[];
AP_position_x =[];
AP_position_y =[];
  vector_1 = ones(1, sqrt(AP_num));
  vector_2 = 0:20:n;
  for i = 1:sqrt(AP_num)
      AP_position_x = [AP_position_x, vector_1 * 20 * (i-1) + 1];
      AP_position_y = [AP_position_y, vector_2 + 1];
  end
  AP_position = [AP_position_x', AP_position_y'];
  
% Walls are implicitly used in the following
  
% Default simulation parameters
Pt = 24; % 15dBm Transimission Power
Pd0 = 37.3; % dBm Reference Power at d0 = 1m
phy = 4.4; %
s_min = -100; % -96dBm
sigma = 3;
VSP = 0.2;
DOI = 0.01;
WAF = 5; % 3
N_max =4;
delta_lmin = 1; %m
tao = 2;
lambda = 0.01;
epsilon = 10; %m
stepsize = 1;

%% Radio Map
% we calculate RSS for each position in Radio_map
% for each AP, we generate a temp Radio_map
Radio_map = [];
rows = floor(m / stepsize); % should be > 0
cols = floor(n / stepsize); % should be > 0
temp = ones(rows,cols)*(-96);
Noise = 0;
Ki=0;
s_temp = 0;
% The following iterative code is too time-consuming
% Next, we change it to matrix operations
% for no = 1:AP_num
%     % AP specific things: transmission power, 
%     Pt_VSP_bj = Pt * (1 + normrnd(0,VSP));
%     for i=1:rows
%         for j=1:cols
%             % distance from (i,j) * stepsize to no-th AP
%             % mind the stepsize please
%             d = sqrt(((i-1)*stepsize +1  - AP_position(no, 1))^2+((j-1)*stepsize +1  - AP_position(no, 2))^2); 
%             % calculate the angle
%             % Note: special points: reference points at the same location with APs
%             if d==0
%                d=1;
%             end
%             theta = asind((j*stepsize +1 - AP_position(no, 2))/d);
%             theta_integ = floor(theta);
%             if theta ==0 & i < AP_position(no, 1)
%                 theta =180; % (i,j) lies in the west of b_j
%             end
%             
%             PL_d = Pd0 + 10 * phy * log10(d);
%             Ki = Cal_Ki(theta_integ,DOI);
%             PL_DOI_bj = PL_d * Ki;
%             
%             % N_obs
%             N_obs = floor(((i-1)*stepsize +1  - AP_position(no, 1))/20) + floor(((j-1)*stepsize +1  - AP_position(no, 1))/20);
%           
%             PL_WAF_d = min(N_obs, N_max) * WAF;
%             
%             Noise = normrnd(0, sigma);
%             
%             s_temp = Pt_VSP_bj - PL_DOI_bj - PL_WAF_d + Noise;
%             if s_temp >= s_min
%                temp(i,j) = s_temp;
%             end               
%         end
%     end
%     Radio_map = [Radio_map temp];
% end


% initialize matrix variables for matrix opertions

d =[];
d_x =[];
d_y =[];
  vector_x = ones(1, m);
  vector_y = 1:1:m;
  for i = 1:n
      d_x = [d_x; vector_x * i];
      d_y = [d_y; vector_y];
  end

index =[];
theta =[];
Ki=[];
PL_DOI_bj=[];
N_obs=[];
temp=[];
Radio_map=[];
for no = 1:AP_num
    fprintf('Radio Map: for the %d th AP\n', no);
    % AP specific things: transmission power, 
    Pt_VSP_bj = Pt * (1 + normrnd(0,VSP));
    % calculate the distance matrix
    d = sqrt((d_x - AP_position(no,1)).^2 + (d_y - AP_position(no,2)).^2);
    % deal with special points: reference points at the same location with APs
    index = find(d == 0);
    d(index) = 1;
    
    % PL_d
    PL_d = Pd0 + 10 * phy * log10(d);
    
    theta = asind((d_y - AP_position(no, 2))./d);
    % deal with the second section + 90
    index_theta = intersect(find(d_x < AP_position(no,1)), find(d_y > AP_position(no,2)));
    theta(index_theta) =  theta(index_theta) + 90;
    % the neg-x axis + 180
    index_theta = intersect(find(d_x < AP_position(no,1)), find(d_y == AP_position(no,2)));
    theta(index_theta) =  theta(index_theta) + 180;
    % the neg-y axis + 360
    index_theta = intersect(find(d_x == AP_position(no,1)), find(d_y < AP_position(no,2)));
    theta(index_theta) =  theta(index_theta) + 360;
    % deal with the third section 
    index_theta = intersect( find(d_x < AP_position(no,1)), find(d_y < AP_position(no,2)) );
    theta(index_theta) =  theta(index_theta) + 270;
    % deal with the fourth section 
    index_theta = intersect( find(d_x > AP_position(no,1)), find(d_y < AP_position(no,2)) );
    theta(index_theta) =  theta(index_theta) + 360;
    
    theta_integ = floor(theta);
    
    Ki = Cal_Ki_matrix(theta_integ,DOI);
    
    PL_DOI_bj = PL_d .* Ki;   

    % N_obs
    N_obs = round(abs((d_x - AP_position(no,1))/20)) + round(abs((d_y - AP_position(no,2))/20));
    %floor(((i-1)*stepsize +1  - AP_position(no, 1))/20) + floor(((j-1)*stepsize +1  - AP_position(no, 1))/20);
    
    index = find(N_obs > N_max);
    N_obs(index) = N_max;
    PL_WAF_d = N_obs .* WAF;
             
    Noise = normrnd(0, sigma,[rows,cols]);
             
    temp = Pt_VSP_bj - PL_DOI_bj - PL_WAF_d + Noise;
    index = find(temp < s_min);
    temp(index)=s_min;
       
    Radio_map = [Radio_map temp];  
end

 save Radio_map Radio_map;