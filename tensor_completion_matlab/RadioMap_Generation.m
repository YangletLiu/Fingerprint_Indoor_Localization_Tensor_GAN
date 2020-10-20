%% Synthesis a radio map
function [Radio_map]=RadioMap_Generation(m,n,AP_position)
% physical size: m x n
% AP_position: %% a m x n matrix with few ones indicating APs' positions
%              a index set (x,y) for each AP.   
% Three propagation model: friis (free space), tworay (with ground),
% log_normal_shadowing (...)
% Output: a matrix Raido_map for signal strength.

% Model:
% Pr(d) = Pt - P(d0) - 10 n log10(d/d0) - X(\sigma)
% Parameter setting
Pt = 24; % 24 dBm Transimission Power
Pd0 = 35; % dBm Reference Power at d0 = 1m
nc=4.4; % the path loss component
Rc = 30; %m, Communication range
RSS0 = -100; %dBm, default for distance > Rc
% We modelel X(\sigma) (Guassiam Noise) uniform random within [0,16];
Noise = 16; %dBm


% we have to calculate the received power for each position in Radio_map
Radio_map = [];
temp = zeros(m,n);
d=0; X=0;
for no = 1:size(AP_position,1) % AP_num
    for i=1:m
        for j=1:n
            d = sqrt((i - AP_position(no, 1))^2+(j - AP_position(no, 2))^2);
            X = floor(rand*16) + 1;
            temp(i,j) = 24 - Pd0 - 10 * nc * log10(d) - X;
        end
    end
    Radio_map = [Radio_map temp];
end
