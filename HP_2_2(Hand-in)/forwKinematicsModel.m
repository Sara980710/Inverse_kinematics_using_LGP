% Model
function frame0 = forwKinematicsModel(thetas, L)
% Point is always at origin of frame 3
frame3 = [0;0;0;1];

% Matrices
trans3to2 = [cos(thetas(3)), -sin(thetas(3)), 0, L(9)*sin(thetas(3));
        sin(thetas(3)), cos(thetas(3)), 0, -L(9)*cos(thetas(3));
        0,0,1,0;
        0,0,0,1];
    
trans2to1 = [cos(thetas(2)), -sin(thetas(2)), 0, L(7)*cos(thetas(2)) + L(8)*sin(thetas(2));
        sin(thetas(2)), cos(thetas(2)), 0, L(7)*sin(thetas(2)) - L(8)*cos(thetas(2));
        0,0,1,-L(5);
        0,0,0,1];
    
trans1to0 = [cos(thetas(1)), 0, sin(thetas(1)), L(6)*cos(thetas(1)) + L(4)*sin(thetas(1));
        sin(thetas(1)), 0, -cos(thetas(1)), L(6)*sin(thetas(1)) - L(4)*cos(thetas(1));
        0,1,0,L(2)+L(3);
        0,0,0,1];

% Calculation
frame0 = trans1to0*trans2to1*trans3to2*frame3;
frame0 = frame0(1:3);
end
