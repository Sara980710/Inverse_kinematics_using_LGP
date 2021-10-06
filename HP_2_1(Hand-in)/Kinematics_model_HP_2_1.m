clear, clc, close all

% Length constants
L = [0.055; 0.315; 0.045; 0.108; 0.005; 0.034; 0.015; 0.088; 0.204];

% Test 1: min of all angles
manualAnswer = [L(6)+L(7); -L(4)+L(5); L(2)+L(3)-L(8)-L(9)];
checkData([0,0,0], L, manualAnswer)

% Test 2: max of theta1
manualAnswer = [-(L(6)+L(7)); L(4)-L(5); L(2)+L(3)-L(8)-L(9)];
checkData([pi,0,0], L, manualAnswer)

% Test 3: max of theta2
manualAnswer = [L(6)-L(7); -L(4)+L(5); L(2)+L(3)+L(8)+L(9)];
checkData([0,pi,0], L, manualAnswer)

% Test 4: max of theta3
manualAnswer = [L(6)+L(7)+L(9); -L(4)+L(5); L(2)+L(3)-L(8)];
checkData([0,0,pi/2], L, manualAnswer)

% Test 5: half of theta1
manualAnswer = [L(4)-L(5); L(6)+L(7); L(2)+L(3)-L(8)-L(9)];
checkData([pi/2,0,0], L, manualAnswer)

% Positions for theta values given in home problem 2.1
answer = round(forwKinematicsModel([pi/4,pi/3,pi/6], L),4);
disp(" ")
disp("Coordinates (x,y,z) when theta = [pi/4, pi/3, pi/6]: ")
disp(answer)

% Check data with given parameters and manual answer
function checkData(thetas, L,  manualAnswer)
    answer = round(forwKinematicsModel(thetas, L),4);
    manualAnswer = round(manualAnswer,4);
    
    if (isequal(answer, manualAnswer))
        disp("Correct!")
    else
        disp("NOT CORRECT!")
    end
end


    