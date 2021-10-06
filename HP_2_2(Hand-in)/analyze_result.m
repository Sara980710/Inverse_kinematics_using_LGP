clear; clc; close all;

% Settings
folder = "result";
resultNr = "6";
nrDatapoints = 1000;

% Length constants
L = [0.055; 0.315; 0.045; 0.108; 0.005; 0.034; 0.015; 0.088; 0.204];

% Fix registers
chromosome = load(folder +'/' + resultNr + '_chromosome.mat').data;
constant_registers = load(folder + '/' + resultNr + '_original_registers.mat').constant_registers;
variable_registers = load(folder + '/' + resultNr + '_original_registers.mat').variable_registers;

input = getPositions(nrDatapoints, L);

register =  [variable_registers constant_registers];
registers = cat(2, input, repmat(register,nrDatapoints,1));

% Get positions
predAngles = DecodeChromosome(chromosome, registers);
actualPositions = registers(:,1:3);
predPositions = zeros(size(actualPositions));
for i = 1:size(registers,1)
    predPositions(i,:) = forwKinematicsModel(predAngles(i,:), L);
end

scatter3(actualPositions(:,1), actualPositions(:,2), actualPositions(:,3))
hold on
scatter3(predPositions(:,1), predPositions(:,2), predPositions(:,3))
xlabel('X')
ylabel('Y')
zlabel('Z')

% Get error
error = sqrt(sum((actualPositions-predPositions).^2,2));
error(error==0) = 1/1000000;
error = sort(error);
x = 1:length(error);

figure

scatter(x,error)
title("Max error: " + max(error) + ", "+(sum(error<0.1)/length(error)*100) + " percent under 0.1")