function yEstimate = DecodeChromosome(chromosome,  registers) 
    %operators = [+, -, *, /, ]
    nrDataPoints  = size(registers,1);
    yEstimate = zeros(nrDataPoints,3);
    for i = 1:nrDataPoints
        yEstimate(i,:) = ExecuteInstructions(chromosome, registers(i,:));
    end
end

function estimate = ExecuteInstructions(chromosome, registers)
    lengthInstruction = 4;
    maxAngles = [pi, pi, pi/2];
    nrGenes = length(chromosome);
    
    %Evaluate estimation of y
    for i = 1:lengthInstruction:nrGenes-3
        iOperator = chromosome(i)+1;
        iDestinationRegister = chromosome(i+1)+1;
        iOperand1 = chromosome(i+2)+1;
        iOperand2 = chromosome(i+3)+1;
        
        operand1 = registers(iOperand1);
        operand2 = registers(iOperand2);
        if (operand1 > 1000000.0)
            operand1 = 1000000.0;
        elseif (operand1 < -1000000.0)
            operand1 = -1000000.0;
        end
        if (operand2 > 1000000.0)
            operand2 = 1000000.0;
        elseif (operand2 < -1000000.0)
            operand2 = -1000000.0;
        end
            
        %Execute instruction
        if iOperator == 1
            result = operand1 + operand2;
        elseif iOperator == 2
            result = operand1 - operand2;
        elseif iOperator == 3
            result = operand1 * operand2;
        elseif iOperator == 4
            if operand2 == 0
                result = 1000000.0; %Handle division by 0
            else
                result = operand1 / operand2;
            end  
        elseif iOperator == 5
            result = cos(operand1);
        elseif iOperator == 6
            result = sin(operand1);
        elseif iOperator == 7
            result = acos(mod(operand1,2)-1);
        elseif iOperator == 8
            result = asin(mod(operand1,2)-1);
        else
            error('Invalid operator index!')
        end

        registers(iDestinationRegister) = result;
    end
    estimate = [mod(registers(1), maxAngles(1)), ...
                mod(registers(2), maxAngles(2)), ...
                mod(registers(3), maxAngles(3))];       
end