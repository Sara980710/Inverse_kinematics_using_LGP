function newPositions = getPositions(nrDatapoints, L)
    % Boundaries
    sphereCenterUpper = [0, 0, L(2) + L(3)];
    sphereRadiusOuter = sqrt((L(6) + L(8) + L(9))^2 + (L(4)-L(5))^2);
    sphereRadiusInnerUpper = sqrt(sqrt(L(8)^2 + (L(7) + L(9))^2)^2 + (L(4)-L(5))^2);
    sphereCenterLower = [0, 0, L(2) + L(3) - L(8)];
    sphereRadiusInnerLower = sqrt(L(9)^2 + (L(4) - L(5))^2);
    upperLowerLimit = L(2) + L(3) - L(8);

    maxPos = [sphereRadiusOuter, ... % x 
                sphereRadiusOuter, ... % y
                L(2) + L(3) + L(8)+ L(9) ... % z
                ]; % meter
    minPos = [-(L(6) + L(8) + L(9)), ... % x 
                L(6) - L(7) - L(9), ... % y
                L(2) + L(3) - L(8) - L(9) ... % z
                ]; % meter
    
    % Calculate points
    points = minPos + rand(nrDatapoints,3) .*(maxPos - minPos);

    squaredOuter = (sphereRadiusOuter^2);
    squaredInnerUpper = (sphereRadiusInnerUpper^2);
    squaredInnerLower = (sphereRadiusInnerLower^2);
    
    for i=1:nrDatapoints
        squaredPoint1 = sum((points(i,:) - sphereCenterUpper).^2);
        squaredPoint2 = sum((points(i,:) - sphereCenterLower).^2);
        while((squaredPoint1 > squaredOuter) || ...
                ((points(i,3) > upperLowerLimit) && (squaredPoint1 < squaredInnerUpper)) || ...
                ((points(i,3) <= upperLowerLimit) && (squaredPoint2 < squaredInnerLower)))
            points(i,:) = minPos + rand(1,3).*(maxPos - minPos);
            squaredPoint1 = sum((points(i,:) - sphereCenterUpper).^2);
            squaredPoint2 = sum((points(i,:) - sphereCenterLower).^2);
        end
    end
    
    newPositions = points;
        
end