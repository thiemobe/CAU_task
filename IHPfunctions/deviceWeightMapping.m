function [devIndWr,devRow,devCol] = deviceWeightMapping(HRS2LRS, Nr, Nr2, devRow, devCol)
    devIndWr = zeros(2,Nr,Nr2);
    for i=1:Nr
        for j=1:Nr2
            while HRS2LRS(devRow,devCol) == 0
                [devRow,devCol] = nextDev(devRow,devCol);
            end
            devIndWr(1, i, j) = devRow;
            devIndWr(2, i, j) = devCol;
            [devRow,devCol] = nextDev(devRow,devCol);
        end
    end
end