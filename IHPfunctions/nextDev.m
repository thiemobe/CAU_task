function [devRow,devCol] = nextDev(devRow,devCol)
        devCol = devCol + 1;
        if devCol == 65
            devCol = 1;
            devRow = devRow+1;
        end
end