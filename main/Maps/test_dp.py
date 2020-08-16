def new_estimatePathCost(startx, starty, endx, endy, myMap):
        hstep = 1 if endx > startx else -1
        vstep = 1 if endy > starty else -1

        if startx == endx and starty == endy:
            return 0

        if startx == endx:
            cost = 0
            idx = endy
            while(idx != starty):
                cost += myMap[idx][startx]
                idx = idx - vstep
            return cost

        if starty == endy:
            cost = 0
            idx = endx
            while(idx != startx):
                cost += myMap[starty][idx]
                idx = idx - hstep
            return cost

        # costMatrix = self.state.mapInfo.map[min(starty, endy): max(starty, endy)][min(startx, endx): max(startx, endx)]
        # dpCost = [[0]*abs(startx - endx) for i in range(abs(starty - endy))]
        costMatrix = myMap
        dpCost = [[0]*(21) for i in range(9)]

        i = endx - hstep
        j = endy
        while ((startx < endx and i >= startx) or (startx >= endx and i <= startx)):
            dpCost[j][i] = dpCost[j][i+hstep] + costMatrix[j][i+hstep]
            i = i - hstep

        i = endx
        j = endy - vstep
        while ((starty < endy and j >= starty) or (starty >= endy and j <= starty)):
            dpCost[j][i] = dpCost[j+vstep][i] + costMatrix[j+vstep][i]
            j = j - vstep
        
        j = endy - vstep
        while ((starty < endy and j >= starty) or (starty >= endy and j <= starty)):
            i = endx - hstep
            while ((startx < endx and i >= startx) or (startx >= endx and i <= startx)):
                dpCost[j][i] = min(dpCost[j][i+hstep] + costMatrix[j][i+hstep], dpCost[j+vstep][i] + costMatrix[j+vstep][i])
                i = i - hstep
            j = j - vstep

        return dpCost[starty][startx]
myMap = [[1, 20, 1, 1, 1, 1, 1, 5, 1, 20, 1, 20, 5, 4, 4, 4, 20, 1, 10, 20, 1], [5, 4, 5, 10, 4, 20, 1, 1, 1, 1, 1, 1, 20, 20, 4, 20, 1, 1, 10, 4, 5], [20, 5, 10, 1, 20, 10, 1, 20, 1, 1, 1, 1, 1, 1, 10, 20, 1, 1, 10, 20, 5], [1, 5, 1, 1, 1, 1, 10, 4, 5, 1, 1, 10, 10, 1, 10, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 20, 20, 1, 5, 1, 20, 4, 5, 10, 1, 1, 1, 20, 10, 20, 20, 20], [20, 1, 1, 20, 5, 5, 20, 20, 1, 1, 5, 10, 1, 1, 1, 1, 20, 4, 20, 20, 5], [4, 10, 20, 5, 10, 5, 5, 20, 1, 5, 1, 1, 1, 1, 5, 1, 20, 20, 20, 5, 4], [1, 20, 5, 10, 4, 10, 5, 20, 1, 1, 1, 1, 10, 1, 1, 1, 1, 20, 5, 4, 1], [1, 5, 10, 4, 4, 10, 5, 20, 1, 20, 1, 10, 4, 10, 20, 1, 20, 5, 4, 1, 4]]
print(new_estimatePathCost(0,8, 4,1, myMap))