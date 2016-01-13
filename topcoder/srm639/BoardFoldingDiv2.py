# Takes too much time

class BoardFoldingDiv2:
    def __init__(self):
        self.already = []

    def canFoldByRow(self, i, paper):
        col = len(paper[0])
        row = len(paper)
        if i <= (row / 2):
            start, mid, end = 0, i, i*2
            vs, ve = mid, row
        else:
            start, mid, end = i*2 - row, i, row
            vs, ve = 0, mid
        for j in range(0, col):
            for r in range(start, mid):
                cr = end - (r - start) - 1
                if paper[r][j] != paper[cr][j]:
                    return 0, vs, ve
        return 1, vs, ve

    def canFoldByCol(self, j, paper):
        col = len(paper[0])
        row = len(paper)
        if j <= (col / 2):
            start, mid, end = 0, j, j*2
            vs, ve = mid, col
        else:
            start, mid, end = j*2 - col, j, col
            vs, ve = 0, mid
        for i in range(0, row):
            for c in range(start, mid):
                cc = end - (c - start) - 1
                if paper[i][c] != paper[i][cc]:
                    return 0, vs, ve
        return 1, vs, ve

    def howMany_r(self, paper, x, y):
        stat = '{},{},{}'.format(paper, x, y)
        if stat in self.already:
            return
        #print("{}".format(stat))
        col = len(paper[0])
        row = len(paper)
        for i in range(1, row):
            ok, vs, ve = self.canFoldByRow(i, paper)
            if ok == 1:
                paper2 = paper[vs:ve]
                self.howMany_r(paper2, x, y+vs)
                if (ve - vs) == vs:
                    paper2 = paper[0:vs]
                    self.howMany_r(paper2, x, y+0)

        for j in range(1, col):
            ok, vs, ve = self.canFoldByCol(j, paper)
            if ok == 1:
                paper2 = []
                for i in range(row):
                    paper2.append(paper[i][vs:ve])
                self.howMany_r(paper2, x+vs, y)
                if (ve - vs) == vs:
                    paper2 = []
                    for i in range(row):
                        paper2.append(paper[i][0:vs])
                    self.howMany_r(paper2, x+0, y)

        self.already.append(stat)

    def howMany(self, paper):
        self.howMany_r(list(paper), 0, 0)
        #print('{}'.format(self.already))
        return len(self.already)


if __name__ == "__main__" :
    ob = BoardFoldingDiv2()
    print('{}'.format(
        ob.howMany(
                ("0110",
                 "1001",
                 "1001",
                 "0110")
        )
    ))