class ChristmasTreeDecorationDiv2:
    def solve(self, col, x, y):
        ans = 0
        n = len(x)
        for i in range(n):
            if col[x[i]-1] != col[y[i]-1]:
                ans += 1

        return ans

if __name__ == "__main__" :
    pass

