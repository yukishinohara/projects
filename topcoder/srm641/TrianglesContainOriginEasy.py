
class TrianglesContainOriginEasy:
    def gai(self, ax, ay, bx, by):
        ex, ey = ax - bx, ay - by
        ax, ay = -ax, -ay
        return ex*ay - ax*ey

    def isTri(self, i, j, k, x, y):
        if self.gai(x[i], y[i], x[j], y[j]) > 0and \
                self.gai(x[j], y[j], x[k], y[k]) > 0 and \
                self.gai(x[k], y[k], x[i], y[i]) > 0:
            return True
        if self.gai(x[i], y[i], x[j], y[j]) < 0and \
                self.gai(x[j], y[j], x[k], y[k]) < 0 and \
                self.gai(x[k], y[k], x[i], y[i]) < 0:
            return True
        return False

    def count(self, x, y):
        n = len(x)
        ans = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if self.isTri(i, j, k, x, y):
                        ans += 1
        return ans


if __name__ == "__main__" :
    na = TrianglesContainOriginEasy()
    print('{}'.format(
        na.count(
                (-1,-2,3,3,2,1),
                (-2,-1,1,2,3,3)
        )
    ))
