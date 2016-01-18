class ForgetfulAddition:
    def minNumber(self, expression):
        n = len(expression)
        minv = 1000000000
        for i in range(1, n):
            tmp = int(expression[:i]) + int(expression[i:])
            if tmp < minv:
                minv = tmp
        return minv
