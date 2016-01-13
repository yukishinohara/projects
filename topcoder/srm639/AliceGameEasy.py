class AliceGameEasy:
    def findMinimumValue(self, x, y):
        z = x + y
        smv = 0
        mxv = 0
        i = 0
        while True:
            smv += i
            if smv == z:
                mxv = i
                break
            if smv > z:
                return -1
            i += 1

        z = x
        turn = 0
        i = mxv
        while i > 0:
            if i <= z:
                z -= i
                turn += 1
            i -= 1

        if z != 0:
            return -1
        return turn

if __name__ == "__main__" :
    alice = AliceGameEasy()
    print('{}'.format(
        alice.findMinimumValue(
            0,
            2
        )
    ))
