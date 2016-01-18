class LightSwitchingPuzzle:
    def minFlips(self, state):
        state_l = list(state)
        n = len(state_l)
        ans = 0
        for i in range(n):
            if 'Y' not in state_l:
                return ans
            if state_l[i] == 'N':
                continue
            for j in range(i, n, i+1):
                state_l[j] = 'Y' if state_l[j] == 'N' else 'N'
            ans += 1
        return ans

if __name__ == "__main__" :
    na = LightSwitchingPuzzle()
    str = ''
    for i in range(1000):
        str += 'Y' if i%2 == 0 else 'N'
    print('{}'.format(
        na.minFlips(str)
    ))
