import string

class ContestScoreboard:
    def findWinner(self, scores):
        dmax = 1000000000
        nteam = len(scores)
        teamname = []
        score = []
        times = []

        scores_l = list(scores)

        for i in range(nteam):
            strs = scores_l[i].split()
            teamname.append(strs.pop(0))
            nsub = len(strs)
            this_score = []
            this_times = []
            for j in range(nsub):
                [s, t] = strs[j].split('/')
                this_score.append(int(s))
                this_times.append(int(t))
            score.append(this_score)
            times.append(this_times)

        now = 0
        points = [0] * nteam
        ans = [0] * nteam
        while now < dmax:
            next_now = dmax
            for i in range(nteam):
                nsub = len(score[i])
                for j in range(nsub):
                    if times[i][j] == now:
                        points[i] += score[i][j]
                    if times[i][j] > now and times[i][j] < next_now:
                        next_now = times[i][j]
            max_point = max(points)
            min_name = "ZZZZZZZZ"
            winner = 0
            for i in range(nteam):
                if points[i] >= max_point and teamname[i] < min_name:
                    winner = i
                    min_name = teamname[i]
            ans[winner] = 1
            # print('{}: {} - {}'.format(now, points, winner))
            now = next_now

        return ans

if __name__ == "__main__" :
    cc = ContestScoreboard()
    ans = cc.findWinner((
        "ABC 998/999999997", "FDS 999/999999998", "DFS 999/999999999", "ADS 999/999999999",
        "DFT 999/999999999", "AFS 999/999999999", "SAD 998/1 1/1 1/999999999"
    ))
    print(ans)
