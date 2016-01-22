class ListeningSongs:
    def listen(self, durations1, durations2, minutes, T):
        remain = minutes*60
        d1 = list(durations1)
        d2 = list(durations2)
        ans = 0
        for t in range(T):
            if len(d1) <= 0 or len(d2) <= 0:
                return -1
            s1, s2 = min(d1), min(d2)
            remain -= (s1 + s2)
            ans += 2
            if remain < 0:
                return -1
            d1.remove(s1)
            d2.remove(s2)
        d1.extend(d2)
        while len(d1) > 0:
            s1 = min(d1)
            remain -= s1
            if remain < 0:
                return ans
            ans += 1
            d1.remove(s1)

        return ans


if __name__ == "__main__" :
    ll = ListeningSongs(
    )
    ans = ll.listen(
            (61, 61, 61),
            (61, 61, 61),
            1000, 3
    )
    print(ans)


