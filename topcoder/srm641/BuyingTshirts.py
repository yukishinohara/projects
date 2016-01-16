class BuyingTshirts:
    def meet(self, T, Q, P):
        qs = []
        q_money = 0
        for i, q in enumerate(Q):
            q_money += q
            if q_money >= T:
                q_money -= T
                qs.append(i)
        p_money = 0
        meet_count = 0
        for i, p in enumerate(P):
            p_money += p
            if p_money >= T:
                p_money -= T
                if i in qs:
                    meet_count += 1
        return meet_count
