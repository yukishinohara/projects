class ElectronicPetEasy:

    def isDifficult(self, st1, p1, t1, st2, p2, t2):
        pet1 = []
        for t in range(t1):
            pet1.append((st1 + (p1 * t)))
        for t in range(t2):
            now = st2 + (p2 * t)
            if now in pet1:
                return 'Difficult'
        return 'Easy'
