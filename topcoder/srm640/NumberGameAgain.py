import math

class NumberGameAgain:
  def solve(self, k, table):
    ttt = []
    for val in table:
      x, ok = val, 1
      while True:
        x = int(x / 2)
        if x in table:
          ok = 0
        if x <= 1:
          break
      if ok == 1:
        ttt.append(val)
    print ttt
    ans = math.pow(2, k) - 2
    x = 1
    for i in range(k):
      start = x
      end = x*2
      for t in ttt:
        if start <= t and t < end:
          ans -= math.pow(2, k-i) - 1
      x *= 2

    return ans


if __name__ == "__main__" :
    na = NumberGameAgain()
    print('{}'.format(
        na.solve(3, (
            2,
            4,
            6
        )
    )))


