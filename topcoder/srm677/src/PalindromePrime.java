
public class PalindromePrime {
	boolean isPrime(int n) {
		int numDiv = 0;
		for (int i=1; i<=n; i++) {
			if ((n % i) == 0) { numDiv++; }
		}
		
		return (numDiv == 2);
	}
	
	boolean isPali(int m) {
		int n = m;
		int x = 0;
		for (; n > 0; n = n / 10) {
			x *= 10;
			x += (n % 10);
		}
		
		return (x == m);
	}
	
	public int count(int L, int R) {
		int ans = 0;
		for (int i=L; i<=R; i++) {
			if (isPali(i) && isPrime(i)) {
				ans ++;
			}
		}
		
		return ans;
	}
	
	public static void main(String []hoge) {
		PalindromePrime p = new PalindromePrime();
		System.out.println("" + p.isPali(12321));
	}
}
