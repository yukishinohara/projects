
public class DivisorDigits {
	public int howMany(int number) {
		int [] ans = new int[12];
		for (int i=1; i<10; i++) {
			ans[i] = 0;
		}
		
		for (int tmp = number; tmp > 0; tmp = tmp / 10) {
			int div = tmp % 10;
			if (div != 0 && number % div == 0) {
				ans[div] += 1;
			}
		}
		
		int finalAns = 0;
		for (int i=1; i<10; i++) {
			finalAns += ans[i];
		}
		
		return finalAns;
	}
	
	public static void main(String []hoge) {
		DivisorDigits h = new DivisorDigits();
		int a = h.howMany(661232);
		a = 0;
	}
}
