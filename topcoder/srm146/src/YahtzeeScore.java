
public class YahtzeeScore {
	public static int maxPoints(int[] toss) {
		int num = toss.length;
		int max = 0;
		
		for (int deme = 1; deme <= 6; deme++) {
			int sum = 0;
			for (int i=0; i<num; i++) {
				if (toss[i] == deme) {
					sum += deme;
				}
			}
			
			if (sum > max) {
				max = sum;
			}
		}
		
		return max;
	}
}
