
public class GreaterGameDiv2 {
	public int calc(int[] snuke, int[] sothe) {
		int N = snuke.length;
		int points = 0;
		for (int i=0; i < N; i++) {
			if (snuke[i] > sothe[i]) {
				points++;
			}
		}
		
		return points;
	}
}
