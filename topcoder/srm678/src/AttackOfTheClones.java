
public class AttackOfTheClones {
	public int count(int[] firstWeek, int[] lastWeek) {
		int N = firstWeek.length;
		
		int [] pos = new int[N];
		for (int i=0; i<N; i++) {
			pos[firstWeek[i] - 1] = i;
		}
		int max = 0;
		for (int j=0; j<N; j++) {
			int dist = pos[lastWeek[j] - 1] - j;
			if (dist > max) {
				max = dist;
			}
		}
		
		return max + 1;
	}

	public static void main(String [] var) {
		int m = 2500;
		int []x = new int[m];
		int []y = new int[m];
		for (int i=0; i<m; i++) {
			x[i] = i+1;
			y[i] = ((i+m-1) % m)+1;
		}
		System.out.println("" + (new AttackOfTheClones()).count(x,y));
	}
}
