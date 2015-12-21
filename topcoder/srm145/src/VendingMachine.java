
public class VendingMachine {
	public static int getExpensiveCol(int [][]price, int current, int M, int N) {
		int maxcol = 0;
		int maxpri = 0;
		for (int j=0; j<N; j++) {
			int sum = 0;
			for (int i=0; i<M; i++) {
				sum += price[i][j];
			}
			if (sum > maxpri) {
				maxcol = j;
				maxpri = sum;
			}
		}
		
		return maxcol;
	}
	
	public static int getMotorRun(int from, int to, int N) {
		int right = (to+N-from) % N;
		int left  = (from+N-to) % N;
		
		System.out.println("" + from + " -> " + to + "  (" + right + ", " + left + ")");
		
		if (right <= left) return right;
		else return left;
	}
	
	public static int motorUse(String[] prices, String[] purchases) {
		String [] row0 = prices[0].split(" ");
		int M = prices.length; // i: num of shelf
		int N = row0.length;   // j: num of column
		int [][] price = new int[M][N];
		int P = purchases.length;  // k: num of purchases
		int [] pRow = new int[P];
		int [] pCol = new int[P];
		int [] pTime = new int [P];
		
		int current = 0;
		int motorrun = 0;
		
		for (int i=0; i<M; i++) {
			String [] rowi = prices[i].split(" ");
			for (int j=0; j<N; j++) {
				price[i][j] = Integer.parseInt(rowi[j]);
			}
		}
		
		for (int k=0; k<P; k++) {
			String [] foo = purchases[k].split("[,:]+");
			pRow[k] = Integer.parseInt(foo[0]);
			pCol[k] = Integer.parseInt(foo[1]);
			pTime[k] = Integer.parseInt(foo[2]);
		}
		
		int next = getExpensiveCol(price, current, M, N);
		motorrun += getMotorRun(current, next, N);
		current = next;
		
		for (int k=0; k<P; k++) {
			if (price[pRow[k]][pCol[k]] == 0) {
				return -1;
			}
			price[pRow[k]][pCol[k]] = 0;
			next = pCol[k];
			motorrun += getMotorRun(current, next, N);
			current = next;
			if (k < P-1 && pTime[k+1]-pTime[k] >= 5) {
				next = getExpensiveCol(price, current, M, N);
				motorrun += getMotorRun(current, next, N);
				current = next;
			}
		}
		
		next = getExpensiveCol(price, current, M, N);
		motorrun += getMotorRun(current, next, N);
		current = next;
		
		return motorrun;
	}
	public static void main(String []var) {
		System.out.println("" + VendingMachine.motorUse(new String[]
		                                                           {"100 100 100"},
		new String[]
		           {"0,0:0", "0,2:5", "0,1:10"}
		));
	}
}
