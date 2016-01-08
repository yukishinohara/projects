
public class IntervalSum {
	//1D
	public int [] getIintervalSumArray(int []input) {
		int n = input.length;
		int []output = new int[n];
		if (n <= 0) {
			return output;
		}
		output[0] = input[0];
		for (int i=1; i<n; i++) {
			output[i] = output[i-1] + input[i];
		}
		return output;
	}
	public int getIntervalSum(int []sums, int x1, int x2) {
		return sums[x2] - sums[x1];
	}
	
	//2D
	public int [][] getIintervalSumArray2D(int [][]input) {
		int row = input.length;
		if (row <= 0) {
			return null;
		}
		int col = input[0].length;
		int [][]output = new int[row][col];
		if (col <= 0) {
			return output;
		}
		output[0][0] = input[0][0];
		for (int i=1; i<row; i++) {
			output[i][0] = input[i][0];
		}
		for (int i=0; i<row; i++) {
			for (int j=1; j<col; j++) {
				output[i][j] = output[i][j-1] + input[i][j];
			}
			if (i==0) {
				continue;
			}
			for (int j=0; j<col; j++) {
				output[i][j] += output[i-1][j];
			}
		}
		return output;
	}
	public int getIntervalSum2D(int [][]sums, int y1, int y2, int x1, int x2) {
		if (y1 < 0 || x1 < 0 || y2 < 1 || x2 < 1) {
			return 0;
		}
		int sumleft = 0, sumabove = 0, sumleftabove = 0;
		if (y1 > 0) {
			sumabove = sums[y1-1][x2-1];
		}
		if (x1 > 0) {
			sumleft = sums[y2-1][x1-1];
		}
		if (y1 > 0 && x1 > 0) {
			sumleftabove = sums[y1-1][x1-1];
		}
		return sums[y2-1][x2-1] + sumleftabove - sumleft - sumabove;
	}
}
