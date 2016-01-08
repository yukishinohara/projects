import java.util.ArrayList;
import java.util.HashMap;


public class ClosestRabbit {
	HashMap<String, Long> already = new HashMap<String, Long>();
	double [][][][] dist = null;
	
	public double sumGraph(String[] board, int row, int col, int r, int n) {
		for (int i=0; i<row; i++) {
			for (int j=0; j<col; j++) {
				if (board[i].substring(j, j+1).equals("#")) {
					continue;
				}
				for (int i2=0; i2<row; i2++) {
					for (int j2=0; j2<col; j2++) {
						if (board[i2].substring(j2, j2+1).equals("#")) {
							continue;
						}
						dist[i][j][i2][j2] = (double)((i2-i)*(i2-i) + (j2-j)*(j2-j));
					}
				}
			}
		}
		
		double total = 0;
		for (int i=0; i<row; i++) {
			for (int j=0; j<col; j++) {
				if (board[i].substring(j, j+1).equals("#")) {
					continue;
				}
				for (int i2=0; i2<row; i2++) {
					for (int j2=0; j2<col; j2++) {
						if (i2 == i && j2 == j) { continue; }
						if (board[i2].substring(j2, j2+1).equals("#")) {
							continue;
						}
						int ok = 1; int ok2 = 1;
						for (int i3=0; i3<row; i3++) {
							for (int j3=0; j3<col; j3++) {
								if (i2 == i3 && j2 == j3) { continue; }
								if (i == i3 && j == j3) { continue; }
								if (board[i3].substring(j3, j3+1).equals("#")) {
									continue;
								}
								if (dist[i][j][i2][j2] < dist[i][j][i3][j3] && 
										dist[i2][j2][i][j] < dist[i2][j2][i3][j3]) {
									ok++;
								}
								else if (dist[i][j][i2][j2] == dist[i][j][i3][j3] && i2<i3 && j2<j3) {
									ok++;
								}
								else if (dist[i2][j2][i][j] == dist[i2][j2][i3][j3] && i<i3 && j<j3) {
									ok++;
								}
							}
						}
						
						double combo1 = 1;
						double combo2 = 1;
						for (int k = 0; k<r-2; k++) {
							combo1 *= (ok) - k;
							combo1 /= 1 + k;
						}
						for (int k = 0; k<r; k++) {
							combo2 *= (n) - k;
							combo2 /= 1 + k;
						}
						total += combo1 / combo2; //*(double)((ok)*(ok-1)) / (double)((n-2)*(n-3));
						//System.out.println(""+combo1+", "+1+", "+((double)((ok)*(ok-1)) / (double)((n-2)*(n-3))));
					}
				}
			}
		}		
		
		return (double)(total) / 2;// / (double)r;// * (double)(r*(r-1)) / (double)(n*(n-1));
	}
	
	public double getExpected(String[] board, int r) {
		int row = board.length;
		int col = board[0].length();
		dist = new double[row][col][row][col];
		int numS = 0;
		for (int i=0; i<row; i++) {
			for (int j=0; j<col; j++) {
				if (!board[i].substring(j, j+1).equals("#")) {
					numS++;
				}
			}
		}
		double total = sumGraph(board, row, col, r, numS);
		long masu = (row*col) - numS;
		long totalNum = 1;
		for (long i=0; i<r; i++) {
			totalNum *= (masu-i);
		}
		double p = (double)(total) / (masu * (masu-1) / 2);
		//System.out.println("" + p + ", " + total + ", " + totalNum);
		double ex = 0;
		for (long i=0; i<r; i++) {
			double pn = 1, pnn = 1;
			for (long j=0; j<i; j++) {
				pn *= p;
			}
			for (long j=i; j<r; j++) {
				pnn *= 1-p;
			}
			ex += i*pn*pnn;
		}
		
		return total;
	}
	

	public static void main(String [] var) {
		System.out.println("" + (new ClosestRabbit()).getExpected((new String[]{
					 ".#####.#####..#....#",
					 "#......#....#.##..##",
					 ".####..#####..#.##.#",
					 ".....#.#...#..#....#",
					 "#####..#....#.#....#"
		}),
		19));
	}
}
