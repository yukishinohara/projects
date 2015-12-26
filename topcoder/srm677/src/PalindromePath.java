import java.util.ArrayList;


public class PalindromePath {
	static final int SUPER_LEARGE = 100000;
	int [][]mems = null;
	int [][]mems2 = null;
	
	int shortest_r(int p0, int p1, ArrayList<Character> []hen, ArrayList<Integer>[] hen2, int n, int l) {
		if (p0 == p1) {
			mems[p0][p1] = 0;
			return 0;
		}
		if (hen2[p0].contains(p1)) {
			assert hen2[p1].contains(p0);
			mems[p0][p1] = 1;
			return 1;
		}
		if (l > n*(n-1)/2) {
			return SUPER_LEARGE;
		}
		if (mems[p0][p1] >= 0 && mems2[p0][p1] < l) { return mems[p0][p1]; }
		
		int minlen = SUPER_LEARGE;
		for (int i=0; i<hen[p0].size(); i++) {
			for (int j=0; j<hen[p1].size(); j++) {
				if (hen[p0].get(i) == hen[p1].get(j)) {
					int n0 = hen2[p0].get(i);
					int n1 = hen2[p1].get(j);
					int clen = 2 + shortest_r(n0, n1, hen, hen2, n, l + 1);
					if (clen < minlen) {
						minlen = clen;
					}
				}
			}
		}
		
		mems[p0][p1] = minlen;
		mems2[p0][p1] = l;
		return minlen;
	}
	
	public int shortestLength(int n, int[] a, int[] b, String c) {
		int m = a.length;
		ArrayList<Character> [] hen = new ArrayList[n];
		ArrayList<Integer> [] hen2 = new ArrayList[n];
		mems = new int[n][n];
		mems2 = new int[n][n];
		char [] moji = c.toCharArray();
		for (int i=0; i<n; i++) {
			hen[i] = new ArrayList<Character>();
			hen2[i] = new ArrayList<Integer>();
			for (int j=0; j<n; j++) {
				mems[i][j] = -1;
				mems2[i][j] = -1;
			}
		}
		for (int i=0; i<m; i++) {
			hen[a[i]].add(moji[i]);
			hen[b[i]].add(moji[i]);
			hen2[a[i]].add(b[i]);
			hen2[b[i]].add(a[i]);
		}
	
		int ans = shortest_r(0, 1, hen, hen2, n, 0);
		if (ans >= SUPER_LEARGE) {
			return -1;
		}
		return ans;
	}
	
	public static void main(String [] hen) {
		PalindromePath p = new PalindromePath();
		System.out.println(""+p.shortestLength(
				20, 
				new int[]{0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 1, 11, 12, 13, 14, 15, 16, 17, 17, 18}, 
				new int[]{2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19},
				"yaxayaxatyaaaaaaaaaax"));
	}
}
