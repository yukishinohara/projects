import java.util.ArrayList;
import java.util.HashMap;


public class ClosestRabbit {
	class UF {

	    private int[] parent;  // parent[i] = parent of i
	    private byte[] rank;   // rank[i] = rank of subtree rooted at i (never more than 31)
	    private int count;     // number of components

	    /**
	     * Initializes an empty union-find data structure with <tt>N</tt> sites
	     * <tt>0</tt> through <tt>N-1</tt>. Each site is initially in its own 
	     * component.
	     *
	     * @param  N the number of sites
	     * @throws IllegalArgumentException if <tt>N &lt; 0</tt>
	     */
	    public UF(int N) {
	        if (N < 0) throw new IllegalArgumentException();
	        count = N;
	        parent = new int[N];
	        rank = new byte[N];
	        for (int i = 0; i < N; i++) {
	            parent[i] = i;
	            rank[i] = 0;
	        }
	    }

	    /**
	     * Returns the component identifier for the component containing site <tt>p</tt>.
	     *
	     * @param  p the integer representing one site
	     * @return the component identifier for the component containing site <tt>p</tt>
	     * @throws IndexOutOfBoundsException unless <tt>0 &le; p &lt; N</tt>
	     */
	    public int find(int p) {
	        validate(p);
	        while (p != parent[p]) {
	            parent[p] = parent[parent[p]];    // path compression by halving
	            p = parent[p];
	        }
	        return p;
	    }

	    /**
	     * Returns the number of components.
	     *
	     * @return the number of components (between <tt>1</tt> and <tt>N</tt>)
	     */
	    public int count() {
	        return count;
	    }
	  
	    /**
	     * Returns true if the the two sites are in the same component.
	     *
	     * @param  p the integer representing one site
	     * @param  q the integer representing the other site
	     * @return <tt>true</tt> if the two sites <tt>p</tt> and <tt>q</tt> are in the same component;
	     *         <tt>false</tt> otherwise
	     * @throws IndexOutOfBoundsException unless
	     *         both <tt>0 &le; p &lt; N</tt> and <tt>0 &le; q &lt; N</tt>
	     */
	    public boolean connected(int p, int q) {
	        return find(p) == find(q);
	    }
	  
	    /**
	     * Merges the component containing site <tt>p</tt> with the 
	     * the component containing site <tt>q</tt>.
	     *
	     * @param  p the integer representing one site
	     * @param  q the integer representing the other site
	     * @throws IndexOutOfBoundsException unless
	     *         both <tt>0 &le; p &lt; N</tt> and <tt>0 &le; q &lt; N</tt>
	     */
	    public void union(int p, int q) {
	        int rootP = find(p);
	        int rootQ = find(q);
	        if (rootP == rootQ) return;

	        // make root of smaller rank point to root of larger rank
	        if      (rank[rootP] < rank[rootQ]) parent[rootP] = rootQ;
	        else if (rank[rootP] > rank[rootQ]) parent[rootQ] = rootP;
	        else {
	            parent[rootQ] = rootP;
	            rank[rootP]++;
	        }
	        count--;
	    }

	    // validate that p is a valid index
	    private void validate(int p) {
	        int N = parent.length;
	        if (p < 0 || p >= N) {
	            throw new IndexOutOfBoundsException("index " + p + " is not between 0 and " + (N-1));  
	        }
	    }
	}
	
	
	HashMap<String, Long> already = new HashMap<String, Long>();
	
	public long sumGraph(String[] board, int r, int row, int col, int orig_r) {
		long total = 0;
		if (r == 0) {
			UF uf = new UF(orig_r);
			ArrayList<Integer> x = new ArrayList<Integer>();
			ArrayList<Integer> y = new ArrayList<Integer>();
			for (int i=0; i<row; i++) {
				for (int j=0; j<col; j++) {
					if (board[i].substring(j, j+1).equals("X")) {
						x.add(i);
						y.add(j);
					}
				}
			}
			for (int i=0; i<orig_r; i++) {
				int mindest = 10000000;
				int minindex = i;
				for (int j=0; j<orig_r; j++) {
					if (i == j) { continue; }
					int tmp = (x.get(i) - x.get(j))*(x.get(i) - x.get(j)) + 
							(y.get(i) - y.get(j))*(y.get(i) - y.get(j));
					if (tmp < mindest) {
						mindest = tmp;
						minindex = j;
					}
				}
				uf.union(i, minindex);
			}
			return uf.count();
		}
		String zenbu = "";
		for (int i=0; i<board.length; i++) {
			zenbu += "" + board[i];
		}
		if (already.containsKey(zenbu)) {
			return already.get(zenbu);
		}
		for (int i=0; i<row; i++) {
			for (int j=0; j<col; j++) {
				if (board[i].substring(j, j+1).equals(".")) {
					String tmp = board[i];
					String usiro = "";
					if (j < col-1) { usiro = tmp.substring(j+1); }
					board[i] = "" + tmp.substring(0, j) + "X" + usiro; 
					total += sumGraph(board, r-1, row, col, orig_r);
					board[i] = tmp;
				}
			}
		}
		already.put(zenbu, total);
		return total;
	}
	
	public double getExpected(String[] board, int r) {
		int row = board.length;
		int col = board[0].length();
		long total = sumGraph(board, r, row, col, r);
		int numS = 0;
		for (int i=0; i<row; i++) {
			for (int j=0; j<col; j++) {
				if (board[i].substring(j, j+1).equals("#")) {
					numS++;
				}
			}
		}
		long masu = (row*col) - numS;
		long totalNum = 1;
		for (long i=0; i<r; i++) {
			totalNum *= (masu-i);
		}
		//System.out.println("" + total);
		return (double)(total) / totalNum;
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
