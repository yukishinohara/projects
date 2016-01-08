import java.util.HashMap;


public class PathGameDiv2 {
	HashMap<String, Integer> mem_best = new HashMap<String, Integer>();
	
	int bestone_r(String[] board, int r, int i) {
		String stat = "" + r + "," + i;
		if (mem_best.containsKey(stat)) {
			return mem_best.get(stat);
		}
		
		int opp = (r + 1) % 2;
		//System.out.println("" + r + ", " + i);
		if (i >= board[0].length()) {
			mem_best.put(stat, 0);
			return 0;
		}
		if (board[r].substring(i, i+1).equals("#")){
			mem_best.put(stat, -10000000);
			return -10000000;
		}
		if (board[opp].substring(i, i+1).equals("#")){
			int ans = bestone_r(board, r, i+1);
			mem_best.put(stat, ans);
			return ans;
		}
		int rone = 1 + bestone_r(board, r, i+1);
		int oone = bestone_r(board, opp, i+1);
		//System.out.println("" + rone + ", " + oone + " (" + r + ", " + i);
		int ans = 0;
		if (rone > oone) { ans = rone; }
		else { ans = oone; }
		
		mem_best.put(stat, ans);
		return ans ;
	}
	
	public int calc(String[] board) {
		int a0 = bestone_r(board, 0, 0);
		int a1 = bestone_r(board, 1, 0);
		return (a0 > a1) ? a0 : a1;
	}
	
	public static void main(String []va) {
		System.out.println("" + (new PathGameDiv2()).calc(new String[] {
				"................#...............", ".....................#.........."
		}));
	}
}
