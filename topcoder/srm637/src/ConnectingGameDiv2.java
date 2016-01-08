// Does not pass
import java.util.HashMap;


public class ConnectingGameDiv2 {
	
	HashMap<String, Integer> mem_best = new HashMap<String, Integer>();
	HashMap<String, Integer> numregion = new HashMap<String, Integer>();

	public String makeStat(String already, int i, int j) {
		String ans = "" + already + "," + i + "," + j;
		return ans;
	}
	
	public int min_r(String[] board, String already, int i, int j) {
		if (j >= board[0].length()) {
			return 0;
		}
		
		String stat = makeStat(already, i, j);
		if (mem_best.containsKey(stat)) {
			return mem_best.get(stat);
		}
		String current = board[i].substring(j, j+1);
		
		int celltotal = 0;
		if (!already.contains(current)) {
			already = "" + already + current;
			celltotal += numregion.get(current);
		}
		
		int minans = 10000000;
		if (i > 0) {
			int tmpans = min_r(board, already, i-1, j+1);
			if (tmpans < minans) {
				minans = tmpans;
			}
		}
		if (i < board.length-1) {
			int tmpans = min_r(board, already, i+1, j+1);
			if (tmpans < minans) {
				minans = tmpans;
			}
		}
		int tmp = min_r(board, already, i, j+1);
		if (tmp < minans) {
			minans = tmp;
		}
		
		celltotal += minans;
		
		mem_best.put(stat, celltotal);
		return celltotal;
	}
	
	public int getmin(String[] board) {
		for (int i=0; i<board.length; i++) {
			for (int j=0; j<board[0].length(); j++) {
				String hoge = board[i].substring(j, j+1);
				int orig = 0;
				if (numregion.containsKey(hoge)) {
					orig = numregion.get(hoge);
				}
				orig++;
				numregion.put(hoge, orig);
			}
		}
		
		int minans = 10000000;
		for (int i=0; i<board.length; i++) {
			int tmp = min_r(board, "", i, 0);
			if (tmp < minans) {
				minans = tmp;
			}
		}


		return minans;
	}


	
	public static void main(String []va) {
		System.out.println("" + (new ConnectingGameDiv2()).getmin(new String[] {
				"AAAAAAaaabcdefg", 
				"AAAAAAhhDDDDDDD", 
				"AAAAiAjDDDDDDDD", 
				"AAAAiijDDDDDDDD", 
				"AAAAAAAkDDDDDDD", 
				"AAAAoAAAlDDDDDD", 
				"AAApBnAAlDDDDDD", 
				"srqBBBmmmmDDDDD"
		}));
	}
}
