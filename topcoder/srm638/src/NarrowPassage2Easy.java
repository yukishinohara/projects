import java.util.ArrayList;


public class NarrowPassage2Easy {
	
	int total = 0;
	ArrayList<String> already = new ArrayList<String>();
	int [] sizeg= null;
	int limitg = 0;
	
	void term_action(int[] perm) {
		String disp = "";
		for (int j=0; j<perm.length; j++) {
			disp = "" + disp + sizeg[perm[j]];
		}
		int ok = 1;
		for (int i=0; i<perm.length; i++) {			
			for (int j = 0; j < i; j++) {
				if (perm[j] > perm[i] && (sizeg[perm[j]] + sizeg[perm[i]]) > limitg) {
					ok = 0;
				}
			}
		}
		if (ok == 0) {
			return;
		}
		
		total++;
	}
	
	void permutation_r(int []perm, int c) {
		int n = perm.length;
		if (c >= n) {
			term_action(perm);
			return;
		}
	
		for (int i=c; i<n; i++) {
			int tmp;
			tmp = perm[i]; perm[i] = perm[c]; perm[c] = tmp;
			permutation_r(perm, c+1);
			tmp = perm[i]; perm[i] = perm[c]; perm[c] = tmp;
		}
	}

	void permutation(int n) {
		if (n < 1) { return; }
		int [] perm = new int[n];
		for (int i=0; i<n; i++) {
			perm[i] = i;
		}
		permutation_r(perm, 0);
	}
	
	public int count(int[] size, int maxSizeSum) {
		sizeg = size;
		limitg = maxSizeSum;
		int [] perm = new int[size.length];
		for (int i=0; i<size.length; i++) {
			perm[i] = i;
		}
		permutation(size.length);
		return total;
	}
	
	public static void main(String [] a) {
		System.out.println("" + (new NarrowPassage2Easy()).count(new int[] 
		                                                                 {2,4,6,1,3,5}
		, 8));
	}
}
