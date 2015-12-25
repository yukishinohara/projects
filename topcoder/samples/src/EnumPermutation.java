
public class EnumPermutation {
	
	void term_action(int[] perm) {
		String disp = "";
		for (int i=0; i<perm.length; i++) { disp = disp + perm[i] + ", "; }
		System.out.println("[" + disp + "]");
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
	
	public static void main(String [] hage) {
		EnumPermutation ep = new EnumPermutation();
		ep.permutation(5);
	}
}
