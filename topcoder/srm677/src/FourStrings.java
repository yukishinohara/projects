
public class FourStrings {
	
	String [] abcd = null;

	String concat(String hidari, String migi) {
		String ans = "" + hidari + migi;
		int hidari_n = hidari.length();
		int migi_n = migi.length();
		if (hidari.contains(migi)) {
			return hidari;
		}
		if (migi.contains(hidari)) {
			return migi;
		}
		
		for (int i=1; i<=hidari_n && i<=migi_n; i++) {
			String hidari_ketu = hidari.substring(hidari_n-i, hidari_n);
			String migi_atama  = migi.substring(0, i);
			if (hidari_ketu.equals(migi_atama)) {
				ans = "" + hidari + migi.substring(i, migi_n);
			}
		}
		
		return ans;
	}
	
	String term_action(int[] perm) {
		String ans = "" + concat(abcd[perm[0]], abcd[perm[1]]);
		ans = concat(ans, abcd[perm[2]]);
		ans = concat(ans, abcd[perm[3]]);
		
		return ans;
	}
	
	String permutation_r(int []perm, int c) {
		int n = perm.length;
		if (c >= n) {
			return term_action(perm);
		}
	
		int minlen = 1000000;
		String minans = "";
		for (int i=c; i<n; i++) {
			int tmp;
			tmp = perm[i]; perm[i] = perm[c]; perm[c] = tmp;
			String tmpans = permutation_r(perm, c+1);
			tmp = perm[i]; perm[i] = perm[c]; perm[c] = tmp;
			
			if (tmpans.length() < minlen) {
				minlen = tmpans.length();
				minans = tmpans;
			}
		}
		
		return minans;
	}

	String permutation(int n) {
		if (n < 1) { return ""; }
		int [] perm = new int[n];
		for (int i=0; i<n; i++) {
			perm[i] = i;
		}
		return permutation_r(perm, 0);
	}
	
	public int shortestLength(String a, String b, String c, String d) {
		abcd = new String[]{a, b, c, d};
		String ans = permutation(4);
		System.out.println(ans);
		return ans.length();
	}

	public static void main(String [] hoge) {
		FourStrings f = new FourStrings();
		System.out.println(f.shortestLength("aba", "b", "b","b"));
	}
}
