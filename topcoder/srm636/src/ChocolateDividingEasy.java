
public class ChocolateDividingEasy {
	int devideNine(int p, int q, int i1, int i2, int j1, int j2) {
		int ans = 0;
		if (p<i1) { ans += 0; }
		else if (p<i2) { ans += 3; }
		else { ans += 6; }
		
		if (q<j1) { ans += 0; }
		else if (q <j2) { ans += 1; }
		else { ans += 2; }
		
		return ans;
	}
	
	public int findBest(String[] chocolate) {
		int r = chocolate.length;
		int c = chocolate[0].length();
		int [][] ch = new int[r][c];
		for (int i=0; i<r; i++) {
			String therow = chocolate[i];
			for (int j=0; j<c; j++) {
				ch[i][j] = ((int)therow.charAt(j)) - (int)'0';
			}
		}
		
		int maxans = 0;
		for (int i1=1; i1<r-1; i1++) {
			for (int i2=i1+1; i2<r; i2++) {
				for (int j1=1; j1<c-1; j1++) {
					for (int j2=j1+1; j2<c; j2++) {
						int tmpmin = 100000000;
						int [] tmpsum = new int[9];
						for (int p=0; p<r; p++) {
							for (int q=0; q<c; q++) {
								tmpsum[devideNine(p, q, i1, i2, j1, j2)] += ch[p][q];
							}
						}
						for (int k=0; k<tmpsum.length; k++) {
							if (tmpsum[k] < tmpmin) {
								tmpmin = tmpsum[k];
							}
						}
						if (tmpmin > maxans) {
							maxans = tmpmin;
						}
						/*
						System.out.print(""+i1+","+i2+","+j1+","+j2+" = [");
						for (int k=0; k<tmpsum.length; k++) {
							System.out.print("" + tmpsum[k] + ",");
						}
						System.out.println("]");
						*/
					}
				}
			}
		}
		
		return maxans;
	}
	public static void main(String [] var) {
		System.out.println("" + (new ChocolateDividingEasy()).findBest(new String[]{
				"36753562",
				"91270936",
				"06261879",
				"20237592",
				"28973612",
				"93194784"
		}));
	}
}
