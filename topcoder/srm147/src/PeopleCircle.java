
public class PeopleCircle {
	
	int nextIndex(char []B, int K, int crr, int n) {
		for (int i=0; i<K; i++) {
			crr = (crr + 1) % n;
			while (B[crr] == 'F') { crr = (crr + 1) % n; }
		}
		
		return crr;
	}
	
	public String order(int numMales, int numFemales, int K) {
		int n = numMales + numFemales;
		char [] B = new char[n];
		for (int i=0; i<n; i++) {
			B[i] = '0';
		}
		
		int crr = n-1;
		for (int i=0; i<numFemales; i++) {
			crr = nextIndex(B, K, crr, n);
			B[crr] = 'F';
		}
		for (int i=0; i<n; i++) {
			if (B[i] == '0') { B[i] = 'M'; }
		}
		
		return String.copyValueOf(B);
	}
}
