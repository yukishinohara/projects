import java.util.ArrayList;


public class InterestingDigits {
	public ArrayList<Integer> discBase(int num, int base) {
		ArrayList<Integer> ans = new ArrayList<Integer>();
		for (int x = num; x > 0; x = x / base) {
			ans.add((x % base));
		}
		return ans;
	}
	public int[] digits(int base) {
		ArrayList<Integer> ans = new ArrayList<Integer>();
		for (int i=2; i<base; i++) {
			for (int j=i; j<base*base*base; j += i) {
				ArrayList<Integer> desc = discBase(j, base);
				int desc_size = desc.size();
				int sumup = 0;
				for (int k=0; k<desc_size; k++) {
					sumup += desc.get(k);
				}
				if (sumup % i != 0) {
					break;
				}
				if (j+i >= base*base*base) {
					ans.add(i);
				}
			}
		}
		
		int [] ians = new int[ans.size()];
		for (int i=0; i<ans.size(); i++) { ians[i] = ans.get(i); }
		return ians;
	}
	
	public static void main(String [] var) {
		InterestingDigits ig = new InterestingDigits();
		int [] arr = ig.digits(13);
		int num = arr.length;
		for (int i=0; i<num; i++) {
			System.out.println(""+arr[i]);
		}
	}
}
