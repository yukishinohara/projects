import java.util.ArrayList;
import java.util.Arrays;


public class GoldenChain {
	public static int minCuts(int[] sections) {
		Arrays.sort(sections);
		ArrayList<Integer> hai = new ArrayList<Integer>();
		for (int i=0; i<sections.length; i++) {
			hai.add(sections[i]);
		}
		
		int joint = 0;
		while(hai.size() != 0) {
			if ( (joint + hai.get(0)) > hai.size() ) 
				{ 
					if (joint > hai.size()) { return joint; }
					return hai.size(); 
				}
			if ( (joint + hai.get(0)) == hai.size()-1 ) 
				{ return hai.size() - 1; }
			joint += hai.get(0);
			hai.remove(0);
		}
		
		return -1;
	}
	
	public static void main(String [] h) {
		int x = minCuts(new int []{1,1,1,1,1});
		System.out.println(""+x);
	}
}
