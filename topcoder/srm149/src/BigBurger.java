
public class BigBurger {
	public int maxWait(int[] arrival, int[] service) {
		int cnum = arrival.length;
		int [] order = new int[cnum];
		int [] finish = new int[cnum];
		int [] total = new int[cnum];
		int [] wait = new int[cnum];
		for (int i=0; i<cnum; i++) {
			if (i==0) { order[i] = arrival[i]; }
			else { order[i] = arrival[i] > finish[i-1] ? arrival[i] : finish[i-1]; }
			finish[i] = order[i] + service[i];
			total[i] = finish[i] - arrival[i];
			wait[i] = total[i] - service[i];
		}
		
		int maxwait = 0;
		for (int i=0; i<cnum; i++) {
			if (wait[i] > maxwait) {
				maxwait = wait[i];
			}
		}
		
		return maxwait;
	}
	public static void main(String [] var) {
		System.out.println("1!4!9!!");
	}
}
