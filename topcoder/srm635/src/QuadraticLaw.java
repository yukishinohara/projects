
public class QuadraticLaw {
	boolean canAcceptLate(long x, long total) {
		if (total - x < x) {
			return false;
		}
		if ((total - x) / x > x - 10) {
			long tmp2 = x * x;
			if (tmp2 + x > total) {
				return false;
			}
			return true;
		}
		return false;
	}
	
	public long getTime(long d) {
		long lb = 0L;
		long ub = d;
		long c = d / 2;
		
		while (lb < ub) {
			if (lb == ub || ub-lb == 1) {
				return lb;
			}
			c = (lb + ub) / 2;
			
			if (canAcceptLate(c, d)) {
				//System.out.println("T "+c);
				lb = c;
			} else {
				//System.out.println("F "+c);
				ub = c;
			}
		}
		
		return c; // cannot be happened
	}
	
	public static void main(String []hoge) {
		QuadraticLaw q = new QuadraticLaw();
		System.out.println("" + q.getTime(1000000000000000000L));
	}
}
