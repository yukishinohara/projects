
public class MNS {
	int global_ans = 0;
	public int all_check = 0;
	
	boolean isok(int []numbers, int []idx) {
		int sum = 0;
		all_check++;
		
		for (int i=0; i<3; i++) {
			sum += numbers[idx[i]];
		}
		
		for (int i=0; i<3; i++) {
			int local_sum = 0;
			for (int j=0; j<3; j++) {
				local_sum += numbers[idx[i*3+j]];
			}
			if (local_sum != sum) {
				return false;
			}
		}
		for (int j=0; j<3; j++) {
			int local_sum = 0;
			for (int i=0; i<3; i++) {
				local_sum += numbers[idx[i*3+j]];
			}
			if (local_sum != sum) {
				return false;
			}
		}
		
		return true;
	}
	
	void search(int[] numbers, int[] already) {
		if (already[numbers.length-1] >= 0) {
			if (isok(numbers, already)) {
				global_ans++;
			}
			return;
		}
		for (int i=0; i<numbers.length; i++) {
			int flag = 0;
			for (int j=0; j<numbers.length; j++) {
				if (already[j] == i) {
					flag = 1;
				}
			}
			
			if (flag != 1) {
				int[] p = new int[numbers.length];
				for (int k = 0; k<numbers.length; k++) {
					p[k] = already[k];
				}
				for (int k = 0; k<numbers.length; k++) {
					if (already[k] < 0) {
						p[k] = i;
						break;
					}
				}
				search(numbers, p);
			}
		}
	}
	public int dupdup(int[] numbers) {
		int [] dups = new int[10];
		int ans = 1;
		for (int i=0; i<dups.length; i++) {
			dups[i] = 0;
		}
		for (int i=0; i<numbers.length; i++) {
			dups[numbers[i]] ++;
		}
		for (int i=0; i<dups.length; i++) {
			int seki = 1;
			for (int j=dups[i]; j>1; j--) {
				seki *= j;
			}
			ans *= seki;
		}
		
		return ans;
	}
	public int combos(int[] numbers) {
		int [] already = new int[numbers.length];
		global_ans = 0;
		for (int i=0; i<numbers.length; i++) {
			already[i] = -1;
		}
		search(numbers, already);
		return global_ans / dupdup(numbers);
	}
	
	public static void main(String []hoge) {
		MNS mns = new MNS();
		int aa = mns.combos(new int [] {1,2,3,3,2,1,2,2,2});
		System.out.println("" + aa);
		System.out.println("" + mns.all_check);
	}
}
