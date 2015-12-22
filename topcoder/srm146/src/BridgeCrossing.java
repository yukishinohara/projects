import java.util.Arrays;


public class BridgeCrossing {
	
	int [] mem_stat = null;
	
	public int getOptimalTime(int stat, int[] times) {
		if (mem_stat[stat] != 0) {
			return mem_stat[stat];
		}

		int num = times.length;
		if (num == 1) {
			return times[0];
		}
		
		int min = Integer.MAX_VALUE / 100;
		for (int i=0; i<num; i++) {
			for (int j=i+1; j<num; j++) {
				if ((stat & (1 << i)) == 0 &&
						(stat & (1 << j)) == 0) {
					// Right
					int stat2 = stat + (1 << i) + (1 << j);					
					int tmp = times[i]>times[j] ? times[i] : times[j];
					if (stat2 == ((1 << num) - 1)) {
						return tmp;
					}
					for (int k=0; k<num; k++) {
						if ((stat2 & (1 << k)) != 0) {
							// Left
							int stat3 = stat2 - (1 << k);
							int tmp2 = tmp + times[k];
							int ret = tmp2 + getOptimalTime(stat3, times);
							if (min > ret) {
								min = ret;
							}
						}
					}
				}
			}
		}

		mem_stat[stat] = min;
		return min;
	}
	
	public int minTime(int[] times) {
		int num = times.length;
		mem_stat = new int[1 << (num+1)];
		Arrays.fill(mem_stat, 0);
		return getOptimalTime(0, times);		
	}
	
	public static void main(String []hoge) {
		BridgeCrossing b = new BridgeCrossing();
		int res = b.minTime(new int []{99, 13, 67, 32, 5, 17});
		System.out.println("" + res);
	}
}
