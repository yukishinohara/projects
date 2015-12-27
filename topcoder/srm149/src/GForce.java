// Does not pass

public class GForce {
	public double avgAccel(int period, int[] accel, int[] time) {
		int numpoint = time.length;
		int start = time[0];
		for (int i=0; i<numpoint; i++) { time[i] -= start; }
		for (int i=0; i<numpoint; i++) { time[i] *= 4; }
		period *= 4;
		
		int end = time[numpoint - 1];
		double [] ggg = new double[end +10];
		
		for (int i=0; i<numpoint-1; i++) {
			for (int t=time[i]; t<time[i+1]; t++) {
				int dt = t - time[i];
				ggg[t] = (double)accel[i] + 
					((double)dt*(double)(accel[i+1] - accel[i])/(double)(time[i+1]-time[i]));
			}
		}
		ggg[end] = accel[numpoint-1];
		
		double maxavg = 0.0;
		int max_start = 0;
		int max_end = 0;
		for (int i=0; i<=(end-period); i++) {
			double tmpsum = 0.0;
			for (int j=i; j<(i+period); j++) {
				tmpsum += ((ggg[j]+ggg[j+1])/2.0);
			}
			tmpsum = tmpsum / (double)(period);
			if (tmpsum > maxavg) {
				maxavg = tmpsum;
				max_start = i;
				max_end = i+period;
			}
		}
		
		if (max_start != 0) {
			double tmpsum2 = 0.0;
			for (int j=max_start; j<(max_end-1); j++) {
				tmpsum2 += ((ggg[j]+ggg[j+1])/2.0);
			}
			for (double delta=0.0; delta < 1.0; delta += 0.000001) {
				double a = (ggg[max_start-1]-ggg[max_start]) * delta + ggg[max_start];
				double b = (ggg[max_end] - ggg[max_end-1]) * (1.0-delta) + ggg[max_end-1];
				double tmpsum = tmpsum2 +
					((ggg[max_start] + a) * delta * 0.5) +
					((ggg[max_end-1] + b) * (1.0-delta) * 0.5);
				tmpsum = tmpsum / (double)(period);
				if (tmpsum > maxavg) {
					maxavg = tmpsum;
				}
			}
		}
				
		return maxavg;
	}
	
	public static void main(String []a) {
		GForce g = new GForce();
		System.out.println("" + g.avgAccel(68, 
				new int[]{0, 100, 50, 110, 0},
				new int[]{0, 100, 110, 170, 180}));
	}
}
