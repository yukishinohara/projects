
public class ExerciseMachine {
	public static int getPercentages(String time){
		String [] baka = time.split("");
		int hrs = Integer.parseInt(""+baka[0]+baka[1]);
		int mnt = Integer.parseInt(""+baka[3]+baka[4]);
		int sec = Integer.parseInt(""+baka[6]+baka[7]);
		int tmn = sec + mnt*60 + hrs*3600;
		
		int [] yaku = {1, 2, 4, 5, 10, 20, 25, 50, 100};
		int numYakku = yaku.length;
		for (int i=numYakku -1; i>=0; i--) {
			int wari  = tmn / yaku[i];
			int amari = tmn % yaku[i];
			if (amari == 0 && wari*yaku[i]==tmn) {
				return yaku[i] - 1;
			}
		}
		
		return 0;
	}
	
	public static void main(String []var) {
		System.out.println("" + ExerciseMachine.getPercentages("00:01:40"));
	}
}