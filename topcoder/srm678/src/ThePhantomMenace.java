
public class ThePhantomMenace {
	public int find(int[] doors, int[] droids) {
		int numDoors = doors.length;
		int numDroids = droids.length;
		
		int maxDoor = 0;
		for (int i=0; i<numDoors; i++) {
			int minDroid = 1000000;
			for (int j=0; j<numDroids; j++) {
				int dist = Math.abs(doors[i] - droids[j]);
				if (dist < minDroid) {
					minDroid = dist;
				}
			}
			if (minDroid > maxDoor) {
				maxDoor = minDroid;
			}
		}
		
		return maxDoor;
	}
}
