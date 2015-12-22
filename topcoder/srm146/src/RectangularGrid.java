
public class RectangularGrid {
	public static long countRectangles(int width, int height) {
		long ans = 0;
		for (int i=1; i<=height; i++) {
			for (int j=1; j<=width; j++) {
				if (i != j) {
					ans += ((height - i) + 1) * ((width - j) + 1);
				}
			}
		}
		
		return ans;
	}
}
