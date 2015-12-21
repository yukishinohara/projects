
public class ImageDithering {
	public int count(String dithered, String[] screen) {
		String [] diths = dithered.split("");
		int ans = 0;
		
		for (String str: screen) {
			String [] indstr = str.split("");
			for (String foo: indstr) {
				for (String bar: diths) {
					if (foo.equals(bar)) {
						ans++;
					}
				}
			}
		}
		
		return ans;
	}
}
