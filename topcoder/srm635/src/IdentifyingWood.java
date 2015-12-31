
public class IdentifyingWood {
	public String check(String s, String t) {
		int tlen = t.length();
		String tmpstr = "" + s;
		for (int i=0; i<tlen; i++) {
			int idx = tmpstr.indexOf(t.substring(i, i+1));
			if (idx < 0) {
				return "Nope.";
			}
			if (idx+1 < tmpstr.length()) {
				tmpstr = tmpstr.substring(idx+1, tmpstr.length());
			} else {
				tmpstr = "";
			}
		}
		
		return "Yep, it's wood.";
	}
	public static void main(String [] var) {
		IdentifyingWood i = new IdentifyingWood();
		System.out.println("" + i.check("absdefgh", "asdf"));
	}
}
