
public class CeyKaps {
	public String applyRule(String swi, String moji) {
		String [] rule = swi.split(":");
		if (moji.equals(rule[0])) {
			return "" + rule[1];
		}
		else if (moji.equals(rule[1])) {
			return "" + rule[0];
		}
		return "" + moji;
	}
	
	public String decipher(String typed, String[] switches) {
		int lentyped = typed.length();
		String ans = "";
		for (int i=0; i<lentyped; i++) {
			String moji = typed.substring(i, i+1);
			for (int j=0; j<switches.length; j++) {
				moji = applyRule(switches[j], moji);
			}
			ans = "" + ans + moji;
		}
		
		return ans;
	}
}

