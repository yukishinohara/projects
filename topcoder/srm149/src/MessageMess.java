// Takes too long

public class MessageMess {
	public String nextstr_r(String instr, String[] dic) {
		if (instr.equals("")) {
			return "";
		}
		int ccnt = 0;
		String ans = "";
		for (int i=0; i<dic.length; i++) {
			int wlen = dic[i].length();
			if (wlen <= instr.length() &&
					instr.substring(0, wlen).equals(dic[i])) {
				String nextstr = nextstr_r(instr.substring(wlen), dic);
				String cans = dic[i] + " " + nextstr;
				if (nextstr.equals("")) {
					cans = dic[i];
				}
				if (nextstr.equals("IMPOSSIBLE!")) {
					continue;
				}
				if (nextstr.equals("AMBIGUOUS!")) {
					return nextstr;
				}
				ccnt++;
				if (ccnt > 1) {
					return "AMBIGUOUS!";
				}
				ans = "" + cans;
			}
		}
		
		if (ccnt <= 0) {
			return "IMPOSSIBLE!";
		}
		return ans;
	}
	
	public String restore(String[] dictionary, String message) {
		return nextstr_r(message, dictionary);
	}
	public static void main(String [] var) {
		MessageMess m = new MessageMess();
		System.out.println(m.restore(new String[]{"A", "B", "S", "D"}, "E"));
	}
}
