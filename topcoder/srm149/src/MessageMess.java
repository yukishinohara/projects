// Takes too long

public class MessageMess {
	String [] mem_nextstr = null;
	
	public String nextstr_r(String instr, String[] dic) {
		if (instr.equals("")) {
			mem_nextstr[instr.length()] = "";
			return "";
		}
		if (mem_nextstr[instr.length()] != null) {
			return mem_nextstr[instr.length()];
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
					mem_nextstr[instr.length()] = nextstr;
					return nextstr;
				}
				ccnt++;
				if (ccnt > 1) {
					mem_nextstr[instr.length()] = "AMBIGUOUS!";
					return "AMBIGUOUS!";
				}
				ans = "" + cans;
			}
		}
		
		if (ccnt <= 0) {
			mem_nextstr[instr.length()] = "IMPOSSIBLE!";
			return "IMPOSSIBLE!";
		}
		mem_nextstr[instr.length()] = ans;
		return ans;
	}
	
	public String restore(String[] dictionary, String message) {
		mem_nextstr = new String[message.length()+1];
		for (int i=0; i<mem_nextstr.length; i++) { mem_nextstr[i]=null; }
		return nextstr_r(message, dictionary);
	}
	public static void main(String [] var) {
		MessageMess m = new MessageMess();
		System.out.println(m.restore(new String[]{"A", "B", "S", "D"}, "E"));
	}
}
