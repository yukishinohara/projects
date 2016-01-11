
public class NamingConvention {
	public String toCamelCase(String variableName) {
		String ans = "" + variableName;
		while(true) {
			int idx = ans.indexOf("_");
			if (idx < 0) {
				break;
			}
			String tmp = ans.substring(idx+1);
			tmp = tmp.substring(0,1).toUpperCase() + tmp.substring(1);
			ans = "" + ans.substring(0, idx) + tmp;
		}
		return ans;
	}
	
}
