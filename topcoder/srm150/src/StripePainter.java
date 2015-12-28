// It does not pass

import java.util.ArrayList;
import java.util.HashMap;


public class StripePainter {
	HashMap<String, Integer> memstat = new HashMap<String, Integer>();
	
	public int getMinSt(String stripes) {
		int slen = stripes.length();
		if (slen == 0) {
			return 0;
		}
		if (slen == 1) {
			return 1;
		}
		if (memstat.containsKey(stripes)) {
			return memstat.get(stripes);
		}
		
		String firstChar = stripes.substring(0, 1);
		boolean allsame = true;
		for (int i=0; i<slen; i++) {
			if (!stripes.substring(i, i+1).equals(firstChar)) {
				allsame = false;
				break;
			}
		}
		if (allsame) {
			return 1;
		}
		
		ArrayList<Integer> sameList = new ArrayList<Integer>();
		for (int i=0; i<slen; i++) {
			if (stripes.substring(i, i+1).equals(firstChar)) {
				sameList.add(i);
			}
		}
		int minans = 1000000;
		int lastsame = sameList.get(sameList.size() - 1);
		for (int i=0; i<sameList.size(); i++) {
			int tmpans = 1;
			for (int j=0; j<i; j++) {
				tmpans += getMinSt(stripes.substring(sameList.get(j), sameList.get(j+1)).replace(firstChar, ""));
			}
			if (sameList.get(i) < lastsame) {
				tmpans += getMinSt(stripes.substring(sameList.get(i)+1, lastsame));
			}
			if (tmpans < minans ) {
				minans = tmpans;
			}
		}
		int lastindex = sameList.get(sameList.size() - 1) + 1;
		if (lastindex < slen) {
			minans += getMinSt(stripes.substring(lastindex, slen).replace(firstChar, ""));
		}

		int ans = minans;
		for (int i=0; i<sameList.size(); i++) {
			int tmpans = 1;
			for (int j=0; j<i; j++) {
				tmpans += getMinSt(stripes.substring(sameList.get(j), sameList.get(j+1)).replace(firstChar, ""));
			}
			if (sameList.get(i) < slen) {
				tmpans += getMinSt(stripes.substring(sameList.get(i)+1, slen));
			}
			if (tmpans < ans ) {
				ans = tmpans;
			}
		}
		
		memstat.put(stripes, ans);
		return ans;
	}
	public int minStrokes(String stripes) {
		memstat = new HashMap<String, Integer>();
		return getMinSt(stripes);
	}
	
	public static void main(String [] var) {
		StripePainter s = new StripePainter();
		String difficult = "ADAEBCBCACBDEAACAEAEABCDABAABCEEBDDCDDDCBEBABDDDBC";
		System.out.println("" + s.minStrokes(difficult));
	}
}
