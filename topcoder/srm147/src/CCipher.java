
public class CCipher {
	public String decode(String cipherText, int shift) {
		char [] angou = cipherText.toCharArray();
		int len = angou.length;
		char [] hirabun = new char[len];
		
		for (int i=0; i<len; i++) {
			hirabun[i] = (char)((((angou[i] - (int)'A' + (int)((int)'Z' - (int)'A' + 1)) - shift) % ((int)'Z' - (int)'A' + 1)) + (int)'A');
		}
		
		return String.copyValueOf(hirabun);
	}
}
