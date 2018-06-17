package universe;

import java.util.*;
import java.io.*;

public class Solution {

    static boolean swapOneTime(int[] hist) {
        int len = hist.length;
        for (int i=len-1; i>0; i--) {
            if (hist[i] == 0) {
                continue;
            }
            hist[i]--;
            hist[i-1]++;
            return true;
        }
        return false;
    }

    static boolean isOkay(long D, int[] hist) {
        long current = 0;
        int len = hist.length;
        for (int i=0; i<len; i++) {
            if (hist[i] == 0) {
                continue;
            }
            current += (1 << i) * hist[i];
        }
        return (current <= D);
    }

    static String solve(long D, String P) {
        // make a histogram
        int[] hist = new int[32];  // P<=30
        for (int i=0; i<32; i++) { hist[i] = 0; }

        // scan P
        int len = P.length();
        for (int i=0, current=0; i<len; i++) {
            if (P.charAt(i) == 'C') {
                current++;
                continue;
            }
            hist[current]++;
        }

        // make answer
        int ans = 0;
        while (!isOkay(D, hist)) {
            if (!swapOneTime(hist)) {
                return "IMPOSSIBLE";
            }
            ans++;
        }
        return "" + ans;
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
        int t = in.nextInt(); // Scanner has functions to read ints, longs, strings, chars, etc.
        for (int i = 1; i <= t; ++i) {
            long d = in.nextLong();
            String p = in.next();
            String ans = solve(d, p);
            System.out.println("Case #" + i + ": " + ans);
        }
    }
}
