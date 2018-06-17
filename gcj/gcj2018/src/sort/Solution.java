package sort;

import java.util.*;
import java.io.*;

public class Solution {
    static String solve(int N, int[]V) {
        int nKisu = (N / 2) + (N % 2);
        int nGusu = (N / 2);
        int []vKisu = new int[nKisu];
        int []vGusu = new int[nGusu];
        int []sortedV = new int[N];

        // Split the list
        for (int i=0; i<N; i++) {
            if (i % 2 == 0) {
                vKisu[i / 2] = V[i];
            } else {
                vGusu[i / 2] = V[i];
            }
        }

        // Sort individually
        Arrays.sort(vKisu);
        Arrays.sort(vGusu);

        // Combine the lists
        for (int i=0; i<N; i++) {
            if (i % 2 == 0) {
                sortedV[i] = vKisu[i / 2];
            } else {
                sortedV[i] = vGusu[i / 2];
            }
        }

        // Check
        Arrays.sort(V);
        for (int i=0; i<N; i++) {
            if (V[i] != sortedV[i]) {
                return "" + i;
            }
        }

        return "OK";
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
        int t = in.nextInt(); // Scanner has functions to read ints, longs, strings, chars, etc.
        for (int i = 1; i <= t; ++i) {
            int n = in.nextInt();
            int []v = new int[n];
            for (int j=0; j<n; j++) {
                v[j] = in.nextInt();
            }
            String ans = solve(n, v);
            System.out.println("Case #" + i + ": " + ans);
        }
    }
}
