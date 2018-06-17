package gopher;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Scanner;

public class Solution {
    static final int OFFSET = 10;

    static int howManyUnPrepared(int [][]G, int X, int Y) {
        int ans = 0;
        for (int x=X; x<X+3; x++) {
            for (int y=Y; y<Y+3; y++) {
                ans += ((G[x][y] == 0) ? 1 : 0);
            }
        }
        return ans;
    }

    static String getNext(int [][]G, int W, int H) {
        int bestAns = -1;
        int ansX = 0;
        int ansY = 0;

        for (int x=0; x<(W-2); x++) {
            for (int y=0; y<(H-2); y++) {
                int ans = howManyUnPrepared(G, x, y);
                if (bestAns <= ans) {
                    bestAns = ans;
                    ansX = x; ansY = y;
                }
            }
        }

        return "" + (OFFSET+1+ansX) + " " + (OFFSET+1+ansY);
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
        int t = in.nextInt(); // Scanner has functions to read ints, longs, strings, chars, etc.
        for (int i = 1; i <= t; ++i) {
            int a = in.nextInt();
            int w, h;
            int [][]g;

            // A = 20 0r 200
            if (a == 20) {
                w = 4; h = 5;
            } else {
                w = 15; h = 14;
            }

            // Initialize the garden
            g = new int[w][h];
            for (int x=0; x<w; x++) {
                for (int y=0; y<h; y++) {
                    g[x][y] = 0;
                }
            }

            while(true) {
                System.out.println(getNext(g, w, h));
                System.out.flush();
                int x = in.nextInt();
                int y = in.nextInt();
                if (x == 0 && y == 0) {
                    break;
                } else if (x < 0 || y < 0) {
                    System.err.println("FATAL");
                    break;
                }

                g[x - OFFSET][y - OFFSET] = 1;
            }
        }
    }
}
