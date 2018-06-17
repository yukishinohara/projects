package ufo;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Scanner;

public class Solution {

    static double findTwoAxis(double A) {
        double lim = Math.PI / 4.0;
        for (double q = 0.0; q < lim; q += 0.0000005) {
            double c = Math.sin(q) / Math.sqrt(2);
            double d = Math.cos(q);
            double s = Math.sqrt(2) * (c + d);
            if (Math.abs(A - s) < 0.0000005) {
                return q;
            }
        }
        System.err.println("FATAL");
        return lim;
    }

    static double findOneAxis(double A) {
        double lim = Math.PI / 4.0;
        for (double p = 0.0; p < lim; p += 0.0000005) {
            double s = Math.sin(p) + Math.cos(p);
            if (Math.abs(A - s) < 0.0000005) {
                return p;
            }
        }
        System.err.println("FATAL");
        return lim;
    }

    static String rotateOneAxis(double P) {
        double x1 = 0.5 * Math.cos(P);
        double y1 = 0.5 * Math.sin(P);
        double z1 = 0;

        double x2 = (-0.5) * Math.sin(P);
        double y2 = 0.5 * Math.cos(P);
        double z2 = 0;

        double x3 = 0;
        double y3 = 0;
        double z3 = 0.5;

        return  "" + x1 + " " + y1 + " " + z1 + "\n" +
                "" + x2 + " " + y2 + " " + z2 + "\n" +
                "" + x3 + " " + y3 + " " + z3;
    }

    static String rotateTwoAxis(double Q) {
        double P = Math.PI / 4.0;
        double x1 = 0.5 * Math.cos(P);
        double y1o = 0.5 * Math.sin(P);
        double y1 = y1o * Math.cos(Q);
        double z1 = y1o * Math.sin(Q);

        double x2 = (-0.5) * Math.sin(P);
        double y2o = 0.5 * Math.cos(P);
        double y2 = y2o * Math.cos(Q);
        double z2 = y2o * Math.sin(Q);

        double x3 = 0;
        double y3 = 0.5 * Math.sin(Q);
        double z3 = 0.5 * Math.cos(Q);

        return  "" + x1 + " " + y1 + " " + z1 + "\n" +
                "" + x2 + " " + y2 + " " + z2 + "\n" +
                "" + x3 + " " + y3 + " " + z3;
    }

    static String solve(double A) {
        if (A < Math.sqrt(2)) {
            return rotateOneAxis(findOneAxis(A));
        } else {
            return rotateTwoAxis(findTwoAxis(A));
        }
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
        int t = in.nextInt(); // Scanner has functions to read ints, longs, strings, chars, etc.
        for (int i = 1; i <= t; ++i) {
            double a = in.nextDouble();
            String ans = solve(a);
            System.out.println("Case #" + i + ":\n" + ans);
        }
    }
}
