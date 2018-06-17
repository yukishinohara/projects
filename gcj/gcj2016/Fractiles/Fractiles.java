import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by yukis on 2016/04/09.
 */
public class Fractiles {

    private int numTest;
    private int[] Ks;
    private int[] Cs;
    private int[] Ss;

    public void fileRead(String filePath) {
        FileReader fr = null;
        BufferedReader br = null;
        try {
            fr = new FileReader(filePath);
            br = new BufferedReader(fr);

            String line;
            line = br.readLine();
            numTest = Integer.parseInt(line);

            Ks = new int[numTest];
            Cs = new int[numTest];
            Ss = new int[numTest];

            for (int i=0; i<numTest; i++) {
                line = br.readLine();
                String []line2 = line.split(" ");
                Ks[i] = Integer.parseInt(line2[0]);
                Cs[i] = Integer.parseInt(line2[1]);
                Ss[i] = Integer.parseInt(line2[2]);
            }
            int forDebug=0;
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                br.close();
                fr.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    private long getIndex(int K, ArrayList<Integer> path) {
        long ans = 0;
        int length = path.size();
        for (int i = 0; i < length; i++) {
            ans = (ans * K) + path.get(i).intValue();
        }
        return ans + 1;
    }

    private ArrayList<Long> solve(int K, int C, int S) {
        int comp = 0;
        ArrayList<Integer> path = new ArrayList<Integer>();
        ArrayList<Long> ans = new ArrayList<Long>();

        for (int k = 0; k < K; k++) {
            path.add(k);
            comp++;
            if (k == K-1) {
                for (int j=comp; j < C; j++) {
                    path.add(k);
                }
                ans.add(getIndex(K, path));
                path = new ArrayList<Integer>();
                comp = 0;
            } else if (comp >= C) {
                ans.add(getIndex(K, path));
                path = new ArrayList<Integer>();
                comp = 0;
            }
        }
        if (ans.size() > S) {
            return null;
        } else {
            return ans;
        }
    }

    public void solve(int t) {
        ArrayList<Long> ans = solve(Ks[t], Cs[t], Ss[t]);
        if (ans == null) {
            System.out.println("Case #" + (t+1) + ": IMPOSSIBLE");
        } else {
            String ansstr = "";
            int length = ans.size();
            for (int i=0; i<length; i++) {
                ansstr += ans.get(i) + ((i==length-1) ? "" : " ");
            }
            System.out.println("Case #" + (t+1) + ": " + ansstr);
        }
    }

    public void solveAll() {
        for (int i=0; i<numTest; i++) {
            solve(i);
        }
    }

    public static void main(String []args) {
        Fractiles obj = new Fractiles();
        obj.fileRead("dat\\D-large.in");
        obj.solveAll();
    }
}
