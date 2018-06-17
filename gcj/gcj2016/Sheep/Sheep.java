import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigInteger;

/**
 * Created by yukis on 2016/04/09.
 */
public class Sheep {

    private int numTest = 0;
    private int[] n = null;

    public void fileRead(String filePath) {
        FileReader fr = null;
        BufferedReader br = null;
        try {
            fr = new FileReader(filePath);
            br = new BufferedReader(fr);

            String line;
            line = br.readLine();
            numTest = Integer.parseInt(line);

            n = new int[numTest];

            for (int i=0; i<numTest; i++) {
                line = br.readLine();
                n[i] = Integer.parseInt(line);
            }

            int fordebug=0;
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

    private boolean isOkay(boolean []okay) {
        for (int i=0; i<10; i++) {
            if (!okay[i]) {
                return false;
            }
        }
        return true;
    }

    public void solve(int t) {
        long N = n[t];
        long current = 0;
        if (N==0) {
            System.out.println("Case #" + (t+1) + ": INSOMNIA");
            return;
        }
        boolean okay[] = new boolean[10];
        for (int i=0; i<10; i++) {
            okay[i] = false;
        }
        for (int i=0; i<999999; i++) {
            current += N;
            long tmp = current;
            while (tmp != 0) {
                okay[(int)(tmp%10)] = true;
                if (isOkay(okay)) {
                    System.out.println("Case #" + (t+1) + ": " + current);
                    return;
                }
                tmp = tmp / 10;
            }
        }
        System.out.println("Case #" + (t+1) + ": ****ERRRRRR**********");
        while(true){}
    }

    public void solveAll() {
        for (int i=0; i<numTest; i++) {
            solve(i);
        }
    }

    public void solveLargeTest() {
        for (int i=0; i<=100000000; i++) {
            n[0] = i;
            System.out.println("=== " + i);
            solve(0);
        }
    }

    public static void main(String []args) {
        Sheep obj = new Sheep();
        obj.fileRead("dat\\A-large.in");
        obj.solveAll();
    }
}
