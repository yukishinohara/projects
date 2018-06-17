import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by yukis on 2016/04/09.
 */
public class Pancakes {

    int numTest = 0;
    String []inputs = null;

    public void fileRead(String filePath) {
        FileReader fr = null;
        BufferedReader br = null;
        try {
            fr = new FileReader(filePath);
            br = new BufferedReader(fr);

            String line;
            line = br.readLine();
            numTest = Integer.parseInt(line);

            inputs = new String[numTest];

            for (int i=0; i<numTest; i++) {
                line = br.readLine();
                inputs[i] = line;
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

    private ArrayList<Integer> trimLast1s(ArrayList<Integer> before) {
        int length = before.size();
        ArrayList<Integer> after = (ArrayList<Integer>)(before.clone());
        for (int i=length-1; i>=0; i--) {
            if (after.get(i).intValue() == 1) {
                after.remove(i);
            } else {
                break;
            }
        }
        return after;
    }

    private ArrayList<Integer> trimDuplicates(ArrayList<Integer> before) {
        int length = before.size();
        ArrayList<Integer> after = (ArrayList<Integer>)(before.clone());
        for (int i=0, last=2; i < after.size(); ) {
            if (after.get(i).intValue() == last) {
                after.remove(i);
            } else {
                last = after.get(i).intValue();
                i++;
            }
        }
        return after;
    }

    private ArrayList<Integer> flipAll(ArrayList<Integer> before) {
        int length = before.size();
        ArrayList<Integer> after = new ArrayList<Integer>();
        for (int i=length-1; i>=0; i--) {
            int x = Integer.valueOf(before.get(i).intValue());
            x = (x==0) ? 1 : 0;
            after.add(x);
        }
        return after;
    }

    private ArrayList<Integer> parse(String input) {
        int length = input.length();
        ArrayList<Integer> ans = new ArrayList<Integer>();
        for (int i=0; i<length; i++) {
            int x = (input.charAt(i) == '-') ? 0 : 1;
            ans.add(Integer.valueOf(x));
        }
        return ans;
    }

    public void testLocalFunc() {
        ArrayList<Integer> []t = new ArrayList[10];
        t[0] = trimLast1s(parse("+-+-+-++++++"));
        t[1] = trimLast1s(parse("------------"));
        t[2] = trimLast1s(parse("++++++++++++"));
        t[3] = trimDuplicates(parse("+++---++--++"));
        t[4] = trimDuplicates(parse("------------"));
        t[5] = trimDuplicates(parse("+-+-+-+-++--"));
        t[6] = flipAll(parse("+++---++--++"));
        t[7] = flipAll(parse("------------"));
        t[8] = flipAll(parse("+-+-+-+-++--"));
        int forDebug = 0;
    }

    public void solve(int t) {
        ArrayList<Integer> S = parse(inputs[t]);
        int flips = 0;
        while(true) {
            S = trimLast1s(S);
            if (S.size() == 0) {
                break;
            }
            S = trimDuplicates(S);
            if (S.get(0).intValue() == 1) {
                S.set(0, Integer.valueOf(0));
                flips++;
            }
            S = flipAll(S);
            flips++;
        }
        System.out.println("Case #" + (t+1) + ": " + flips);
    }

    public void solveAll() {
        for (int t=0; t<numTest; t++) {
            solve(t);
        }
    }

    public static void main(String []args) {
        Pancakes obj = new Pancakes();
        obj.fileRead("dat\\B-large.in");
        obj.solveAll();
    }
}
