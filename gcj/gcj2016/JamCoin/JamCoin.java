import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.BitSet;

/**
 * Created by yukis on 2016/04/09.
 */
public class JamCoin {

    private ArrayList<String> foundCoins = null;
    private ArrayList<String[]> foundFactors = null;

    // From: http://faruk.akgul.org/blog/javas-missing-algorithm-biginteger-sqrt/
    private BigInteger sqrt(BigInteger n) {
        BigInteger a = BigInteger.ONE;
        BigInteger b = new BigInteger(n.shiftRight(5).add(new BigInteger("8")).toString());
        while(b.compareTo(a) >= 0) {
            BigInteger mid = new BigInteger(a.add(b).shiftRight(1).toString());
            if(mid.multiply(mid).compareTo(n) > 0) b = mid.subtract(BigInteger.ONE);
            else a = mid.add(BigInteger.ONE);
        }
        return a.subtract(BigInteger.ONE);
    }

    public String makeBitString(BigInteger bits) {
        String ans = "";
        BigInteger val = bits;
        while(!val.equals(BigInteger.ZERO)) {
            ans = "" + (((val.and(BigInteger.ONE)).equals(BigInteger.ONE)) ? "1" : "0") + ans;
            val = val.shiftRight(1);
        }
        return ans;
    }

    public void listAllJamCoin(int N) {
        BigInteger MIN = BigInteger.ONE.shiftLeft(N-1).add(BigInteger.ONE);
        BigInteger MAX = BigInteger.ONE.shiftLeft(N).subtract(BigInteger.ONE);
        foundCoins = new ArrayList<String>();
        foundFactors = new ArrayList<String[]>();
        for (BigInteger i = MIN; i.compareTo(MAX) <= 0; i = i.add(new BigInteger("2"))) {
            String coin = makeBitString(i);
            String []factors = getFactorsOrNull(coin);
            if (factors == null) {
                continue;
            }
            if (!isJamCoin(coin, factors)) {
                continue;
            }
            foundCoins.add(coin);
            foundFactors.add(factors);
        }
        int debug=0;
    }

    public void printAllJamCoin() {
        int num = foundCoins.size();
        for (int i=0; i<num; i++) {
            System.out.print(foundCoins.get(i));
            for (int j=0; j<9; j++) {
                System.out.print(" " + foundFactors.get(i)[j]);
            }
            System.out.println();
        }
    }

    public String[] getFactorsOrNull(String coin) {
        int length = coin.length();
        BigInteger[] K = new BigInteger[10];
        BigInteger[] F = new BigInteger[10];
        String [] ans = new String[9];
        for (int i=0; i<10; i++) {
            K[i] = BigInteger.ZERO;
            F[i] = BigInteger.valueOf(i+1);
        }
        for (int i=0; i<length; i++) {
            for (int j=1; j<10; j++) {
                BigInteger keta = (coin.charAt(i) == '1') ? BigInteger.ONE : BigInteger.ZERO;
                K[j] = K[j].multiply(F[j]).add(keta);
            }
        }
        for (int j=1; j<10; j++) {
            if (K[j].isProbablePrime(1000)) {
                return null;
            }
            ans[j-1] = "0";
            BigInteger root = sqrt(K[j]).add(BigInteger.TEN);
            for (BigInteger i = new BigInteger("2"); i.compareTo(root) <= 0; i = i.add(BigInteger.ONE)) {
                if (K[j].remainder(i).equals(BigInteger.ZERO) && !K[j].equals(i)) {
                    ans[j-1] = i.toString();
                    break;
                }
            }
        }
        return ans;
    }

    public boolean isJamCoin(String coin, String[] factors) {
        int length = coin.length();
        BigInteger[] K = new BigInteger[10];
        BigInteger[] F = new BigInteger[10];
        for (int i=0; i<10; i++) {
            K[i] = BigInteger.ZERO;
            F[i] = BigInteger.valueOf(i+1);
        }
        for (int i=0; i<length; i++) {
            for (int j=1; j<10; j++) {
                BigInteger keta = (coin.charAt(i) == '1') ? BigInteger.ONE : BigInteger.ZERO;
                K[j] = K[j].multiply(F[j]).add(keta);
            }
        }
        int forDebug=0;
        for (int j=1; j<10; j++) {
            if (K[j].remainder(new BigInteger(factors[j-1])).equals(BigInteger.ZERO)) {
                continue;
            }
            return false;
        }
        return true;
    }

    public void testInputFile(String filePath) {
        FileReader fr = null;
        BufferedReader br = null;
        try {
            fr = new FileReader(filePath);
            br = new BufferedReader(fr);

            String line;
            line = br.readLine();

            while(true) {
                line = br.readLine();
                if(line == null) {
                    break;
                }
                String [] line2 = line.split(" ");
                String [] factors = new String[9];
                for (int j=1; j<10; j++) {
                    factors[j-1] = line2[j];
                }
                System.out.println("" + line2[0] + ": " + isJamCoin(line2[0], factors));
            }
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

    public void testLocalFunc() {
        System.out.println(makeBitString(BigInteger.valueOf(0)));
        System.out.println(makeBitString(BigInteger.valueOf(1)));
        System.out.println(makeBitString(BigInteger.valueOf(2)));
        System.out.println(makeBitString(BigInteger.valueOf(3)));
        System.out.println(makeBitString(BigInteger.valueOf(4)));
        System.out.println(makeBitString(BigInteger.valueOf(100)));
        System.out.println(makeBitString(BigInteger.valueOf(1024)));
        System.out.println(makeBitString(new BigInteger("4294967295")));
        System.out.println(makeBitString(new BigInteger("4294967296")));
        System.out.println(""+isJamCoin("100011", new String[]{
                "5", "13", "147", "31", "43", "1121", "73", "77", "629"
        }));
        System.out.println(""+isJamCoin("100011", new String[]{
                "2", "13", "147", "31", "43", "1121", "73", "77", "629"
        }));
        /*
        System.out.println(""+isJamCoin("100011", new String[]{
                "", "", "", "", "", "", "", "", ""
        }));*/
    }

    public static void main(String []args) {
        JamCoin obj = new JamCoin();
        obj.testInputFile("dat\\C-large.out");
    }
}
