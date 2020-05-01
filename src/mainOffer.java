import Bean.ListNode;

import java.util.Arrays;

public class mainOffer {
    public static void main(String[] args) {
//        int[] a = new int[]{1, 3, 4, 5, 5};
//        int[] b = Arrays.copyOfRange(a, 2, 6);
//        for (int i : b) {
//            System.out.println(i);
//        }
        int k = 2;
        int[] a = new int[]{4, 7, 2};
        System.out.print(Arrays.binarySearch(a, k));
    }
}
