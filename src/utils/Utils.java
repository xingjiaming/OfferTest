package utils;

import Bean.ListNode;

import java.util.List;

public class Utils {
    public static void printListString(List<?> input) {
        for (Object tmpString : input) {
            System.out.print(tmpString + " ");
        }
        System.out.println();
    }
}
