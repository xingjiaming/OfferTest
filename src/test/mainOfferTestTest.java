package test;

import Bean.ListNode;
import Bean.TreeNode;
import org.junit.Test;
import org.junit.Before;
import org.junit.After;
import algorithm.offerTest;
import utils.Utils;
import utils.UtilsListNode;
import utils.UtilsTree;

import java.util.ArrayList;
import java.util.Arrays;

import static utils.Utils.printListString;

/**
 * Main Tester.
 *
 * @author 邢家明
 * @version 1.0
 * @since 2020-04-25
 */
public class mainOfferTestTest {
    offerTest mOffer;

    @Before
    public void before() throws Exception {
        mOffer = new offerTest();
    }

    @After
    public void after() throws Exception {
    }

    /**
     * Method: main(String[] args)
     */
    @Test
    public void testMain() throws Exception {
    }

    /**
     * Method: print()
     */
    @Test
    public void testReplaceSpace() throws Exception {
        String result = mOffer.replaceSpace(new StringBuffer(" dd"));
        System.out.println(result);
    }

    @Test
    public void testFind() throws Exception {
//        int[][] arry = new int[][]{{1, 2, 8, 9}, {2, 4, 9, 12}, {4, 7, 10, 13}, {6, 8, 11, 15}};
//        int[] arry = new int[]{1, 2, 3};
        int[][] arry = new int[][]{{}, {}};
        System.out.println(mOffer.Find(5, arry));
    }

    @Test
    public void testbinarySearch() throws Exception {
//        int[][] arry = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        int[] arry = new int[]{1, 2, 3};
        System.out.println(mOffer.binarySearch(3, arry));
    }

    @Test
    public void testAboutList() throws Exception {
        ListNode listNode = UtilsListNode.createList(new int[]{1, 2, 3, 4, 5}, UtilsListNode.CREAT_LIST_BEGIN);
        UtilsListNode.printList(listNode);
    }

    @Test
    public void testAboutTree() throws Exception {
        TreeNode treeNode = UtilsTree.createTree("AB#D##C##".toCharArray());
//        UtilsTree.printTreeQ(treeNode);
//        UtilsTree.printTreeZ(treeNode);
//        UtilsTree.printTreeH(treeNode);
        UtilsTree.show(treeNode);
    }

    @Test
    public void reConstructBinaryTree() throws Exception {
        int[] pre = new int[]{1, 2, 4, 7, 3, 5, 6, 8};
        int[] in = new int[]{4, 7, 2, 1, 5, 3, 8, 6};
        TreeNode treeNode = mOffer.reConstructBinaryTree(pre, in);
        UtilsTree.show(treeNode);
    }

    @Test
    public void minNumberInRotateArray() throws Exception {
        System.out.println(mOffer.minNumberInRotateArray(new int[]{3, 4, 5, 1, 2}));
    }

    @Test
    public void Fibonacci() throws Exception {
        System.out.println(mOffer.Fibonacci(3));
    }

    @Test
    public void reOrderArray() throws Exception {
        int[] a = new int[]{1, 2, 3, 4, 5, 6};
        mOffer.reOrderArray(a);
        for (int i : a) {
            System.out.println(i);
        }
    }

    @Test
    public void Merge() throws Exception {
        ListNode listNode1 = UtilsListNode.createList(new int[]{2, 4, 5}, UtilsListNode.CREAT_LIST_END);
        ListNode listNode2 = UtilsListNode.createList(new int[]{1, 3, 8}, UtilsListNode.CREAT_LIST_END);
        UtilsListNode.printList(mOffer.Merge(listNode1, listNode2));
    }

    @Test
    public void HasSubtree() throws Exception {
        TreeNode treeNode = UtilsTree.createTree(new char[]{'1', '2', '4', '#', '#', '5',
                '#', '#', '3', '6', '#', '#', '7', '#', '#'});
        TreeNode treeNode1 = UtilsTree.createTree(new char[]{'2', '4', '#', '#', '5', '#', '#'});
        UtilsTree.show(treeNode);
        UtilsTree.show(treeNode1);
        System.out.println(mOffer.HasSubtree(treeNode, treeNode1));
    }

    @Test
    public void Mirror() throws Exception {
        TreeNode treeNode = UtilsTree.createTree(new char[]{'1', '2', '4', '#', '#', '5',
                '#', '#', '3', '6', '#', '#', '7', '#', '#'});
        UtilsTree.show(treeNode);
        mOffer.Mirror(treeNode);
        UtilsTree.show(treeNode);
    }

    @Test
    public void printMatrix() throws Exception {
//        int[][] t = new int[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
        int[][] t = new int[][]{{1}, {2}, {3}, {4}};
        ArrayList<Integer> integerArrayList = mOffer.printMatrix(t);
        printListString(integerArrayList);
    }

    @Test
    public void IsPopOrder() throws Exception {
        int[] input = new int[]{1, 2, 3, 4, 5};
        int[] res = new int[]{1, 2, 3, 4, 5};
        boolean result = mOffer.IsPopOrder(input, res);
        System.out.println(result);
    }

    @Test
    public void PrintFromTopToBottom() throws Exception {
        TreeNode treeNode = UtilsTree.createTree(new char[]{'1', '2', '4', '#', '#', '5',
                '#', '#', '3', '6', '#', '#', '7', '#', '#'});
        UtilsTree.show(treeNode);
        Utils.printListString(mOffer.PrintFromTopToBottom(treeNode));
        System.out.println(mOffer.TreeDepth(treeNode));
    }

    @Test
    public void VerifySquenceOfBST() throws Exception {
        System.out.println(mOffer.VerifySquenceOfBST(new int[]{1, 1, 1, 1}));
    }

    @Test
    public void FindPath() throws Exception {
        TreeNode treeNode = UtilsTree.createTree(new char[]{'4', '2', '1', '#', '#', '3',
                '#', '#', '5', '#', '#'});
        UtilsTree.show(treeNode);
        ArrayList<ArrayList<Integer>> temp = mOffer.FindPath(treeNode, 152);
//        System.out.print(temp.size() + "  ");
        for (ArrayList<Integer> i : temp) {
            for (Integer j : i) {
                System.out.print(j + "  ");
            }
            System.out.println("  ");
        }
    }

    @Test
    public void Convert() throws Exception {
        TreeNode treeNode = UtilsTree.createTree(new char[]{'4', '2', '1', '#', '#', '3',
                '#', '#', '5', '#', '#'});
        UtilsTree.show(treeNode);
        mOffer.Convert(treeNode);
    }

    @Test
    public void GetLeastNumbers_Solution() throws Exception {
        Utils.printListString(mOffer.GetLeastNumbers_Solution(new int[]{4, 5, 1, 6, 2, 7, 3, 8}, 0));
    }

    @Test
    public void NumberOf1Between1AndN_Solution() throws Exception {
        System.out.println(mOffer.NumberOf1Between1AndN_Solution(1));
    }

    @Test
    public void PrintMinNumber() throws Exception {
        System.out.println(mOffer.PrintMinNumber(new int[]{3, 32, 321}));
        System.out.println(("acb").compareTo("aca"));
    }

    @Test
    public void GetUglyNumber_Solution() throws Exception {
        System.out.println(mOffer.GetUglyNumber_Solution(11));
    }

    @Test
    public void FirstNotRepeatingChar() throws Exception {
        System.out.println(mOffer.FirstNotRepeatingChar("NXWtnzyoHoBhUJaPauJaAitLWNMlkKwDYbbigdMMaYfkVPhGZcrEwp"));
    }

    @Test
    public void InversePairs() throws Exception {
        System.out.println(mOffer.InversePairs(new int[]{1, 2, 3, 4, 5, 6, 7, 0}));
    }

    @Test
    public void GetNumberOfK() throws Exception {
        System.out.println(mOffer.GetNumberOfK(new int[]{3}, 3));
    }

    @Test
    public void FindNumsAppearOnce() throws Exception {
        int[] arrays = new int[]{2, 4, 3, 6, 3, 2, 5, 5};
        int[] num2 = new int[arrays.length - 2];
        int[] num1 = new int[2];
        mOffer.FindNumsAppearOnce(arrays, num1, num2);
        System.out.println(Arrays.toString(num1));
    }

    @Test
    public void FindNumbersWithSum() throws Exception {
        Utils.printListString(mOffer.FindNumbersWithSum(new int[]{1, 2, 3, 4, 5}, 6));
    }

    @Test
    public void ReverseSentence() throws Exception {
        System.out.println(mOffer.ReverseSentence(" "));
    }

    @Test
    public void isSymmetrical() throws Exception {
        TreeNode treeNode = UtilsTree.createTree(new char[]{'4', '2', '1', '#', '#', '1', '#', '#', '2', '#', '#'});
        UtilsTree.show(treeNode);
        System.out.println(mOffer.isSymmetric(treeNode));
    }

    @Test
    public void LastRemaining_Solution() throws Exception {
        System.out.println(mOffer.LastRemaining_Solution(5, 3));
    }

    @Test
    public void deleteDuplication() throws Exception {
        UtilsListNode.printList(mOffer.deleteDuplication(UtilsListNode.createList(new int[]{1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5}
                , UtilsListNode.CREAT_LIST_END)));
    }

    @Test
    public void Add() throws Exception {
        System.out.println(mOffer.Add(5, 3));
    }
} 
