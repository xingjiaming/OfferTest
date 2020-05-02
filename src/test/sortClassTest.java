package test;

import org.junit.Test;
import algorithm.sortClass;

import java.util.Arrays;

/**
 * sortClass Tester.
 *
 * @author xing
 * @version 1.0
 * @since 2020-04-30
 */
public class sortClassTest {
    sortClass mSortClass = new sortClass();

    @Test
    public void sortTest() throws Exception {
        int[] test = new int[]{80, 19, 4,4,4 ,3, 5, 999};
        mSortClass.insertSort(test);
        System.out.println(Arrays.toString(test));
    }

} 
