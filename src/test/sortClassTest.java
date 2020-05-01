package test;

import org.junit.Test;
import org.junit.Before;
import org.junit.After;
import testClass.sortClass;

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
        int[] test = new int[]{80, 19, 4, 3, 5, 999};
        mSortClass.merge(test);
        System.out.println(Arrays.toString(test));
    }

} 
