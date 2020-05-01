package testClass;

public class sortClass {
    public sortClass() {
    }

    /**
     * 01 冒泡排序
     * 思想：
     * 两两元素进行比较，每次遍历把最大或者最小值放到数组的数组的后面，保证后面是有序的
     * 关于遍历次数说明：
     * 1、遍历次数要小于lenth，长度是3的时候，遍历两次就行
     * 2、交换的元素需要从0开始，不能到最后一个下标，因为需要交换元素
     */
    public void bubbleSort(int[] arrays) {
        if (arrays == null || arrays.length == 1) {
            return;
        }
        int lenth = arrays.length;
        for (int i = 1; i < lenth; i++) {
            for (int j = 0; j < lenth - i; j++) {
                if (arrays[j] > arrays[j + 1]) {
                    swap(arrays, j, j + 1);
                }
            }
        }
    }

    void swap(int[] arrays, int start, int end) {
        int temp = arrays[start];
        arrays[start] = arrays[end];
        arrays[end] = temp;
    }

    /**
     * 快速排序
     * 思想：
     * 1/ 基于游标 i j 和参考标准 tmp 把数组调整到 左边小于tmp，右边大于tmp
     * 2/ 当i j相遇，说明已经排序完成，此时吧i和游标位置的数字交换即可
     * <p>
     * 说明:
     * 1/ 相遇时时，i和j是相等的，并且指向的是小于tmp的那一段
     * 2/ 油标右移等号是必须的，要不然第一个元素永远会被交换
     *
     * @param arrays
     */
    public void quickSort(int[] arrays) {
        if (arrays == null || arrays.length == 1) {
            return;
        }
        quickSort(arrays, 0, arrays.length - 1);
    }

    void quickSort(int[] arrays, int start, int end) {
        if (start >= end) {
            return;
        }
        int temp = arrays[start];
        int i = start;
        int j = end;
        while (i < j) {
            // j 是会永远指向小于temp的位置
            while (arrays[j] > temp && i < j) {
                j--;
            }
            // j和i相等的时候 i<j控制，所以不会遍历，所以循环结束
            // 这个等号是必须的，要不然第一个元素永远会被交换
            while (arrays[i] <= temp && i < j) {
                i++;
            }
            swap(arrays, i, j);
        }
        arrays[start] = arrays[i];
        arrays[i] = temp;
        quickSort(arrays, start, i - 1);
        quickSort(arrays, i + 1, end);
    }

    /**
     * 归并排序
     * 思想：
     * 1/ 把数组分割成一个一个小的序列，最后把一个一个小的序列归并到一起，保证合并后的序列是有序的
     * <p>
     * 步骤：
     * 1/ 归：将数组分成一个一个小的序列 --
     * 2/ 并：两个有序数组合并成一个有序的数组，分别把数组的最小值放进去，
     * 如果最后有剩余，把剩余放进去，类似合并有序链表
     * <p>
     * 说明：while循环将数组剩下的元素拷贝的目标的数组里
     */
    public void merge(int[] arrays) {
        if (arrays == null || arrays.length == 1) {
            return;
        }
        decompose(arrays, 0, arrays.length - 1);
    }

    private void decompose(int[] arrays, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = (start + end) / 2;
        decompose(arrays, start, mid);
        decompose(arrays, mid + 1, end);
        compose(arrays, start, mid, end);
    }

    private void compose(int[] arrays, int left, int mid, int right) {
        int leftstart = left;
        int leftEnd = mid;
        int rightstart = mid + 1;
        int rightEnd = right;
        int[] temp = new int[arrays.length];
        int k = left;
        while (leftstart <= leftEnd && rightstart <= rightEnd) {
            if (arrays[leftstart] < arrays[rightstart]) {
                temp[k++] = arrays[leftstart++];
            } else {
                temp[k++] = arrays[rightstart++];
            }
        }
        while (leftstart <= leftEnd) temp[k++] = arrays[leftstart++];
        while (rightstart <= rightEnd) temp[k++] = arrays[rightstart++];

        for (int i = left; i <= right; i++) {
            arrays[i] = temp[i];
        }
    }
}
