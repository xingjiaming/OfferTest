package algorithm;

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
     * 02 快速排序
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
     * 03 归并排序
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
        // 这个地方有个性能优化，不是所有的数组都需要copy
        // 如果数组只赋值一半的区间，就好很多
//        for (int i = 0; i < k; i++) {
//            arrays[left + i] = temp[i];
//        }
        for (int i = left; i <= right; i++) {
            arrays[i] = temp[i];
        }
    }

    /**
     * 04 堆排序
     * <p>
     * 思想：
     * 1/ 把输入的树写成完全二叉树，给予非叶子节点，从右向左，从下向上左堆化
     * 2/ 把堆顶的元素和最后的元素进行交换
     * 3/ 对剩下的元素进行堆化
     * 堆化
     * 1/ 给予当前的index找最大值，如果最大值和堆顶需要交换位置
     * 2/ 因为交换位置后打乱了原有的顺序，所以需要把当前的index重新做堆化，依次递推
     */
    public void GetLeastNumbers_Solution(int[] input) {
        if (input == null || input.length == 1) {
            return;
        }
        //初始化创建一个堆
        //从右向左从下向上
        for (int i = (input.length / 2 + 1); i >= 0; i--) {
            heapify(input, i, input.length);
        }

        for (int i = 0; i < input.length; i++) {
            swap(input, 0, input.length - i - 1);
            heapify(input, 0, input.length - i - 1);
        }
    }

    /**
     * 堆化
     *
     * @param array 需要堆化的数组
     * @param index 需要堆化的节点
     */
    void heapify(int[] array, int index, int lenth) {
        int max = index;
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        //找最小值
        if (left < lenth/*防止下标越界*/ && array[left] > array[index]) max = left;
        if (right < lenth/*防止下标越界*/ && array[right] > array[index]) max = right;
        if (max != index) {
            //交换
            swap(array, index, max);
            //重新调整堆
            heapify(array, index, lenth);
        }
    }

    /**
     * 05 选择排序
     * <p>
     * 思想：
     * 每次遍历选择最大的值，然后把最大放在最后一个位置
     * <p>
     * 注意：
     * 最小值和最大值的标记，每次都需要更新
     */
    public void selectSort(int[] arrays) {
        int len = arrays.length;
        for (int i = 1; i < len; i++) {
            int max = arrays[0];
            int index = 0;
            for (int j = 0; j < len - i + 1; j++) {
                if (arrays[j] >= max) {
                    max = arrays[j];
                    index = j;
                }
            }
            swap(arrays, index, len - i);
        }
    }

    /**
     * 06 插入排序
     * <p>
     * 思想：
     * 前n个序列是有序队列，取n+1的元素，插入到前n个有序序列的有序位置
     * 遍历到最后一个元素结束
     */
    public void insertSort(int[] arrays) {
        int len = arrays.length;
        for (int i = 1; i < len; i++) {
            for (int j = i; j > 0 && arrays[j - 1] > arrays[j]; j--) {
                swap(arrays, j, j - 1);
            }
        }
    }
}
