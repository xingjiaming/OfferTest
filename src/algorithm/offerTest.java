package algorithm;

import Bean.ListNode;
import Bean.RandomListNode;
import Bean.TreeNode;

import java.util.*;

public class offerTest {
    /**
     * 01 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，
     * 输入这样的一个二维数组和一个整数，判断数组中是否含有该整数
     * <p>
     * 思路1：遍历每一行，找到第一个大于目标值，剪纸，然后进行剩下条件的二分查找
     * 思路2：左下角元素m是行中最小的，是一列中最大的。
     * 当m == target时，查到结果，直接返回；
     * 当m > target时，因为m是一行中最小的，所以向上移动一行，继续查找；
     * 当m < target时，因为m是一列中最大的，所以向右移动一列，继续查找。
     *
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int[][] array) {
        int lenRow = array.length;
        int lenCol = array[0].length;
        if (lenCol <= 0) {
            return false;
        }
        if (target > array[lenRow - 1][lenCol - 1] || target < array[0][0]) {
            return false;
        }
        int col = 0;
        int raw = lenRow - 1;
        while (raw >= 0 && col < lenCol) {
            if (target > array[raw][col]) {
                col++;
            } else if (target < array[raw][col]) {
                raw--;
            } else {
                return true;
            }
        }
//        for (int i = 0; i < lenRow; i++) {
//            if (target < array[i][0]) {
//                continue;
//            }
//            if (Arrays.binarySearch(array[i], target) > 0) {
//                return true;
//            }
//        }
        return false;
    }

    /**
     * 附加：二分查找
     *
     * @param target
     * @param array
     * @return
     */
    public boolean binarySearch(int target, int[] array) {
        int lenRow = array.length;
        int low = 0;
        int hight = lenRow - 1;
        int mid = (hight + low) / 2;
        //必须用等于号，要不然，low和mid和high在一起的时候，会不执行
        while (low <= hight) {
            if (target > array[mid]) {
                low = mid + 1;
            } else if (target < array[mid]) {
                hight = mid - 1;
            } else {
                return true;
            }
            mid = (hight + low) / 2;
        }
        return false;
    }

    /**
     * 02 将一个字符串中的每个空格替换成“%20”
     *
     * @param str
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        return str.toString().replace(" ", "%20");
    }

    /**
     * 03 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
     * <p>
     * peek 不改变栈值
     * pop 会出栈
     *
     * @param listNode
     * @return
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> list = new ArrayList<>();
        if (listNode == null) {
            return list;
        }
        ListNode listTemp = listNode;
        Stack<Integer> stack = new Stack<>();
        while (listTemp != null) {
            stack.push(listTemp.val);
            listTemp = listTemp.next;
        }
        while (!stack.empty()) {
            list.add(stack.peek());
        }
        return list;
    }

    /**
     * 04 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     * 例如
     * 输入前序遍历序列{1,2,4,7,3,5,6,8}
     * 中序遍历序列{4,7,2,1,5,3,8,6}
     * 重建二叉树并返回。
     * <p>
     * 二分查找一定是有序的！！！
     *
     * @param pre
     * @param in
     * @return
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length == 0 || in.length == 0) {
            return null;
        }
        int index = -1;
        for (int i = 0; i < in.length; i++) {
            if (in[i] == pre[0]) {
                index = i;
                break;
            }
        }
        if (index < 0) {
            return null;
        }
        TreeNode root = new TreeNode(pre[0]);
        root.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, index + 1), Arrays.copyOfRange(in, 0, index));
        root.right = reConstructBinaryTree(Arrays.copyOfRange(pre, index + 1, pre.length), Arrays.copyOfRange(in, index + 1, in.length));
        return root;
    }

    /**
     * 05 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型
     */
    Stack<Integer> stack1 = new Stack<Integer>();
    //stack2是一个中间的缓存，如果有值要优先弹它
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (stack2.empty()) {
            while (!stack1.empty()) {
                stack2.push(stack1.pop());
            }
        }
        return stack2.peek();
    }

    /**
     * 06 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
     * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
     * NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
     */
    public int minNumberInRotateArray(int[] array) {
//        直接遍历就行
//        int min = array[0];
//        for (int i = 0; i < array.length; i++) {
//            if (array[i] <= min) {
//                min = array[i];
//            }
//        }
//        return min;
        /**
         * 二分的精髓：
         * 终止条件：
         * 如果a[min]>a[min+1]或者a[min-1]>a[min]
         * 上下的条件：
         * 大于其实位置，+，小于其实位置，-
         */
        int len = array.length;
        if (len == 0) {
            return -1;
        }
        int low = 0;
        int hight = len - 1;
        int mid = 0;
        while (low <= hight) {
            mid = (low + hight) / 2;
            if (array[mid] > array[mid + 1]) {
                return array[mid + 1];
            }
            if (array[mid - 1] > array[mid]) {
                return array[mid];
            }
            if (array[mid] > array[0]) {
                low = mid + 1;
            } else {
                hight = mid - 1;
            }
        }
        return -1;
    }

    /**
     * 07 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
     * n<=39
     * fn = fn-1 + fn-2
     *
     * @param n
     * @return
     */
    public int Fibonacci(int n) {
        if (n > 39) {
            return 0;
        }
        if (n == 0) return 0;
        if (n == 1) return 1;
        return Fibonacci(n - 1) + Fibonacci(n - 2);
    }

    /**
     * 08 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
     * 斐波那契数列 变形
     */
    public int JumpFloor(int target) {
        if (target == 0) return 1;
        if (target == 1) return 1;
        return JumpFloor(target - 1) + JumpFloor(target - 2);
    }

    /**
     * 09 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法
     * 斐波那契数列 变形
     */
    public int JumpFloorII(int target) {
        if (target == 0) return 1;
        if (target == 1) return 1;
        return JumpFloorII(target - 1) * 2;
    }

    /**
     * 10 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
     */
    public int RectCover(int target) {
        if (target <= 0) {
            return 0;
        }
        if (target == 1)
            return 1;
        if (target == 2)
            return 2;

        return RectCover(target - 1) + RectCover(target - 2);
    }

    /**
     * 11 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
     * <p>
     * 思路：
     * <p>
     * 计算机中，负数用补码表示。
     * 如负数 -7
     * 符号位  数值位
     * 1      0000111   即-7的原码为（1000 0111）
     * 反码1 1111000 （负数的反码与原码符号位相同，数值为取反）
     * 补码1 1111001 （负数的补码是在反码的基础上加1）
     * <p>
     * 一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，它后面的两位0变成了1，而前面的1保持不变，因此得到的结果是1011。
     * 我们发现减1的结果是把最右边的一个1开始的所有位都取反了。这个时候如果我们再把原来的整数和减去1之后的结果做与运算，
     * 从原来整数最右边一个1那一位开始所有位都会变成0。如1100&1011=1000.也就是说，把一个整数减去1，再和原整数做与运算，
     * 会把该整数最右边一个1变成0。那么一个整数的二进制有多少个1，就可以进行多少次这样的操作。
     * ————————————————
     * 版权声明：本文为CSDN博主「wanqian_2019」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
     * 原文链接：https://blog.csdn.net/qq_28632639/article/details/87966115
     *
     * @param n
     * @return
     */
    public int NumberOf1(int n) {
        int result = 0;
        while (n != 0) {
            result++;
            n = n & (n - 1);
        }
        return result;
    }

    /**
     * 12 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
     *
     * @param base
     * @param exponent
     * @return
     */
    public double Power(double base, int exponent) {
        if (Math.abs(base) < 0.00001)
            return 0.0;
        if (exponent == 0)
            return 1.0;
        double result = base;
        if (exponent > 0) {
            for (int i = 1; i < exponent; i++) {
                result *= base;
            }
        }
        if (exponent < 0) {
            for (int i = 1; i < Math.abs(exponent); i++) {
                result *= base;
            }
            result = 1 / result;
        }
        return result;
    }

    /**
     * 13 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     */
    public void reOrderArray(int[] array) {
        if (array == null || array.length == 0) {
            return;
        }
        ArrayList<Integer> tempJ = new ArrayList<>();
        ArrayList<Integer> tempO = new ArrayList<>();
        for (int i : array) {
            if (i % 2 == 0) {
                tempO.add(i);
            } else {
                tempJ.add(i);
            }
        }
        tempJ.addAll(tempO);
        for (int j = 0; j < tempJ.size(); j++) {
            array[j] = tempJ.get(j);
        }
// 需要保持顺序
//        int left = 0;
//        int right = array.length - 1;
//        while (left < right) {
//            while (left < right && array[left] % 2 != 0) {
//                left++;
//            }
//            while (left < right && array[right] % 2 == 0) {
//                right--;
//            }
//            if (left < right) {
//                int temp = array[left];
//                array[left] = array[right];
//                array[right] = temp;
//            }
//        }
    }

    /**
     * 14 输入一个链表，输出该链表中倒数第k个结点。
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null)
            return null;
        ListNode headTemp = head;
        for (int i = 0; i < k; i++) {
            if (headTemp == null) {
                return null;
            }
            headTemp = headTemp.next;
        }
        while (headTemp != null) {
            headTemp = headTemp.next;
            head = head.next;
        }
        return head;
    }

    /**
     * 15 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null && list2 == null)
            return null;
        if (list1 == null)
            return list2;
        if (list2 == null)
            return list1;
        ListNode listNode = new ListNode(0);
        if (list2.val > list1.val) {
            listNode.val = list1.val;
            list1 = list1.next;
        } else {
            listNode.val = list2.val;
            list2 = list2.next;
        }
        listNode.next = Merge(list1, list2);
        return listNode;
    }

    /**
     * 16 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
     *
     * @param root1 总
     * @param root2 子
     * @return
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return false;
        }
        if (isSubTree(root1, root2)) {
            return true;
        }
        return HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }

    private boolean isSubTree(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null) {
            return false;
        }
        if (root2 == null) {
            return true;
        }
        if (root1.val == root2.val) {
            return isSubTree(root1.left, root2.left) && isSubTree(root1.right, root2.right);
        } else {
            return false;
        }
    }

    /**
     * 17 操作给定的二叉树，将其变换为源二叉树的镜像。
     */
    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode tmp = new TreeNode(0);
        tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        Mirror(root.left);
        Mirror(root.right);
    }

    /**
     * 18 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵：
     * 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
     * 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
     *
     * @param matrix
     * @return
     */
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        int rowLen = matrix.length;
        int colLen = matrix[0].length;
        int r1 = 0, c1 = 0;
        int r2 = rowLen - 1;
        int c2 = colLen - 1;
        ArrayList<Integer> res = new ArrayList<>();
        addInList(res, r1, c1, r2, c2, matrix);
        return res;
    }

    /**
     * 19 顺时针打印矩阵
     *
     * @param res
     * @param r1
     * @param c1
     * @param r2
     * @param c2
     * @param matrix
     */
    private void addInList(ArrayList<Integer> res,
                           int r1, int c1,
                           int r2, int c2,
                           int[][] matrix) {
        if (c1 > c2 || r1 > r2) {
            return;
        }
        int r = r1, c = c1;
        while (c < c2) {
            // →
            res.add(matrix[r][c]);
            c++;
        }
        while (r < r2) {
            // ↓
            res.add(matrix[r][c]);
            r++;
        }
        // 如果单列，单行，需要考虑去重的逻辑！
        if (c1 == c2 || r1 == r2) {
            res.add(matrix[r][c]);
            return;
        }
        while (c > c1) {
            res.add(matrix[r][c]);
            c--;
        }
        while (r > r1) {
            res.add(matrix[r][c]);
            r--;
        }
        addInList(res, r1 + 1, c1 + 1, r2 - 1, c2 - 1, matrix);
    }

    /**
     * 20 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
     * 所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     *
     * @param node
     */
//    Stack<Integer> stack1 = new Stack<>();
//    Stack<Integer> stack2 = new Stack<>();
//
//    public void push1(int node) {
//        stack1.push(node);
//        if (stack2.isEmpty()) {
//            stack2.push(node);
//        } else {
//            if (stack2.peek() >= node) {
//                stack2.push(node);
//            }
//        }
//    }
//
//    public void pop1() {
//        if (stack1.pop() == stack2.peek()) {
//            stack2.pop();
//        }
//        stack1.pop();
//    }
//
//    public int top1() {
//        return stack1.peek();
//    }
//
//    public int min1() {
//        return stack2.peek();
//    }

    /**
     * 21 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
     * 例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，
     * 但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
     *
     * @param pushA
     * @param popA
     * @return
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        int lenPushA = pushA.length;
        int lenPopA = popA.length;
        Stack<Integer> stack = new Stack<>();
        int indexPop = 0;
        // push
        int i = 0;
        for (; i < lenPushA; i++) {
            stack.push(pushA[i]);
            if (pushA[i] == popA[indexPop]) {
                if (!stack.isEmpty()) {
                    stack.pop();
                    indexPop++;
                }
            }
        }

        while (!stack.isEmpty() && (stack.peek() == popA[indexPop])) {
            indexPop++;
            stack.pop();
        }
        return stack.isEmpty();
    }

    /**
     * 22 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
     * <p>
     * 思想：
     * 二叉树的层次遍历，这里采用队列的方式
     *
     * @param root
     * @return
     */
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> arrayList = new ArrayList<>();
        if (root == null) {
            return arrayList;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();// 表示一层有多少个点
            while (size != 0) {
                TreeNode treeNode = queue.poll();
                arrayList.add(treeNode.val);
                if (treeNode.left != null) {
                    queue.offer(treeNode.left);
                }
                if (treeNode.right != null) {
                    queue.offer(treeNode.right);
                }
                size--;// 每次遍历结束，这一行减一
            }
        }
        return arrayList;
    }

    /**
     * 23 输入一个非空整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
     * 如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
     *
     * @param sequence
     * @return
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        int len = sequence.length;
        int root = sequence[len - 1];
        int i = 0;
        for (; i < len - 1; i++) {
            if (sequence[i] > root) break;
        }
        for (int j = i; j < len - 1; j++) {
            if (sequence[j] < root) return false;
        }
        // 递归的终止条件，如果substring长度是1或者0的时候，返回true
        if (Math.abs(i - 0) == 1 || Math.abs(i - 0) == 0 ||
                Math.abs(i - len) == 1 || Math.abs(i - len) == 0) {
            return true;
        }
        return VerifySquenceOfBST(Arrays.copyOfRange(sequence, 0, i)) && VerifySquenceOfBST(Arrays.copyOfRange(sequence, i, len - 1));
    }

    /**
     * 24 输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
     * 路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
     *
     * @param root
     * @param target
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        if (root == null) {
            return res;
        }
        FindPathRes(root, target, res, temp);
        return res;
    }

    void FindPathRes(TreeNode root, int target, ArrayList<ArrayList<Integer>> res, ArrayList<Integer> temp) {
        if (root == null) {
            return;
        }
        int c = target - root.val;
        // 看好题目，这个地方一定是叶子节点
        if (c == 0 && root.left == null && root.right == null) {
            temp.add(root.val);
            res.add(new ArrayList<>(temp));
            temp.remove(temp.size() - 1);
        }
        // 如果小于0的话，跳过这个分支就行了
        if (c > 0) {
            temp.add(root.val);
            FindPathRes(root.left, c, res, temp);
            FindPathRes(root.right, c, res, temp);
            temp.remove(temp.size() - 1);
        }
    }

    /**
     * 25 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针random指向一个随机节点），
     * 请对此链表进行深拷贝，并返回拷贝后的头结点。（注意，输出结果中请不要返回参数中的节点引用，
     * 否则判题程序会直接返回空）
     *
     * @param pHead
     * @return
     */
    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) {
            return null;
        }
        RandomListNode res = new RandomListNode(pHead.label);
        RandomListNode tail = res;
        clone(pHead, tail);
        return res;
    }

    void clone(RandomListNode pHead, RandomListNode tail) {
        if (pHead == null) {
            return;
        }
        tail.label = pHead.label;
        if (pHead.next == null) {
            tail.next = null;
        } else {
            tail.next = new RandomListNode(pHead.next.label);
        }
        if (pHead.random == null) {
            tail.random = null;
        } else {
            tail.random = new RandomListNode(pHead.random.label);
        }
        clone(pHead.next, tail.next);
    }

    /**
     * 26 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
     * <p>
     * 利用搜索二叉树的排序特效
     * <p>
     * pre表示链表的前驱节点，cur表示当前的遍历节点
     * 先遍历左
     * 再进行连接
     * 再遍历右
     * <p>
     * 注意迭代的终止条件，注意保持根节点，注意函数调用的时候传值的这个问题！！！！！，pre如果作为形参，是传值
     */
    TreeNode res = null;
    TreeNode pre = null;

    public TreeNode Convert(TreeNode pRootOfTree) {
        TreeNode cur = pRootOfTree;
        dfs(cur);
        return res;
    }

    private void dfs(TreeNode cur) {
        if (cur == null) {
            return;
        }
        dfs(cur.left);
        if (res == null) {
            res = cur;
        } else {
            cur.left = pre;
            pre.right = cur;
        }
        pre = cur;
        dfs(cur.right);
    }

    /**
     * 27 输入一个字符串,按字典序打印出该字符串中字符的所有排列。
     * 例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
     * <p>
     * 字典序
     * 1/ 从右向左找到第一个小于右边的数的位置 a
     * 2/ 找到右边向左第一个大于a位置数的位置 b
     * 3/ 换a和b的位置
     * 4/ a后面的位置排序
     */
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> res = new ArrayList<>();
        if (str == null || str.length() == 0) {
            return res;
        }
        char[] temp = str.toCharArray();
        Arrays.sort(temp);

        res.add(new String(temp));

        int len = str.length();

        boolean isH = true;

        while (isH) {
            int indexA = 0;
            int indexB = 0;

            isH = false;

            for (int i = len - 2; i >= 0; i--) {
                if (temp[i] < temp[i + 1]) {
                    indexA = i;
                    isH = true;
                    break;
                }
            }

            if (isH) {
                for (int i = len - 1; i >= 0; i--) {
                    if (temp[i] > temp[indexA]) {
                        indexB = i;
                        break;
                    }
                }

                char tempChar = temp[indexA];
                temp[indexA] = temp[indexB];
                temp[indexB] = tempChar;

                sortArray(temp, indexA + 1, len);
                res.add(new String(temp));
            }
        }
        return res;
    }

    private char[] sortArray(char[] temp, int indexA, int len) {
        for (int i = indexA; i < len; i++) {
            for (int j = indexA; j < len - (i - indexA) - 1; j++) {
                if (temp[j] > temp[j + 1]) {
                    char t = temp[j];
                    temp[j] = temp[j + 1];
                    temp[j + 1] = t;
                }
            }
        }
        return temp;
    }

    /**
     * 28数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
     * 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
     * <p>
     * 思路是用：hash
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        int count = array.length / 2;
        if (count == 0) {
            return array[0];
        }
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i : array) {
            if (hashMap.containsKey(i)) {
                int t = hashMap.get(i);
                if (t + 1 <= count) {
                    hashMap.put(i, t + 1);
                } else {
                    return i;
                }
            } else {
                hashMap.put(i, 1);
            }
        }
        return 0;
    }

    /**
     * 29输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> res = new ArrayList<>();
        if (k == 0 || k > input.length) {
            return res;
        }
        //初始化创建一个堆
        for (int i = (input.length / 2 - 1); i >= 0; i--) {
            heapify(input, i, input.length);
        }

        for (int i = 0; i < k; i++) {
            res.add(input[0]);
            sweap(input, 0, input.length - i - 1);
            heapify(input, 0, input.length - i - 1);
        }
        return res;
    }

    void sweap(int[] array, int start, int end) {
        int tmp = array[start];
        array[start] = array[end];
        array[end] = tmp;
    }

    /**
     * 堆化
     *
     * @param array,需要堆化的数组
     * @param index         需要堆化的节点
     */
    void heapify(int[] array, int index, int lenth) {
        int min = index;
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        //找最小值
        if (left < lenth/*防止下标越界*/ && array[left] < array[index]) min = left;
        if (right < lenth/*防止下标越界*/ && array[right] < array[index]) min = right;
        if (min != index) {
            //交换
            sweap(array, index, min);
            //重新调整堆
            heapify(array, index, lenth);
        }
    }

    /**
     * 30 最大连续子序列的和，
     * 思路动态规划
     * 迭代关系：
     * res[i] = Math.max(array[i], array[i] + res[i - 1]);
     *
     * @param array
     * @return
     */
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array == null) {
            return 0;
        }
        int[] res = new int[array.length];
        res[0] = array[0];
        int count = res[0];
        for (int i = 1; i < array.length; i++) {
            res[i] = Math.max(array[i], array[i] + res[i - 1]);
            count = Math.max(res[i], count);
        }
        return count;
    }

    /**
     * 31 整数中1出现的次数（从1到n整数中1出现的次数）
     *
     * @param n
     * @return
     */
    public int NumberOf1Between1AndN_Solution(int n) {
        int count = 0;
        for (int i = 0; i <= n; i++) {
            char[] temp = String.valueOf(i).toCharArray();
            count += number(temp);
        }
        return count;
    }

    int number(char[] input) {
        int count = 0;
        for (char i : input) {
            if (i == '1') {
                count++;
            }
        }
        return count;
    }

    /**
     * 32 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
     * 例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
     * <p>
     * compareTo 是根据字典序的排序规则来的
     * compareTo 的用法：
     * 如果这个字符串是等参数字符串那么返回值0
     * 如果这个字符串是按字典顺序小于字符串参数那么返回小于0的值
     * 如果此字符串是按字典顺序大于字符串参数那么返回一个大于0的值
     * <p>
     * eg：
     * String s1 = "abc"; 
     * String s2 = "abcd"; 
     * String s3 = "abcdfg"; 
     * String s4 = "1bcdfg"; 
     * String s5 = "cdfg"; 
     * System.out.println( s1.compareTo(s2) ); // -1 (前面相等,s1长度小1) 
     * System.out.println( s1.compareTo(s3) ); // -3 (前面相等,s1长度小3) 
     * System.out.println( s1.compareTo(s4) ); // 48 ("a"的ASCII码是97,"1"的的ASCII码是49,所以返回48) 
     * System.out.println( s1.compareTo(s5) ); // -2 ("a"的ASCII码是97,"c"的ASCII码是99,所以返回-2)
     *
     * @param numbers
     * @return
     */
    public String PrintMinNumber(int[] numbers) {
        String[] strings = new String[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            strings[i] = String.valueOf(numbers[i]);
        }
        Arrays.sort(strings, (o1, o2) -> {
            String s1 = o1 + o2;
            String s2 = o2 + o1;
            return s1.compareTo(s2);
        });
        StringBuilder stringBuilder = new StringBuilder();
        for (String s : strings) {
            stringBuilder.append(s);
        }
        return stringBuilder.toString();
    }

    /**
     * 33 把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。
     * 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
     * <p>
     * GetUglyNumber_Solution1 超时了
     *
     * @param index
     * @return
     */
    public int GetUglyNumber_Solution(int index) {
        if (index < 7) {
            return index;
        }
        int[] array = new int[index];
        int t2 = 0;
        int t3 = 0;
        int t5 = 0;
        array[0] = 1;
        for (int i = 1; i < index; i++) {
            int min = Math.min(array[t2] * 2, Math.min(array[t3] * 3, array[t5] * 5));
            array[i] = min;
            if (min == array[t2] * 2) t2++;
            if (min == array[t3] * 3) t3++;
            if (min == array[t5] * 5) t5++;
        }
        return array[index - 1];
    }

    public int GetUglyNumber_Solution1(int index) {
        if (index <= 0) {
            return 0;
        }
        int max = 1;
        int i = 1;
        while (max != index) {
            if (isUglyNum(++i)) {
                max++;
            }
        }
        return i;
    }

    public boolean isUglyNum(int number) {
        if (number == 1 || number == 2 || number == 3 || number == 5) {
            return true;
        }
        while (number != 1) {
            if (number % 2 == 0) {
                number /= 2;
            } else if (number % 3 == 0) {
                number /= 3;
            } else if (number % 5 == 0) {
                number /= 5;
            } else {
                return false;
            }
        }
        return true;
    }

    /**
     * 34 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置,
     * 如果没有则返回 -1（需要区分大小写）.（从0开始计数）
     * <p>
     * hashmap是无序的，linkedhashmap是有序的
     * hashmap 遍历
     *
     * @param str
     * @return
     */
    public int FirstNotRepeatingChar(String str) {
        int res = -1;
        LinkedHashMap<Character, Integer> hashMap = new LinkedHashMap<>();
        for (int i = 0; i < str.length(); i++) {
            if (!hashMap.containsKey(str.charAt(i))) {
                hashMap.put(str.charAt(i), 1);
            } else {
                hashMap.put(str.charAt(i), hashMap.get(str.charAt(i)) + 1);
            }
        }

        Iterator iterator = hashMap.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Character, Integer> entry = (Map.Entry<Character, Integer>) iterator.next();
            if (entry.getValue() == 1) {
                return str.indexOf(entry.getKey());
            }
        }
        return -1;
    }

    /**
     * 35 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。
     * 并将P对1000000007取模的结果输出。 即输出P%1000000007
     * <p>
     * 思路：
     * 利用归并排序，如果序列a的最小值大于序列b的最小值，a序列的全部就是可以组成逆序对
     */
    int count = 0;

    public int InversePairs(int[] arrays) {
        if (arrays == null || arrays.length == 1) {
            return 0;
        }
        decompose(arrays, 0, arrays.length - 1);
        return count;
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
        int[] temp = new int[right - left + 1];
        int k = 0;
        while (leftstart <= leftEnd && rightstart <= rightEnd) {
            if (arrays[leftstart] < arrays[rightstart]) {
                temp[k++] = arrays[leftstart++];
            } else {
                temp[k++] = arrays[rightstart++];
                count = (count + (mid - leftstart + 1)) % 1000000007;
            }
        }
        while (leftstart <= leftEnd) temp[k++] = arrays[leftstart++];
        while (rightstart <= rightEnd) temp[k++] = arrays[rightstart++];

        for (int i = 0; i < k; i++) {
            arrays[left + i] = temp[i];
        }
    }

    /**
     * 36 输入两个链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）
     * <p>
     * 思路：
     * 两个链表，不停的循环和便利，总会有相遇的时候
     *
     * @param pHead1
     * @param pHead2
     * @return
     */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null) {
            return null;
        }
        ListNode p1 = pHead1;
        ListNode p2 = pHead2;
        while (p1 != p2) {
            p1 = p1.next;
            p2 = p2.next;
            if (p1 != p2) {
                if (p1 == null) p1 = pHead1;
                if (p2 == null) p2 = pHead2;
            }/*说明p1和p2没有共同到尾部*/
        }
        return p1;
    }

    /**
     * 37 统计一个数字在排序数组中出现的次数。
     * <p>
     * 思路:二分查找找到位置，然后计数
     * <p>
     * 注意：
     * 1/ binarySearch 如果数组有重复元素，是不能确定具体是那个位置，所以需要前后都遍历一遍
     * 2/ j >= 0 && array[j] == compare，这种判断一定要注意数学
     * 下标为0是要遍历的，但是--完以后，while的判断条件如果先判断array[j] == compare，会被崩掉
     *
     * @param array
     * @param k
     * @return
     */
    public int GetNumberOfK(int[] array, int k) {
        int index = Arrays.binarySearch(array, k);
        int count = 0;
        if (index >= 0) {
            int compare = array[index];
            int i = index + 1;
            int j = index;
            while (i < array.length && array[i] == compare) {
                count++;
                i++;
            }
            while (j >= 0 && array[j] == compare) {
                count++;
                j--;
            }
        }
        return count;
    }

    /**
     * 38 输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
     * <p>
     * 思路：
     * 1、用递归，如果左孩子右孩子，取深度大的那个，遍历到根节点的时候，return就行了
     * 2、用层次遍历的方式
     *
     * @param root
     * @return
     */
    public int TreeDepth(TreeNode root) {
        /**  方法 一
         if (root == null) {
         return 0;
         }
         int left = TreeDepth(root.left);
         int right = TreeDepth(root.right);
         return Math.max(left, right) + 1;**/
        if (root == null) {
            return 0;
        }
        int count = 0;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();// 表示一层有多少个点
            count++;
            while (size != 0) {
                TreeNode treeNode = queue.poll();
                if (treeNode.left != null) {
                    queue.offer(treeNode.left);
                }
                if (treeNode.right != null) {
                    queue.offer(treeNode.right);
                }
                size--;// 每次遍历结束，这一行减一
            }
        }
        return count;
    }

    /**
     * 39 输入一棵二叉树，判断该二叉树是否是平衡二叉树。
     * <p>
     * 思路：
     * 1、如果是null返回true
     * 2、平衡二叉树的性质，如果是平衡，那左右都是平衡的
     * 3、如果左右节点的深度大于1，返回false
     * 4、如果左右节点的深度小于1，说明当前节点平衡，继续向下递归
     *
     * @param root
     * @return
     */
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (Math.abs(IsBalancedDepth(root.left) - IsBalancedDepth(root.right)) > 1) {
            return false;
        } else {
            return IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right);
        }
    }

    int IsBalancedDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = IsBalancedDepth(root.left);
        int right = IsBalancedDepth(root.right);
        int count = Math.max(left, right) + 1;
        return count;
    }

    /**
     * 40 一个整型数组里除了两个数字之外，其他的数字都出现了两次。
     * 请写程序找出这两个只出现一次的数字。
     * 思路：
     * 方法  1 ：用hash计数的方法
     *
     * @param array
     * @param num1
     * @param num2
     */
    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
        Map<Integer, Integer> map = new LinkedHashMap<>();
        for (int i = 0; i < array.length; i++) {
            if (!map.containsKey(array[i])) {
                map.put(array[i], 1);
            } else {
                map.put(array[i], 1 + map.get(array[i]));
            }
        }

        Iterator iterator = map.entrySet().iterator();
        int i = 0;
        int j = 0;
        int count = 0;
        while (iterator.hasNext()) {
            Map.Entry<Integer, Integer> entry = (Map.Entry<Integer, Integer>) iterator.next();
            if (entry.getValue() == 1 && count == 0) {
                num1[0] = entry.getKey();
                count++;
            } else if (entry.getValue() == 1 && count == 1) {
                num2[0] = entry.getKey();
                count++;
            }
        }
    }

    /**
     * 41 输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
     * <p>
     * 思路：
     * 暴力的方法
     *
     * @param sum
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        for (int i = 1; i < sum; i++) {
            int j = i;
            int count = 0;
            ArrayList<Integer> temp = new ArrayList<>();
            while (count < sum) {
                temp.add(j);
                count += j;
                if (count == sum) {
                    res.add(temp);
                    break;
                }
                j++;
            }
        }
        return res;
    }

    /**
     * 42 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，
     * 如果有多对数字的和等于S，输出两个数的乘积最小的。
     * <p>
     * 思想：
     * 左右游标来控制，参考快速排序，关键词：两个数组，递增排序
     *
     * @param array
     * @param sum
     * @return
     */
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        ArrayList<Integer> out = new ArrayList<>();
        if (array == null || array.length == 0) {
            return out;
        }
        int len = array.length;
        int i = 0;
        int j = len - 1;
        int compare = array[array.length - 1] * array[array.length - 1];
        int[] res = new int[2];
        while (i < j) {
            int temp = array[i] + array[j];
            if (temp > sum && i < j) {
                j--;
            } else if (temp < sum && i < j) {
                i++;
            } else {
                if (array[i] * array[j] < compare) {
                    compare = array[i] * array[j];
                    res[0] = array[i];
                    res[1] = array[j];
                }
                i++;
                j--;
            }
        }
        if (res[0] != 0 && res[1] != 0) {
            out.add(res[0]);
            out.add(res[1]);
        }
        return out;
    }

    /**
     * 43 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。
     * 对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
     * 例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
     * 是不是很简单？OK，搞定它！
     * <p>
     * 思想：
     * 用string的接口 substring
     *
     * @param str
     * @param n
     * @return
     */
    public String LeftRotateString(String str, int n) {
        if (str == null || str.length() == 0) {
            return "";
        }
        if (str.length() <= n) {
            return str;
        }
        int len = str.length();
        return str.substring(n, len) + str.substring(0, n);
    }

    /**
     * 44 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，
     * 有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。
     * 后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。
     * Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
     * <p>
     * 思路：
     * trim是去掉首位空格的意思
     * split 分割字符串
     * 基于空格分割字符串，利用stack拼接字符串
     *
     * @param str
     * @return
     */
    public String ReverseSentence(String str) {
        if (str == null || str.length() == 0) {
            return str;
        }
        String[] temp = str.split(" ");
        if (temp.length == 0) {
            return str;
        }
        Stack<String> stack = new Stack<>();
        for (String i : temp) {
            stack.push(i);
        }
        StringBuilder stringBuilder = new StringBuilder();
        while (!stack.isEmpty()) {
            stringBuilder.append(stack.pop()).append(" ");
        }
        return stringBuilder.toString().trim();
    }

    /**
     * 45 LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,
     * 想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”
     * 不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”
     * (大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何，
     * 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。
     * <p>
     * 思想：
     * 1、大王小王代表 0
     * 2、如果可以凑成的话，最大值和最小值的差一定要小于5
     * 3、手里的有效牌一定是小于等于5，这里可有用set做去重处理
     */
    public boolean isContinuous(int[] numbers) {
        if (numbers == null || numbers.length < 5) {
            return false;
        }
        //TreeSet插入是有序的
        TreeSet<Integer> set = new TreeSet<>();
        int num = 0;
        for (int i = 0; i < numbers.length; i++) {
            if (numbers[i] == 0) {
                num++;//计大王小王的个数
            } else {
                set.add(numbers[i]);
            }
        }
        //1、不能有重复的牌
        //2、最大值和最小值的差值要小于5
        if ((set.last() - set.first()) < 5 && (set.size() + num) == 5) {
            return true;
        }
        return false;
    }

    /**
     * 46 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。
     * HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。
     * 每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,
     * 从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。
     * 请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
     * 如果没有小朋友，请返回-1
     * <p>
     * 思想：
     * 方法1：建立一个循环链表，然后反复的遍历，每次遍历到m-1次的时候，去除节点，然后遍历到只有一个元素的时候，返回就行了
     *
     * @param n
     * @param m
     * @return
     */
    public int LastRemaining_Solution(int n, int m) {
        if (n <= 0 || m <= 0) {
            return -1;
        }
        ListNode head = new ListNode(0);
        ListNode node = head;

        for (int i = 1; i < n; i++) {
            node.next = new ListNode(i);
            node = node.next;
        }

        node.next = head;
        node = head;

        while (node.next != node) {
            int k = 1;
            while (k < m - 1) {
                k++;
                node = node.next;
            }
            node.next = node.next.next;
            node = node.next;
        }
        return node.val;
    }

    /**
     * 47 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
     * <p>
     * 思想：
     * 链接：https://www.nowcoder.com/questionTerminal/7a0da8fc483247ff8800059e12d7caf1?answerType=1&f=discussion
     * 来源：牛客网
     * <p>
     * 短路求值。
     * 作为"&&"和"||"操作符的操作数表达式，这些表达式在进行求值时，只要最终的结果已经可以确定是真或假，求值过程便告终止，这称之为短路求值（short-circuit evaluation）。
     * 假如expr1和expr2都是表达式，并且expr1的值为0，在下面这个逻辑表达式的求值过程中：
     * <p>
     * expr1 && expr2
     * expr2将不会进行求值，因为整个逻辑表达式的值已经可以确定为0。
     * expr1 || expr2
     * expr2将不会进行求值，因为整个逻辑表达式的值已经确定为1。
     * <p>
     * 思路：因此可以利用左边的表达式来作为递归结束的判断条件。因此递归的表达式就在右边了。而想到递归的解法，必然是sum=Sum(n)=Sum(n-1)+n
     * 使用&&,表示两边都为真，才为真，左边为假，右边就没用了。因此在不断递归时，直到左边为假时，才不执行右边。因此在第一次进行右边的判断时，就进入递归的调用。
     * 想到结束条件在左边，只能是n=0时结束，即从n递减到0结束，所以递归的调用理所当然放在了右边。由于左边需要不断的进行条件判断，因为需要一个每次递归后都递减的变量，
     * 而n是递减的，因此用n来作为左边的变量，int sum = n;
     * 而短路求值左边可以写为(n>0)或（n!=0），右边写为sum +=sum(n-1)，再加一个判断（实际上需要保证右边的条件一直为真），因此java版的解法就出来了
     *
     * @param n
     * @return
     */
    public int Sum_Solution(int n) {
        int count = n;
        boolean res = (n > 0) && (count = (Sum_Solution(n - 1) + count)) > 0;
        /* 右边的这个表达式是为了保证true,因为用了短路求值得方法 */
        return count;
    }

    /**
     * 48 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
     * <p>
     * 思路:
     * 执行加法 x ^ y
     * 进位操作 ( x & y ) << 1
     * 每次计算的结果有两部分:
     * (1) 当前位的部分
     * (2) 还有一个就是进位的值
     * <p>
     * 例如:
     * 按位加法： res = 11 ^ 01 = 10
     * 与运算进位： jinwei = (11 & 01) << 1 = ( 01 ) << 1 = 010
     * 因为有进位，所以需要继续加：res = res ^ jinwei = 10 ^ 010 = 00
     * 当前的进位是：jinwei = (10 & 10) << 1 = 100
     * 因为有进位，所以需要继续加
     * res = 100^000 = 100
     * jinwei = (100^000)<<1 == 0, 没有进位，当前计算结束
     * <p>
     * 链接：https://www.nowcoder.com/questionTerminal/59ac416b4b944300b617d4f7f111b215?answerType=1&f=discussion
     * 来源：牛客网
     *
     * @param num1
     * @param num2
     * @return
     */
    public int Add(int num1, int num2) {
        int res = 0;
        int ca = 0;
        do {
            res = num1 ^ num2;
            ca = (num1 & num2) << 1;
            num1 = res;
            num2 = ca;
        } while (ca != 0);
        return res;
    }

    /**
     * 50 在一个长度为n的数组里的所有数字都在0到n-1的范围内。
     * 数组中某些数字是重复的，但不知道有几个数字是重复的。
     * 也不知道每个数字重复几次。请找出数组中任意一个重复的数字。
     * 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
     * <p>
     * 思想：
     * hash
     *
     * @param numbers
     * @param length
     * @param duplication
     * @return
     */
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        Map<Integer, Integer> map = new LinkedHashMap<>();
        if (numbers == null || numbers.length <= 1) {
            return false;
        }
        for (int i = 0; i < numbers.length; i++) {
            if (!map.containsKey(numbers[i])) {
                map.put(numbers[i], 1);
            } else {
                duplication[0] = numbers[i];
                return true;
            }
        }
        return false;
    }

    /**
     * 56 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
     * 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
     * <p>
     * 思路：
     * 方法1：用hash的方法计数，新建链表返回
     * 方法2：通过cur和next两个指针遍历，但是头结点需要单独处理，容易出空指针
     *
     * @param pHead
     * @return
     */
    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null) {
            return null;
        }
        ListNode res = new ListNode(0);
        ListNode node = res;
        Map<Integer, Integer> map = new LinkedHashMap<>();
        while (pHead != null) {
            if (!map.containsKey(pHead.val)) {
                map.put(pHead.val, 1);
            } else {
                map.put(pHead.val, map.get(pHead.val) + 1);
            }
            pHead = pHead.next;
        }

        Iterator iterator = map.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Integer, Integer> tmp = (Map.Entry<Integer, Integer>) iterator.next();
            if (tmp.getValue() == 1) {
                node.next = new ListNode(tmp.getKey());
                node = node.next;
            }
        }
        node.next = null;
        return res.next;
    }

    /**
     * 58 请实现一个函数，用来判断一颗二叉树是不是对称的。
     * 注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
     */
    public static boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetrictmp(root.left, root.right);
    }

    /**
     * isSymmetrictmp 判读左右节点是否是相等的
     * 思想：
     * 左孩子等右孩子，右孩子等于左孩子
     * 1
     * / \
     * 2   2
     * / \ / \
     * 3  4 4  3
     *
     * @param left
     * @param right
     * @return
     */
    private static boolean isSymmetrictmp(TreeNode left, TreeNode right) {
        if ((left == null) && (right == null)) {
            return true;
        }

        if ((left == null) || (right == null)) {
            return false;
        }

        if (left.val != right.val) {
            return false;
        }
        return isSymmetrictmp(left.left, right.right) && isSymmetrictmp(left.right, right.left);
    }

    /**
     * 59 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，
     * 第二层按照从右至左的顺序打印，
     * 第三行按照从左到右的顺序打印，其他行以此类推。
     * <p>
     * 思路：
     * 层次遍历，每次修改ArrayList的插入顺序
     *
     * @param pRoot
     * @return
     */
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (pRoot == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(pRoot);
        boolean isLeft = true;
        while (!queue.isEmpty()) {
            int size = queue.size();// 表示一层有多少个点
            ArrayList<Integer> tmp = new ArrayList<>();
            while (size != 0) {
                TreeNode treeNode = queue.poll();
                if (isLeft) {
                    tmp.add(treeNode.val);
                } else {
                    //表示插入的点
                    tmp.add(0, treeNode.val);
                }
                if (treeNode.left != null) {
                    queue.offer(treeNode.left);
                }
                if (treeNode.right != null) {
                    queue.offer(treeNode.right);
                }
                size--;// 每次遍历结束，这一行减一
            }
            isLeft = !isLeft;
            res.add(tmp);
        }
        return res;
    }

    /**
     * 60 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
     * <p>
     * 思想：
     * 层次遍历
     *
     * @param pRoot
     * @return
     */
    public ArrayList<ArrayList<Integer>> Print_1(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (pRoot == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList();
        queue.offer(pRoot);
        while (!queue.isEmpty()) {
            int size = queue.size();
            ArrayList<Integer> tmp = new ArrayList<>();
            while (size > 0) {
                TreeNode treeNode = queue.poll();
                tmp.add(treeNode.val);
                if (treeNode.left != null) {
                    queue.offer(treeNode.left);
                }
                if (treeNode.right != null) {
                    queue.offer(treeNode.right);
                }
                size--;
            }
            res.add(tmp);
        }
        return res;
    }
}
