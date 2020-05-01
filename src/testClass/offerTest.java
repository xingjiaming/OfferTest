package testClass;

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
            TreeNode treeNode = queue.poll();
            arrayList.add(treeNode.val);
            if (treeNode.left != null) {
                queue.offer(treeNode.left);
            }
            if (treeNode.right != null) {
                queue.offer(treeNode.right);
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
     * 32 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
     * <p>
     * compareTo 是根据字典序的排序规则来的
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
}
