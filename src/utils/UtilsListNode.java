package utils;

import Bean.ListNode;

public class UtilsListNode {
    public static final int CREAT_LIST_BEGIN = 1;
    public static final int CREAT_LIST_END = 2;

    /**
     * 传入的头结点，不是头指针，这个需要区分开
     *
     * @param listNode
     */
    public static void printList(ListNode listNode) {
        if (listNode == null) {
            return;
        }
        ListNode listNode1 = listNode;
        while (listNode1 != null) {
            System.out.println(listNode1.val);
            listNode1 = listNode1.next;
        }
    }

    /**
     * 头插入
     *
     * @param arrays
     * @return
     */
    public static ListNode createList(int[] arrays, int code) {
        if (code == CREAT_LIST_BEGIN) {
            return getListNodefromHead(arrays);
        } else if (code == CREAT_LIST_END) {
            return getListNodefromeEnd(arrays);
        }
        return null;
    }

    private static ListNode getListNodefromHead(int[] arrays) {
        ListNode listNode = new ListNode(0);
        listNode.next = null;
        for (int i = 0; i < arrays.length; i++) {
            ListNode bean = new ListNode(arrays[i]);
            bean.next = listNode.next;
            listNode.next = bean;
        }
        return listNode.next;
    }

    private static ListNode getListNodefromeEnd(int[] arrays) {
        ListNode listNode = new ListNode(0);
        ListNode temp = listNode;
        for (int i = 0; i < arrays.length; i++) {
            ListNode bean = new ListNode(arrays[i]);
            temp.next = bean;
            temp = temp.next;
        }
        temp.next = null;
        return listNode.next;
    }
}
