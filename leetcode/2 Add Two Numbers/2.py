from __future__ import annotations
from typing import Optional


class ListNode:
    """Definition for singly-linked list."""

    def __init__(
        self,
        val: int = 0,
        next: ListNode = None,
    ):
        self.val: int = val
        self.next: ListNode = next


class Solution:
    """Add Two Numbers"""

    def addTwoNumbers(
        self,
        l1: Optional[ListNode],
        l2: Optional[ListNode],
    ) -> Optional[ListNode]:
        """The function adds two numbers represented by linked lists."""
        # Carry is the number that is carried over to the next digit.
        carry: int = 0
        # Answer is the head of the linked list.
        answer: ListNode = ListNode(
            val=None,
            next=None,
        )
        # Dummy is the head of the linked list that will be returned.
        dummy = answer
        # Digit is the number that is added to the linked list.
        digit: int = 0
        # Iterate over the linked lists.
        while l1 or l2 or carry:
            # Add the values of the linked lists and the carry.
            if l1:
                carry += l1.val
                l1 = l1.next
            # Add the values of the linked lists and the carry.
            if l2:
                carry += l2.val
                l2 = l2.next
            # Calculate the digit and the carry.
            carry, digit = divmod(carry, 10)
            # Add the digit to the linked list.
            answer.next = ListNode(
                val=digit,
                next=None,
            )
            # Move to the next node.
            answer = answer.next
        # Return the linked list.
        return dummy.next
