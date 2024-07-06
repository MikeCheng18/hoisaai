"""This module contains a solution for the problem 4. Median of Two Sorted Arrays."""

from typing import List


class Solution:
    """Median of Two Sorted Arrays"""

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """The function finds the median of two sorted arrays."""
        # Get the lengths of the arrays.
        nums1_len: int = len(nums1)
        nums2_len: int = len(nums2)
        # Initialize the indices and medians.
        i: int = 0
        j: int = 0
        median_1: int = 0
        median_2: int = 0
        # Iterate only half of the sum of the lengths of the arrays.
        for _ in range(
            0,
            # Number of iterations
            (nums1_len + nums2_len) // 2 + 1,
        ):
            # Set the previous median to the current median.
            median_2 = median_1
            # Check if the indices are less than the lengths of the arrays.
            if i < nums1_len and j < nums2_len:
                # Compare the elements of the arrays.
                if nums1[i] > nums2[j]:
                    median_1 = nums2[j]
                    j += 1
                else:
                    median_1 = nums1[i]
                    i += 1
            # Check if the index is less than the length of the first array.
            elif i < nums1_len:
                median_1 = nums1[i]
                i += 1
            else:
                median_1 = nums2[j]
                j += 1
        # Return the median.
        # If the sum of the lengths of the arrays is odd, return the median.
        if (nums1_len + nums2_len) % 2 == 1:
            return float(median_1)
        # If the sum of the lengths of the arrays is even, return the average of the medians.
        else:
            ans = float(median_1) + float(median_2)
            return ans / 2.0
