class Solution:
    """Longest Palindromic Substring"""

    def longestPalindrome(self, s: str) -> str:
        """The function finds the longest palindromic substring."""
        # If the length of the string is less than or equal to 1, return the string.
        if len(s) <= 1:
            return s
        max_length = 1
        max_string = s[0]
        # Create a 2D array to store the results of the palindromic substrings.
        dp = [[False for _ in range(len(s))] for _ in range(len(s))]
        # Iterate over the string.
        for i in range(len(s)):
            # Set the diagonal elements to True.
            dp[i][i] = True
            # Iterate over the string.
            for j in range(i):
                # Check if the characters are the same and the substring is a palindrome.
                if s[j] == s[i] and (i - j <= 2 or dp[j + 1][i - 1]):
                    # Set the value to True.
                    dp[j][i] = True
                    # Check if the length of the substring is greater than the maximum length.
                    if i - j + 1 > max_length:
                        # Update the maximum length and the maximum string.
                        max_length = i - j + 1
                        max_string = s[j : i + 1]
        # Return the maximum string.
        return max_string
