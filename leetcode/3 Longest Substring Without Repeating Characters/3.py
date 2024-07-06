class Solution:
    """Longest Substring Without Repeating Characters"""

    def lengthOfLongestSubstring(self, s: str) -> int:
        """The function returns the length of the longest substring without repeating characters."""
        # Longest length is the length of the longest substring without repeating characters.
        longest_length = 0
        # Bag of characters is the set of characters in the current substring.
        bag_of_characters = set()
        # Left pointer is the leftmost index of the current substring.
        left_pointer = 0
        # Iterate over the string.
        for right_pointer, right_character in enumerate(s):
            # If the right character is not in the bag of characters, update the longest length.
            if right_character not in bag_of_characters:
                longest_length = max(longest_length, right_pointer - left_pointer + 1)
            else:
                # Remove the leftmost character from the bag of characters
                # if the right character is in the bag of characters.
                while right_character in bag_of_characters:
                    bag_of_characters.remove(s[left_pointer])
                    left_pointer += 1
            # Add the right character to the bag of characters.
            bag_of_characters.add(s[right_pointer])
        # Return the longest length.
        return longest_length
