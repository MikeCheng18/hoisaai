class Solution:
    """Zigzag Conversion"""

    def convert(self, s: str, numRows: int) -> str:
        """The function converts the string to a zigzag pattern."""
        # If the number of rows is 1 or greater than or equal to the length of the string,
        # return the string.
        if numRows == 1 or numRows >= len(s):
            return s
        # Initialize the index and the change in row.
        index, change_in_row = 0, 1
        # Create a list of lists to store the rows.
        rows = [[] for _ in range(numRows)]
        # Iterate over the string.
        for char in s:
            # Append the character to the row.
            rows[index].append(char)
            # Check if the index is 0 or the last row.
            if index == 0:
                change_in_row = 1
            elif index == numRows - 1:
                change_in_row = -1
            # Update the index.
            index += change_in_row
        # Join the rows and return the result.
        for i in range(numRows):
            rows[i] = "".join(rows[i])
        # Return the result.
        return "".join(rows)
