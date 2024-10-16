def filter_empty_strings(my_list):
  """Filters out empty strings from a list.

  Args:
      my_list: The list to filter.

  Returns:
      A new list with only the non-empty strings from the original list.
  """
  return list(filter(None, my_list))

# Example usage
my_list = ['', '', '', '-1']
filtered_list = filter_empty_strings(my_list)
print(filtered_list)  # Output: ['-1']

