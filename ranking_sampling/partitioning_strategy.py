def generate_intervals(start_range, end_range, m, interval_multiplier=2):
    """
    Generate m intervals within a specified range [start_range, end_range],
    where each interval is interval_multiplier times the size of the previous one.
    The second interval is interval_multiplier times the size of the first, and so on.

    Parameters:
    - start_range: The starting point of the large range.
    - end_range: The ending point of the large range.
    - m: The number of intervals.
    - interval_multiplier: The scaling factor of intervals

    Returns:
    - A list of tuples representing the start and end points of the intervals.
    """

    total_range = end_range - start_range
    # Calculate the sum of the geometric series, e.g., given n=2 (1 + 2 + 4 + ... + 2^(m-1))
    total_sum = sum([interval_multiplier ** i for i in range(m)])

    # The first interval size is determined by dividing the total range by the sum of the series
    first_interval_size = total_range / total_sum

    intervals = []
    start = start_range
    size = first_interval_size

    for i in range(m):
        # Calculate the end point for each interval
        end = start + size
        if int(end) == int(start):
            end = end + 1
        intervals.append([int(start), int(min(end, end_range))])  # Store the start and end for each interval
        start = end
        size *= interval_multiplier  # scaling the size of the next interval

    # Ensure the last interval ends exactly at end_range
    intervals[-1] = [int(intervals[-1][0]), int(end_range)]

    return intervals



if __name__ == "__main__":
    intervals = generate_intervals(start_range=3, end_range=1200, m=8, interval_multiplier=1)
    for interval in intervals:
        print(f'start:{interval[0]} end:{interval[1]}')