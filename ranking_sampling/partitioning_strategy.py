def generate_intervals(start_range, end_range, m, interval_multiplier=2.0, score=False, num_samples=1):
    """
    Generate m intervals within a specified range [start_range, end_range],
    where each interval is interval_multiplier times the size of the previous one.
    The second interval is interval_multiplier times the size of the first, and so on.

    Parameters:
    - start_range: The starting point of the large range (can be float).
    - end_range: The ending point of the large range (can be float).
    - m: The number of intervals.
    - interval_multiplier: The scaling factor of intervals
    - score: whether generate float intervals

    Returns:
    - A list of tuples representing the start and end points of the intervals.
    """

    total_range = end_range - start_range
    # Calculate the sum of the geometric series, e.g., given m=2, (1 + 2 + 4 + ... + 2^(m-1))
    total_sum = sum([interval_multiplier ** i for i in range(m)])

    # The first interval size is determined by dividing the total range by the sum of the series
    first_interval_size = total_range / total_sum

    intervals = []
    start = start_range
    size = first_interval_size
    if score:
        for i in range(m):
            # Calculate the end point for each interval
            end = start + size
            # Ensure that the end point does not exceed the end_range
            if end > end_range:
                end = end_range
            intervals.append([start, min(end, end_range)])  # Store the start and end for each interval
            start = end
            size *= interval_multiplier  # Scaling the size of the next interval

        # Ensure the last interval ends exactly at end_range
        intervals[-1] = [intervals[-1][0], end_range]

        return intervals

    else:

        for i in range(m):
            # Calculate the end point for each interval
            end = start + size
            if int(end) <= int(start+num_samples):
                end = start + num_samples
            intervals.append([int(start), int(min(end, end_range))])  # Store the start and end for each interval
            start = end
            size *= interval_multiplier  # scaling the size of the next interval

        return intervals

if __name__ == "__main__":
    intervals = generate_intervals(start_range=3, end_range=1000, m=10, interval_multiplier=1.8)
    for interval in intervals:
        print(f'start:{interval[0]} end:{interval[1]}')