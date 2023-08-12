def moving_average(scores, window_size):
    """
    Computes the moving average of the given scores.

    Parameters:
    - scores (list): A list of numerical scores.
    - window_size (int): The window size for the moving average.

    Returns:
    - list: Moving average of the given scores.
    """
    cumsum = [0]
    moving_aves = []

    for i, x in enumerate(scores, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= window_size:
            moving_ave = (cumsum[i] - cumsum[i - window_size]) / window_size
            moving_aves.append(moving_ave)

    return moving_aves
