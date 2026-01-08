BEGIN {
    FS = " "    # Field separator is a comma
    window_size = 7
}

{
    values[NR] = $2        # Store the numeric values in a mapping with record numbers
    dates[NR] = $1         # Store corresponding dates
    sum += $2              # Add current value to sum
    count++

    if (NR > window_size) {
        sum -= values[NR - window_size]  # Subtract the oldest value outside the window
        count--
    }

    # Print moving average only once window is populated
    if (count == window_size) {
        print dates[NR - window_size + 1] " " sum / count
    }
}
