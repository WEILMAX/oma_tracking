import pandas as pd

def aggregate_time_series(
    data: pd.DataFrame,
    freq: str = 'D',
    ) -> pd.DataFrame:
    """Resamples each time series in the input DataFrame
    to a longer timespan and computes the average over each new timespan.

    Args:
        data (pd.DataFrame): Multiple time series with higher sampling rate.
        freq (str, optional): Frequency of the resampled time series.
        Defaults to 'D'.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Resample each time series in the dataframe to the desired frequency
    resampled = data.resample(freq).mean()#.shift(1, freq=freq)

    # Check if the resampled data at least contains 90% of the original data
    full_length = data.groupby(pd.Grouper(freq=freq)).size().max()
    mask = data.groupby(pd.Grouper(freq=freq)).size() >= 0.9*full_length # type: ignore
    mask.index = resampled.index

    # Return only the resampled data that contains at least 75% of the original data
    return resampled.loc[mask]