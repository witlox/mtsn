import numpy as np
import pandas as pd
from scipy import integrate
from statsmodels.tsa.seasonal import STL


class STLTemporalIntegration:
    """
    A class for implementing Seasonal-Trend decomposition with Temporal Integration
    for business KPI analysis.
    """

    def __init__(self, data, date_column, value_column, period=None):
        """
        Initialize with time series data.

        Parameters:
        -----------
        data : pandas DataFrame
            DataFrame containing the time series data
        date_column : str
            Name of the column containing datetime values
        value_column : str
            Name of the column containing KPI values
        period : int, optional
            The seasonal period (e.g., 7 for weekly, 12 for monthly, 365 for annual)
            If None, it will be inferred from the data frequency
        """
        self.df = data.copy()
        self.date_col = date_column
        self.value_col = value_column

        if self.df.index.name != self.date_col:
            # Ensure date column is datetime type
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            # Sort by date
            self.df = self.df.sort_values(by=self.date_col).reset_index(drop=True)
            # Set date as index
            self.df.set_index(self.date_col, inplace=True)

        # Infer frequency if period is not provided
        if period is None:
            # Try to infer frequency from data
            freq = pd.infer_freq(self.df.index)
            if freq == 'D':
                self.period = 7  # Daily data, weekly seasonality
            elif freq == 'B':
                self.period = 5  # Business days, weekly seasonality
            elif freq in ['M', 'MS']:
                self.period = 12  # Monthly data, annual seasonality
            elif freq in ['Q', 'QS']:
                self.period = 4  # Quarterly data, annual seasonality
            else:
                self.period = 7  # Default to weekly
                print(f"Warning: Could not infer seasonality period from frequency {freq}. Defaulting to 7.")
        else:
            self.period = period

        # Initialize decomposition components
        self.decomposition = None
        self.trend = None
        self.seasonal = None
        self.resid = None

        print(f"Initialized with {len(self.df)} observations and seasonal period of {self.period}")

    def decompose(self, seasonal_deg=1, trend_deg=1, low_pass_deg=1, seasonal_jump=1, trend_jump=1, low_pass_jump=1, robust=False):
        """
        Perform STL decomposition on the time series.

        Parameters:
        -----------
        seasonal_deg : int
            Degree of seasonal LOESS
        trend_deg : int
            Degree of trend LOESS
        low_pass_deg : int
            Degree of low-pass LOESS
        seasonal_jump : int
            Jump for seasonal LOESS
        trend_jump : int
            Jump for trend LOESS
        low_pass_jump : int
            Jump for low-pass LOESS
        robust : bool
            Whether to use robust fitting
        """
        # Handle missing values if any
        if self.df[self.value_col].isnull().any():
            print("Warning: Missing values detected. Interpolating...")
            self.df[self.value_col] = self.df[self.value_col].interpolate(method='linear')

        # Perform STL decomposition
        stl = STL(self.df[self.value_col],
                  period=self.period,
                  seasonal_deg=seasonal_deg,
                  trend_deg=trend_deg,
                  low_pass_deg=low_pass_deg,
                  seasonal_jump=seasonal_jump,
                  trend_jump=trend_jump,
                  low_pass_jump=low_pass_jump,
                  robust=robust)

        result = stl.fit()

        # Store decomposition components
        self.trend = result.trend
        self.seasonal = result.seasonal
        self.resid = result.resid

        # Create DataFrame with decomposition results
        self.decomposition = pd.DataFrame({
            'original': self.df[self.value_col],
            'trend': self.trend,
            'seasonal': self.seasonal,
            'remainder': self.resid,
            'deseasonalized': self.df[self.value_col] - self.seasonal
        }, index=self.df.index)

        print(f"Decomposition completed. Components extracted: trend, seasonal, remainder")
        return self

    def temporal_integration(self, start_date=None, end_date=None, component='trend', weighted=False, lambda_weight=0.1, normalize=False):
        """
        Calculate temporal integration over a specified time interval.

        Parameters:
        -----------
        start_date : datetime or str, optional
            Start date for integration. If None, uses the first date.
        end_date : datetime or str, optional
            End date for integration. If None, uses the last date.
        component : str, optional
            Component to integrate: 'trend', 'original', 'deseasonalized'
        weighted : bool, optional
            Whether to apply exponential weighting
        lambda_weight : float, optional
            Rate parameter for exponential weighting
        normalize : bool, optional
            Whether to normalize by time interval length

        Returns:
        --------
        float
            The temporal integration value
        """
        if self.trend is None:
            raise ValueError("Decomposition has not been performed. Call decompose() first.")

        # Set default dates if not provided
        if start_date is None:
            start_date = self.df.index.min()
        else:
            start_date = pd.to_datetime(start_date)

        if end_date is None:
            end_date = self.df.index.max()
        else:
            end_date = pd.to_datetime(end_date)

        # Get data for the specified time interval
        mask = (self.decomposition.index >= start_date) & (self.decomposition.index <= end_date)
        subset = self.decomposition[mask]

        if len(subset) == 0:
            raise ValueError(f"No data found between {start_date} and {end_date}")

        # Select component to integrate
        if component == 'trend':
            values = subset['trend']
        elif component == 'original':
            values = subset['original']
        elif component == 'deseasonalized':
            values = subset['deseasonalized']
        else:
            raise ValueError(f"Unknown component: {component}")

        # Calculate time differences in days
        time_deltas = [(t - subset.index[0]).total_seconds() / (24*3600) for t in subset.index]

        # Apply weighting if requested
        if weighted:
            # Calculate exponential weights
            max_time = time_deltas[-1]
            weights = np.exp(lambda_weight * (np.array(time_deltas) - max_time))
            weighted_values = values * weights

            # Perform numerical integration using trapezoidal rule
            integration = integrate.trapezoid(weighted_values, time_deltas)
        else:
            # Perform numerical integration using trapezoidal rule
            integration = integrate.trapezoid(values, time_deltas)

        # Normalize if requested
        if normalize:
            time_span = time_deltas[-1]
            integration = integration / time_span

        return integration

    def compare_integrations(self, intervals, component='trend', weighted=False, lambda_weight=0.1, normalize=True):
        """
        Compare temporal integrations across multiple time intervals.

        Parameters:
        -----------
        intervals : list of tuples
            List of (start_date, end_date, label) tuples defining intervals
        component : str, optional
            Component to integrate
        weighted : bool, optional
            Whether to apply exponential weighting
        lambda_weight : float, optional
            Rate parameter for exponential weighting
        normalize : bool, optional
            Whether to normalize by time interval length

        Returns:
        --------
        pandas DataFrame
            DataFrame with integration results for each interval
        """
        results = []

        for start_date, end_date, label in intervals:
            # Calculate integrations for different components
            trend_integration = self.temporal_integration(
                start_date, end_date, 'trend', weighted, lambda_weight, normalize)

            original_integration = self.temporal_integration(
                start_date, end_date, 'original', weighted, lambda_weight, normalize)

            deseasonalized_integration = self.temporal_integration(
                start_date, end_date, 'deseasonalized', weighted, lambda_weight, normalize)

            # Calculate time span in days
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            span_days = (end - start).total_seconds() / (24*3600)

            results.append({
                'interval': label,
                'start_date': start,
                'end_date': end,
                'span_days': span_days,
                'trend_integration': trend_integration,
                'original_integration': original_integration,
                'deseasonalized_integration': deseasonalized_integration,
                'trend_vs_original_diff_pct': (trend_integration - original_integration) / original_integration * 100 if original_integration != 0 else np.nan
            })

        return pd.DataFrame(results)

    def reset_index(self):
        """
        Reset the index of the DataFrame.
        """
        self.df.reset_index(inplace=True)

    def iterrows(self):
        """
        Iterate over DataFrame rows.
        """
        return self.df.iterrows()
