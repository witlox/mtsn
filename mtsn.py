import warnings
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import networkx as nx

from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore


class MTSN:
    """
    Multi-Layer Temporal-Seasonal Network (MTSN) for KPI Analysis.

    This class implements the MTSN framework as described in the paper:
    "Multi-Layer Temporal-Seasonal Network for KPI Analysis" by P. Witlox.
    """

    def __init__(self, data=None, dates=None, seasonal_periods=None, trend_window=None):
        """
        Initialize the MTSN framework.

        Parameters:
        -----------
        data : array-like or pandas.Series
            The time series data (KPI values)
        dates : array-like or pandas.DatetimeIndex
            The corresponding dates for the time series data
        seasonal_periods : list of int or int
            The periods of the seasonality components (e.g., [7, 30, 365] for weekly, monthly, and yearly)
            If a single integer is provided, it will be converted to a list with one element
        trend_window : int
            The window length for the trend component in STL decomposition
        """
        self.data = data
        self.dates = dates
        if seasonal_periods is not None and not isinstance(seasonal_periods, list):
            self.seasonal_periods = [seasonal_periods]
        else:
            self.seasonal_periods = seasonal_periods

        self.trend_window = trend_window

        # Initialize components
        self.decomposition = None
        self.graph = None
        self.hotspots = None

        if data is not None and dates is not None and seasonal_periods is not None:
            self._validate_inputs()

    def _validate_inputs(self):
        """Validate input data and parameters."""
        # Convert to pandas Series if needed
        if not isinstance(self.data, pd.Series):
            if self.dates is not None:
                self.data = pd.Series(self.data, index=self.dates)
            else:
                self.data = pd.Series(self.data)

        # Ensure dates are in datetime format
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.DatetimeIndex(self.data.index)
                self.dates = self.data.index
            except:
                raise ValueError("Dates must be convertible to datetime format")

        # Validate seasonal periods
        if not all(period > 1 for period in self.seasonal_periods):
            raise ValueError("All seasonal periods must be greater than 1")

        # Sort seasonal periods in ascending order for efficient decomposition
        self.seasonal_periods.sort()

        if self.trend_window is not None:
            min_required = 2 * max(self.seasonal_periods) + 1
            if self.trend_window < min_required:
                raise ValueError(f"trend_window must be >= {min_required} for given seasonal_periods")
            if self.trend_window % 2 == 0:
                self.trend_window += 1  # Ensure odd
        else:
            # Calculate trend window as 2x + 1 of max period to ensure it's odd and large enough
            max_period = max(self.seasonal_periods)
            self.trend_window = 2 * max_period + 1
            # Ensure it's at least 11
            self.trend_window = max(self.trend_window, 11)

    def fit(self, data=None, dates=None, seasonal_periods=None, trend_window=None):
        """
        Fit the MTSN model to the data.

        This method performs the time series decomposition and constructs the graph.

        Parameters:
        -----------
        data : array-like or pandas.Series, optional
            The time series data (KPI values)
        dates : array-like or pandas.DatetimeIndex, optional
            The corresponding dates for the time series data
        seasonal_periods : list of int or int, optional
            The periods of the seasonality components
        trend_window : int, optional
            The window length for the trend component in STL decomposition

        Returns:
        --------
        self : object
            Returns self.
        """
        # Update parameters if provided
        if data is not None:
            self.data = data
        if dates is not None:
            self.dates = dates
        if seasonal_periods is not None:
            # Convert single period to list if needed
            if not isinstance(seasonal_periods, list):
                self.seasonal_periods = [seasonal_periods]
            else:
                self.seasonal_periods = seasonal_periods
        if trend_window is not None:
            self.trend_window = trend_window

        self._validate_inputs()

        # Perform time series decomposition
        self._decompose_time_series()

        # Construct the MTSN graph
        self._construct_graph()

        return self

    def _decompose_time_series(self, seasonal_smoothing=None):
        """
        Perform time series decomposition using STL.
        """
        # Handle missing values if any
        has_missing = self.data.isnull().any()
        if has_missing:
            warnings.warn("Data contains missing values. Interpolating before decomposition.")
            self.data = self.data.interpolate()

        # Initialize DataFrame to store decomposition
        self.decomposition = pd.DataFrame({'original': self.data}, index=self.data.index)

        # Initialize storage for individual seasonal components
        self.seasonal_components = {}

        # Start with the original series
        remainder = self.data.copy()

        # Apply STL decomposition for each seasonal period
        for i, period in enumerate(self.seasonal_periods):
            seasonal_id = f"seasonal_{period}"

            # Set seasonal smoothing
            if seasonal_smoothing is None:
                # Default calculation: typically 7 for weekly patterns, larger for longer periods
                seasonal_smoothing = min(max(7, period), 2 * period + 1)
                if seasonal_smoothing % 2 == 0:
                    seasonal_smoothing += 1

            # Ensure trend window is odd and greater than period for each decomposition
            adjusted_trend = max(period + 2, self.trend_window)
            if adjusted_trend % 2 == 0:
                adjusted_trend += 1

            # Apply STL decomposition with the adjusted trend window
            stl = STL(remainder, period=period, seasonal=seasonal_smoothing, trend=adjusted_trend, robust=True)
            result = stl.fit()

            # Store this seasonal component
            self.seasonal_components[seasonal_id] = result.seasonal
            self.decomposition[seasonal_id] = result.seasonal

            # If this is the last seasonal period, also store the trend and remainder
            if i == len(self.seasonal_periods) - 1:
                self.decomposition['trend'] = result.trend
                self.decomposition['remainder'] = result.resid

            # Update remainder by removing this seasonal component
            remainder = remainder - result.seasonal

        # Create a combined seasonal component (sum of all individual seasonal components)
        self.decomposition['seasonal'] = sum(self.seasonal_components.values())

        return self.decomposition


    def _construct_graph(self, seasonal_corr_threshold=0.5):
        """
        Construct the Multi-Layer Temporal-Seasonal Network graph.

        Parameters:
        -----------
        seasonal_corr_threshold : float
            Threshold for seasonal correlation to add seasonal cycle edges

        Returns:
        --------
        networkx.DiGraph
            The constructed MTSN graph
        """

        def calculate_seasonal_correlation(seasonal_component, lag):
            """Calculate auto-correlation of seasonal component at specified lag"""
            # Extract the seasonal component series
            s_series = np.array(seasonal_component)
            n = len(s_series)

            if lag >= n:
                return 0.0

            # Calculate mean
            mean = np.mean(s_series)

            # Calculate numerator (covariance)
            numerator = 0
            for i in range(n - lag):
                numerator += (s_series[i] - mean) * (s_series[i + lag] - mean)

            # Calculate denominator (variance)
            denominator = np.sum((s_series - mean) ** 2)

            # Return autocorrelation
            if denominator == 0:
                return 0.0
            return numerator / denominator

        # Initialize directed graph
        self.graph = nx.DiGraph()

        # Add time nodes and component nodes
        for i, date in enumerate(self.decomposition.index):
            # Create time node
            time_node_id = f"t_{i}"
            self.graph.add_node(time_node_id, type='time', date=date, index=i, value=self.decomposition.iloc[i]['original'])

            # Create component nodes
            trend_node_id = f"T_{i}"
            remainder_node_id = f"R_{i}"

            self.graph.add_node(trend_node_id, type='trend', date=date, index=i, value=self.decomposition.iloc[i]['trend'])
            self.graph.add_node(remainder_node_id, type='remainder', date=date, index=i, value=self.decomposition.iloc[i]['remainder'])

            # Add combined seasonal node
            seasonal_node_id = f"S_{i}"
            self.graph.add_node(seasonal_node_id, type='seasonal', date=date, index=i, value=self.decomposition.iloc[i]['seasonal'])

            # Add individual seasonal component nodes
            for period in self.seasonal_periods:
                seasonal_id = f"seasonal_{period}"
                seasonal_comp_node_id = f"S{period}_{i}"

                self.graph.add_node(seasonal_comp_node_id, type='seasonal_component', period=period, date=date, index=i, value=self.decomposition.iloc[i][seasonal_id])

                # Add edge from combined seasonal node to this component
                self.graph.add_edge(seasonal_node_id, seasonal_comp_node_id, weight=1.0, type='seasonal_decomposition')

            # Add decomposition edges (from time node to primary component nodes)
            self.graph.add_edge(time_node_id, trend_node_id, weight=1.0, type='decomposition')
            self.graph.add_edge(time_node_id, seasonal_node_id, weight=1.0, type='decomposition')
            self.graph.add_edge(time_node_id, remainder_node_id, weight=1.0, type='decomposition')

            # Add temporal progression edges (between consecutive time nodes)
            if i < len(self.decomposition) - 1:
                next_time_node_id = f"t_{i + 1}"
                time_diff = (self.decomposition.index[i + 1] - date).total_seconds() / (24 * 3600)  # in days
                self.graph.add_edge(time_node_id, next_time_node_id, weight=time_diff, type='temporal_progression')

        # Add seasonal cycle edges for each seasonal period
        for period in self.seasonal_periods:
            seasonal_id = f"seasonal_{period}"
            seasonal_array = np.array(self.seasonal_components[seasonal_id])

            for i in range(len(self.decomposition) - period):
                seasonal_comp_node_id = f"S{period}_{i}"
                seasonal_cycle_node_id = f"S{period}_{i + period}"

                seasonal_corr = calculate_seasonal_correlation(seasonal_array, period)

                # Add edge if correlation is significant
                if  seasonal_corr > seasonal_corr_threshold:
                    self.graph.add_edge(seasonal_comp_node_id, seasonal_cycle_node_id, weight=seasonal_corr, type='seasonal_cycle', period=period)

        # Add pattern nodes for each identified seasonal pattern
        for period in self.seasonal_periods:
            pattern_node_id = f"P_{period}"
            self.graph.add_node(pattern_node_id, type='pattern', period=period)

            # Connect pattern node to corresponding seasonal component nodes
            for i in range(self.decomposition.shape[0]):
                seasonal_comp_node_id = f"S{period}_{i}"
                phase = i % period
                seasonal_id = f"seasonal_{period}"
                amplitude = np.std(self.decomposition[seasonal_id])

                # Weight based on phase of seasonal pattern
                weight = amplitude * (np.sin(2 * np.pi * phase / period) + 1) / 2  # Scaled to [0,1]

                self.graph.add_edge(pattern_node_id, seasonal_comp_node_id, weight=weight, phase=phase, type='pattern_recognition', period=period)

        # Add interval nodes for common time periods (e.g., quarters, months)
        self._add_interval_nodes()

        return self.graph

    def _add_interval_nodes(self, interval_types=None):
        """
        Add interval nodes for relevant time periods.

        Parameters:
        -----------
        interval_types : list of str
            List of interval types to add (e.g., 'quarterly', 'monthly', 'weekly')
        """
        if interval_types is None:
            interval_types = ['quarterly']  # Default to quarterly for backward compatibility

        dates = [v for k, v in nx.get_node_attributes(self.graph,'date').items() if 't_' in k]
        min_time = min(sorted(dates))
        max_time = max(sorted(dates))

        intervals = []
        if 'daily' in interval_types:
            current = min_time.replace(hour=0, minute=0, second=0, microsecond=0)
            while current <= max_time:
                interval_end = current + timedelta(days=1) - timedelta(microseconds=1)
                intervals.append((current, min(interval_end, max_time), 'daily'))
                current += timedelta(days=1)
        if 'weekly' in interval_types:
            current = min_time - timedelta(days=min_time.weekday())
            current = current.replace(hour=0, minute=0, second=0, microsecond=0)
            while current <= max_time:
                interval_end = current + timedelta(days=6, hours=23, minutes=59, seconds=59)
                intervals.append((current, min(interval_end, max_time), 'weekly'))
                current += timedelta(weeks=1)
        if 'monthly' in interval_types:
            current = min_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while current <= max_time:
                next_month = current.month % 12 + 1
                next_year = current.year + (current.month // 12)
                interval_end = datetime(next_year, next_month, 1) - timedelta(microseconds=1)
                if interval_end > max_time:
                    interval_end = max_time
                intervals.append((current, interval_end, 'monthly'))
                current = datetime(next_year, next_month, 1)
        if 'quarterly' in interval_types:
            current_year = min_time.year
            current_quarter = (min_time.month - 1) // 3
            current = datetime(current_year, current_quarter * 3 + 1, 1)
            current = current.replace(hour=0, minute=0, second=0, microsecond=0)
            while current <= max_time:
                next_quarter = current_quarter + 1
                if next_quarter > 3:
                    next_year = current_year + 1
                    next_quarter = 0
                else:
                    next_year = current_year
                interval_end = datetime(next_year if next_quarter == 0 else current_year, next_quarter * 3 + 1 if next_quarter < 3 else 1,1) - timedelta(microseconds=1)
                if interval_end > max_time:
                    interval_end = max_time
                intervals.append((current, interval_end, 'quarterly'))
                current_quarter = next_quarter
                current_year = next_year if next_quarter == 0 else current_year
                current = datetime(current_year, next_quarter * 3 + 1, 1)
        if 'yearly' in interval_types:
            current = datetime(min_time.year, 1, 1)
            current = current.replace(hour=0, minute=0, second=0, microsecond=0)
            while current <= max_time:
                interval_end = datetime(current.year + 1, 1, 1) - timedelta(microseconds=1)
                if interval_end > max_time:
                    interval_end = max_time
                intervals.append((current, interval_end, 'yearly'))
                current = datetime(current.year + 1, 1, 1)

        # Add interval nodes and edges
        for start_time, end_time, interval in intervals:
            time_points = [d for d in [v for k, v in nx.get_node_attributes(self.graph,'date').items() if 't_' in k] if start_time <= d <= end_time]
            time_points.sort()

            if len(time_points) < 2:
                return

            interval_node_id = f"I_{start_time}_{end_time}"
            self.graph.add_node(interval_node_id, type='interval', start=start_time, end=end_time, interval=interval)

            for i in range(len(time_points)):
                time_node_id = f"T_{time_points[i]}"

                # Calculate true trapezoidal weight
                if i == 0 or i == len(time_points) - 1:
                    # For end points, consider time interval to next/previous point
                    dt = 0
                    if i == 0 and len(time_points) > 1:
                        dt = time_points[1] - time_points[0]
                    elif i == len(time_points) - 1 and len(time_points) > 1:
                        dt = time_points[i] - time_points[i - 1]
                    weight = 0.5 * dt
                else:
                    # For interior points, use distance to adjacent points
                    dt_left = time_points[i] - time_points[i - 1]
                    dt_right = time_points[i + 1] - time_points[i]
                    weight = 0.5 * (dt_left + dt_right)

                self.graph.add_edge(interval_node_id, time_node_id, weight=weight, type='temporal_integration')

    def temporal_integration(self, component='trend', start_date=None, end_date=None):
        """
        Perform temporal integration of a component over a specified time interval.

        Parameters:
        -----------
        component : str
            The component to integrate ('original', 'trend', 'seasonal', 'remainder',
            or 'seasonal_{period}' for a specific seasonal component)
        start_date : datetime or str, optional
            The start date of the integration period
        end_date : datetime or str, optional
            The end date of the integration period

        Returns:
        --------
        float
            The integrated value
        """
        if self.decomposition is None:
            raise ValueError("The model must be fitted first using the fit() method")

        # Validate component
        if component not in self.decomposition.columns:
            raise ValueError(
                f"Component '{component}' not found. Available components: {list(self.decomposition.columns)}")

        # Convert string dates to datetime if necessary
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # If no dates specified, use full range
        if start_date is None:
            start_date = self.decomposition.index.min()
        if end_date is None:
            end_date = self.decomposition.index.max()

        # Filter data for the specified period
        mask = (self.decomposition.index >= start_date) & (self.decomposition.index <= end_date)
        period_data = self.decomposition.loc[mask]

        if len(period_data) == 0:
            raise ValueError(f"No data found in the specified time range: {start_date} to {end_date}")

        # Convert dates to numeric (days since start) for trapezoidal integration
        days = [(date - period_data.index[0]).total_seconds() / (24 * 3600) for date in period_data.index]
        values = period_data[component].values

        # Apply trapezoidal rule for integration
        integrated_value = np.trapz(values, days)

        return integrated_value

    def detect_hotspots(self, threshold=2.0, component='remainder'):
        """
        Detect hotspots in the KPI using graph centrality measures.

        Parameters:
        -----------
        threshold : float
            Z-score threshold for hotspot detection
        component : str
            The component to analyze for hotspots ('remainder' by default,
            can also be 'original', 'seasonal', or a specific seasonal component)

        Returns:
        --------
        pandas.DataFrame
            Detected hotspots with their characteristics
        """
        if self.graph is None:
            raise ValueError("The model must be fitted first using the fit() method")

        # Determine node type based on component
        if component == 'remainder':
            node_type = 'remainder'
        elif component == 'original':
            node_type = 'time'
        elif component == 'trend':
            node_type = 'trend'
        elif component == 'seasonal':
            node_type = 'seasonal'
        elif component.startswith('seasonal_'):
            period = int(component.split('_')[1])
            # Filter nodes by both type and period
            target_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'seasonal_component' and attr.get('period') == period]
        else:
            raise ValueError(f"Invalid component '{component}' for hotspot detection")

        # Get the target nodes
        if not component.startswith('seasonal_'):
            target_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == node_type]

        # Initialize subgraph for centrality calculation
        subgraph = nx.DiGraph()

        # Add nodes to subgraph
        for node in target_nodes:
            subgraph.add_node(node, **self.graph.nodes[node])

        # Add edges between consecutive nodes
        for i in range(len(target_nodes) - 1):
            node1 = target_nodes[i]
            node2 = target_nodes[i + 1]

            val1 = self.graph.nodes[node1]['value']
            val2 = self.graph.nodes[node2]['value']

            # Weight inversely proportional to value difference
            weight = 1.0 / (abs(val1 - val2) + 1e-10)

            subgraph.add_edge(node1, node2, weight=weight)

        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(subgraph, weight='weight', normalized=True)

        # Extract values and calculate z-scores
        values = [self.graph.nodes[n]['value'] for n in target_nodes]
        value_zscores = zscore(values)
        centrality_zscores = zscore(list(centrality.values()))

        # Identify hotspots
        hotspots = []

        for i, node in enumerate(target_nodes):
            idx = self.graph.nodes[node]['index']
            date = self.graph.nodes[node]['date']
            value = values[i]
            value_zscore = value_zscores[i]
            centrality_zscore = centrality_zscores[i]

            # Hotspot if either value or centrality exceeds threshold
            if abs(value_zscore) > threshold or abs(centrality_zscore) > threshold:
                severity = max(abs(value_zscore), abs(centrality_zscore))

                hotspot_data = {
                    'index': idx,
                    'date': date,
                    'component': component,
                    'value': value,
                    'value_zscore': value_zscore,
                    'centrality': centrality[node],
                    'centrality_zscore': centrality_zscore,
                    'severity': severity
                }

                # Add period information for seasonal components
                if component.startswith('seasonal_'):
                    hotspot_data['period'] = period

                hotspots.append(hotspot_data)

        # Convert to DataFrame and sort by severity
        if hotspots:
            columns = ['index', 'date', 'component', 'value', 'value_zscore', 'centrality', 'centrality_zscore', 'severity']
            if component.startswith('seasonal_'):
                columns.append('period')
            self.hotspots = pd.DataFrame(hotspots, columns=columns).sort_values('severity', ascending=False)
        else:
            columns = ['index', 'date', 'component', 'value', 'value_zscore', 'centrality', 'centrality_zscore', 'severity']
            if component.startswith('seasonal_'):
                columns.append('period')
            self.hotspots = pd.DataFrame(columns=columns)

        return self.hotspots

    def get_seasonal_components(self):
        """
        Get all seasonal components.

        Returns:
        --------
        dict
            Dictionary of seasonal components with keys 'seasonal_{period}'
        """
        if self.seasonal_components is None:
            raise ValueError("The model must be fitted first using the fit() method")

        return self.seasonal_components
