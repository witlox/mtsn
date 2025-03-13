import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import zscore

from stl import STLTemporalIntegration

warnings.filterwarnings("ignore")

class MTSN:
    """
    Multi-Layer Temporal-Seasonal Network (MTSN) for KPI analysis with hotspot detection.
    """

    def __init__(self, data=None, date_column=None, value_column=None, period=None, hotspot_threshold=2.5, autocorr_threshold=0.3):
        """
        Initialize the MTSN framework.

        Parameters:
        -----------
        data : pandas DataFrame
            DataFrame containing time series data
        date_column : str
            Name of the column containing datetime values
        value_column : str
            Name of the column containing KPI values
        period : int
            The seasonal period (e.g., 7 for weekly, 12 for monthly)
        hotspot_threshold : float
            Z-score threshold for hotspot detection
        autocorr_threshold : float
            Threshold for significant seasonal autocorrelation
        """
        self.df = None
        self.date_col = date_column
        self.value_col = value_column
        self.period = period
        self.hotspot_threshold = hotspot_threshold
        self.autocorr_threshold = autocorr_threshold

        # Graph components
        self.graph = None
        self.decomposition = None

        # Component series
        self.trend = None
        self.seasonal = None
        self.remainder = None

        # Hotspot information
        self.hotspots = None

        # Comparison results
        self.comparison_results = None
        self.hotspot_evaluation = None
        self.integration_comparison = None

        # If data is provided, load it
        if data is not None and date_column is not None and value_column is not None:
            self.load_data(data, date_column, value_column)

    def load_data(self, data, date_column, value_column):
        """
        Load time series data into the MTSN framework.

        Parameters:
        -----------
        data : pandas DataFrame
            DataFrame containing time series data
        date_column : str
            Name of the column containing datetime values
        value_column : str
            Name of the column containing KPI values
        """
        self.df = data.copy()
        self.date_col = date_column
        self.value_col = value_column

        # Ensure date column is datetime type
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        # Sort by date
        self.df = self.df.sort_values(by=self.date_col).reset_index(drop=True)

        # Set date as index
        self.df.set_index(self.date_col, inplace=True)

        # Infer frequency if period is not provided
        if self.period is None:
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
                print(f"Warning: Could not infer seasonality period from frequency. Defaulting to 7.")

        print(f"Loaded {len(self.df)} observations with seasonal period {self.period}")
        return self

    def decompose(self, seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False):
        """
        Perform STL decomposition on the time series data.

        Parameters:
        -----------
        seasonal_deg : int
            Degree of seasonal LOESS
        trend_deg : int
            Degree of trend LOESS
        low_pass_deg : int
            Degree of low-pass LOESS
        robust : bool
            Whether to use robust fitting
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Handle missing values if any
        if self.df[self.value_col].isnull().any():
            print("Warning: Missing values detected. Interpolating...")
            self.df[self.value_col] = self.df[self.value_col].interpolate(method='linear')

        print(f"loading date from {self.date_col} and value from {self.value_col} from dataframe with headers {self.df.columns}")

        # Perform STL decomposition
        stl = STLTemporalIntegration(self.df, self.date_col, self.value_col)
        stl.decompose(
                  seasonal_deg=seasonal_deg,
                  trend_deg=trend_deg,
                  low_pass_deg=low_pass_deg,
                  robust=robust)

        self.decomposition = stl.decomposition
        # Reset index to get date as a column
        self.decomposition.reset_index()
        print(f"Decomposition completed.")


    def construct_graph(self):
        """
        Construct the Multi-Layer Temporal-Seasonal Network from decomposed data.
        """
        if self.decomposition is None:
            raise ValueError("Decomposition has not been performed. Call decompose() first.")

        self.trend = self.decomposition['trend'].values
        self.seasonal = self.decomposition['seasonal'].values
        self.remainder = self.decomposition['remainder'].values

        # Initialize a directed graph
        graph = nx.DiGraph()

        # Add time point nodes
        for i, (date, row) in enumerate(self.decomposition.iterrows()):
            # Add time point node
            time_node = f"t_{i}"
            graph.add_node(time_node, type='time', date=date, value=row['original'])

            # Add component nodes
            trend_node = f"T_{i}"
            seasonal_node = f"S_{i}"
            remainder_node = f"R_{i}"

            graph.add_node(trend_node, type='trend', value=row['trend'])
            graph.add_node(seasonal_node, type='seasonal', value=row['seasonal'])
            graph.add_node(remainder_node, type='remainder', value=row['remainder'])

            # Add decomposition edges
            graph.add_edge(time_node, trend_node, weight=1, type='decomposition')
            graph.add_edge(time_node, seasonal_node, weight=1, type='decomposition')
            graph.add_edge(time_node, remainder_node, weight=1, type='decomposition')

            # Add temporal progression edge (except for the last point)
            if i < self.decomposition.shape[0] - 1:
                next_date = self.decomposition.iloc[i+1].name
                delta_t = (next_date - date).total_seconds() / (24*3600)  # in days
                graph.add_edge(f"t_{i}", f"t_{i+1}", weight=delta_t, type='temporal')

        # Add seasonal cycle edges
        for i in range(self.decomposition.shape[0] - self.period):
            # Calculate seasonal autocorrelation
            seasonal_values = self.decomposition['seasonal'].values
            if i + self.period < len(seasonal_values):
                # Get windows around the current position and the position one period later
                subset1 = seasonal_values[max(0, i-self.period//2):min(len(seasonal_values), i+self.period//2)]
                subset2 = seasonal_values[max(0, i+self.period-self.period//2):min(len(seasonal_values), i+self.period+self.period//2)]

                # Ensure both arrays have the same length
                min_length = min(len(subset1), len(subset2))

                if min_length > 1:
                    # Calculate correlation using equally sized arrays
                    corr = np.corrcoef(subset1[:min_length], subset2[:min_length])[0, 1]

                    # Add edge if correlation is significant
                    if not np.isnan(corr) and abs(corr) > self.autocorr_threshold:
                        graph.add_edge(f"S_{i}", f"S_{i+self.period}", weight=abs(corr), type='seasonal_cycle')

        # Add pattern node for the primary seasonal pattern
        pattern_node = f"P_{self.period}"
        graph.add_node(pattern_node, type='pattern', period=self.period)

        # Connect pattern node to seasonal components
        for i in range(self.decomposition.shape[0]):
            phase = i % self.period
            amplitude = np.std(self.decomposition['seasonal'])
            weight = amplitude * np.sin(2 * np.pi * phase / self.period)
            graph.add_edge(pattern_node, f"S_{i}", weight=abs(weight), type='pattern')

        # Add interval nodes for quarters
        dates = self.decomposition.index
        min_date, max_date = dates.min(), dates.max()

        # Create quarterly intervals
        current = pd.Timestamp(min_date.year, min_date.month, 1)
        quarter_end_dates = pd.date_range(start=current, end=max_date, freq='Q')

        for q_idx, quarter_end_date in enumerate(quarter_end_dates):
            if q_idx == 0:
                quarter_start = min_date
            else:
                quarter_start = quarter_end_dates[q_idx-1] + pd.Timedelta(days=1)

            if quarter_end_date > max_date:
                quarter_end_date = max_date

            # Create interval node
            interval_name = f"I_{quarter_start.strftime('%Y-%m-%d')}_{quarter_end_date.strftime('%Y-%m-%d')}"
            graph.add_node(interval_name, type='interval', start_date=quarter_start, end_date=quarter_end_date)

            # Connect to all encompassed time points with integration weights
            for i, date in enumerate(dates):
                if quarter_start <= date <= quarter_end_date:
                    # Use trapezoidal weighting
                    if date == quarter_start or date == quarter_end_date:
                        weight = 0.5
                    else:
                        weight = 1.0

                    # Get time delta in days
                    if i < len(dates) - 1:
                        next_date = dates[i+1]
                        delta_t = (next_date - date).total_seconds() / (24*3600)
                    else:
                        # For the last point, use the previous delta
                        prev_date = dates[i-1]
                        delta_t = (date - prev_date).total_seconds() / (24*3600)

                    graph.add_edge(interval_name, f"t_{i}", weight=weight * delta_t, type='integration')

        self.graph = graph
        print(f"Graph constructed with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

    def detect_hotspots(self):
        """
        Detect hotspots in the KPI data based on remainder component centrality.
        """
        if self.graph is None:
            raise ValueError("Graph has not been constructed. Call construct_graph() first.")

        # Create a subgraph of remainder nodes and their connections
        remainder_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'remainder']

        # Create a separate graph for centrality calculation
        centrality_graph = nx.Graph()

        # Add remainder nodes
        for node in remainder_nodes:
            centrality_graph.add_node(node, value=self.graph.nodes[node]['value'])

        # Connect remainder nodes that are adjacent in time
        for i in range(len(remainder_nodes) - 1):
            node1 = remainder_nodes[i]
            node2 = remainder_nodes[i + 1]

            # Calculate weight based on similarity (inverse of absolute difference)
            val1 = self.graph.nodes[node1]['value']
            val2 = self.graph.nodes[node2]['value']

            # Add small constant to avoid division by zero
            weight = 1.0 / (abs(val1 - val2) + 0.0001)

            centrality_graph.add_edge(node1, node2, weight=weight)

        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(centrality_graph, weight='weight')

        # Calculate z-scores of remainder values
        remainder_values = np.array([self.graph.nodes[n]['value'] for n in remainder_nodes])
        remainder_zscore = zscore(remainder_values)

        # Calculate z-scores of centrality values
        centrality_values = np.array([centrality[n] for n in remainder_nodes])
        centrality_zscore = zscore(centrality_values)

        # Identify hotspots where both remainder and centrality z-scores exceed threshold
        hotspots = []
        for i, node in enumerate(remainder_nodes):
            node_idx = int(node.split('_')[1])  # Extract index from node name

            if abs(remainder_zscore[i]) > self.hotspot_threshold or abs(centrality_zscore[i]) > self.hotspot_threshold:
                hotspots.append({
                    'index': node_idx,
                    'date': self.decomposition.index[i],
                    'remainder_value': remainder_values[i],
                    'remainder_zscore': remainder_zscore[i],
                    'centrality': centrality_values[i],
                    'centrality_zscore': centrality_zscore[i],
                    'severity': max(abs(remainder_zscore[i]), abs(centrality_zscore[i]))
                })

        # Sort hotspots by severity
        self.hotspots = sorted(hotspots, key=lambda x: x['severity'], reverse=True)

        print(f"Detected {len(self.hotspots)} hotspots.")

    def calculate_temporal_integration(self, start_date=None, end_date=None, component='trend'):
        """
        Calculate the temporal integration of a component over a specified interval.

        Parameters:
        -----------
        start_date : datetime
            Start date for integration
        end_date : datetime
            End date for integration
        component : str
            Component to integrate ('trend', 'original', 'seasonal', 'remainder')

        Returns:
        --------
        float
            The value of the temporal integration
        """
        if self.decomposition is None:
            raise ValueError("Decomposition has not been performed. Call decompose() first.")

        # Default to full range if dates not provided
        if start_date is None:
            start_date = self.decomposition[self.date_col].min()
        if end_date is None:
            end_date = self.decomposition[self.date_col].max()

        # Convert to pandas datetime if not already
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter by date range
        if self.decomposition.index.name == self.date_col:
            mask = (self.decomposition.index >= start_date) & (self.decomposition.index <= end_date)
        else:
            mask = (self.decomposition[self.date_col] >= start_date) & (self.decomposition[self.date_col] <= end_date)
        subset = self.decomposition[mask]

        if len(subset) == 0:
            raise ValueError(f"No data found between {start_date} and {end_date}")

        # Get time deltas in days
        if self.decomposition.index.name == self.date_col:
            time_deltas = [(t - subset.index[0]).total_seconds() / (24 * 3600) for t in subset.index]
        else:
            time_deltas = [(t - subset[self.date_col].iloc[0]).total_seconds() / (24*3600) for t in subset[self.date_col]]

        # Perform trapezoidal integration
        return trapezoid(subset[component], time_deltas)

    def get_subgraph_for_interval(self, start_date, end_date):
        """
        Extract a subgraph corresponding to a specific time interval.

        Parameters:
        -----------
        start_date : datetime
            Start date of the interval
        end_date : datetime
            End date of the interval

        Returns:
        --------
        networkx.DiGraph
            Subgraph for the specified interval
        """
        if self.graph is None:
            raise ValueError("Graph has not been constructed. Call construct_graph() first.")

        # Convert to pandas datetime if not already
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Find nodes within the interval
        time_nodes = []
        for node, attr in self.graph.nodes(data=True):
            if attr.get('type') == 'time' and 'date' in attr:
                if start_date <= attr['date'] <= end_date:
                    time_nodes.append(node)

        # Get component nodes for these time nodes
        component_nodes = []
        for time_node in time_nodes:
            for neighbor in self.graph.successors(time_node):
                component_nodes.append(neighbor)

        # Get interval nodes that overlap with the specified interval
        interval_nodes = []
        for node, attr in self.graph.nodes(data=True):
            if attr.get('type') == 'interval' and 'start_date' in attr and 'end_date' in attr:
                if attr['start_date'] <= end_date and attr['end_date'] >= start_date:
                    interval_nodes.append(node)

        # Get pattern nodes
        pattern_nodes = [node for node, attr in self.graph.nodes(data=True) if attr.get('type') == 'pattern']

        # Combine all nodes
        all_nodes = time_nodes + component_nodes + interval_nodes + pattern_nodes

        # Create subgraph
        return self.graph.subgraph(all_nodes)
