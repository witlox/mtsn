from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import MSTL  # Changed from STL to MSTL


class MTSN:
    """
    A class implementing the mathematical framework for network-based analysis of KPIs.
    Handles temporal dependencies, community detection, and key influencer identification.
    Uses MSTL for multiple seasonal time series decomposition.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.1, gamma: float = 0.1):
        """
        Initialize the KPI Network Analyzer.

        Parameters:
        -----------
        alpha : float
            Regularization parameter for the Frobenius norm in graph learning.
        beta : float
            Regularization parameter for the L1 norm in graph learning.
        gamma : float
            Regularization parameter for temporal consistency.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kpi_data = None
        self.decomposed_data = {}
        self.graph_series = {}
        self.adjacency_matrices = {}
        self.laplacian_matrices = {}
        self.communities = {}
        self.centrality_measures = {}

    def load_data(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
        kpi_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Load KPI time series data.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing KPI time series data.
        date_column : str
            Name of the column containing dates.
        kpi_columns : List[str], optional
            List of column names representing KPIs. If None, all columns except date_column are used.
        """
        if kpi_columns is None:
            kpi_columns = [col for col in data.columns if col != date_column]

        # Set date as index and extract only KPI columns
        self.kpi_data = data.set_index(date_column)[kpi_columns].copy()
        self.kpi_names = kpi_columns
        self.n_kpis = len(kpi_columns)

        print(
            f"Loaded data with {self.n_kpis} KPIs and {len(self.kpi_data)} time points."
        )

    def decompose_time_series(self, periods: List[int] = [12]) -> Dict:
        """
        Apply Multiple Seasonal-Trend decomposition using Loess (MSTL) to each KPI time series.

        Parameters:
        -----------
        periods : List[int]
            List of seasonal periods to consider. For example, [7, 30] for weekly and monthly seasonality.
        robust : bool
            Flag indicating whether to use robust fitting.

        Returns:
        --------
        Dict : Dictionary containing decomposed components for each KPI.
        """
        self.decomposed_data = {}

        for kpi in self.kpi_names:
            if self.kpi_data[kpi].isnull().sum() > 0:
                # Handle missing values with simple interpolation for decomposition
                series = self.kpi_data[kpi].interpolate(method="linear")
            else:
                series = self.kpi_data[kpi]

            # Apply MSTL decomposition
            mstl = MSTL(series, periods=periods)
            result = mstl.fit()

            # Extract trend and residual components
            trend = result.trend
            residual = result.resid

            # For MSTL, the seasonal component is a DataFrame with multiple columns for each seasonal period
            # We'll store both combined and individual seasonal components
            seasonal_combined = result.seasonal.sum(axis=1)

            self.decomposed_data[kpi] = {
                "trend": trend,
                "seasonal": seasonal_combined,  # Combined seasonal components
                "seasonal_components": result.seasonal,  # Individual seasonal components
                "residual": residual,
                "original": series,
            }

        print(
            f"Multi-seasonal time series decomposition completed for {len(self.decomposed_data)} KPIs."
        )
        return self.decomposed_data

    def compute_seasonal_similarity(self) -> pd.DataFrame:
        """
        Compute the seasonal similarity matrix between KPIs.

        Returns:
        --------
        pd.DataFrame : Seasonal similarity matrix.
        """
        if not self.decomposed_data:
            raise ValueError("Time series decomposition must be performed first.")

        seasonal_sim = pd.DataFrame(index=self.kpi_names, columns=self.kpi_names)

        for kpi1 in self.kpi_names:
            for kpi2 in self.kpi_names:
                seasonal1 = self.decomposed_data[kpi1]["seasonal"]
                seasonal2 = self.decomposed_data[kpi2]["seasonal"]

                # Calculate Pearson correlation between seasonal components
                corr = np.corrcoef(seasonal1, seasonal2)[0, 1]
                seasonal_sim.loc[kpi1, kpi2] = corr

        return seasonal_sim

    def _objective_function(
        self, A_flat: np.ndarray, X: np.ndarray, A_prev: Optional[np.ndarray] = None
    ) -> float:
        """
        Objective function for graph learning optimization.

        Parameters:
        -----------
        A_flat : np.ndarray
            Flattened adjacency matrix.
        X : np.ndarray
            KPI measurements matrix.
        A_prev : np.ndarray, optional
            Previous adjacency matrix for temporal consistency.

        Returns:
        --------
        float : Value of the objective function.
        """
        n = self.n_kpis
        # Reshape the flattened array back to a matrix, ensuring zeros on diagonal
        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = A_flat[idx]
                    idx += 1

        # Reconstruction term
        recon_error = np.linalg.norm(X - A @ X, "fro") ** 2

        # Regularization terms
        frob_reg = self.alpha * np.linalg.norm(A, "fro") ** 2
        l1_reg = self.beta * np.sum(np.abs(A))

        # Temporal consistency term
        temporal_reg = 0
        if A_prev is not None:
            temporal_reg = self.gamma * np.linalg.norm(A - A_prev, "fro") ** 2

        return recon_error + frob_reg + l1_reg + temporal_reg

    def _objective_gradient(
        self, A_flat: np.ndarray, X: np.ndarray, A_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Gradient of objective function for optimization.

        Parameters:
        -----------
        A_flat : np.ndarray
            Flattened adjacency matrix.
        X : np.ndarray
            KPI measurements matrix.
        A_prev : np.ndarray, optional
            Previous adjacency matrix for temporal consistency.

        Returns:
        --------
        np.ndarray : Gradient of the objective function.
        """
        n = self.n_kpis
        # Reshape the flattened array back to a matrix, ensuring zeros on diagonal
        A = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = A_flat[idx]
                    idx += 1

        # Gradient of reconstruction term
        recon_grad = -2 * ((X - A @ X) @ X.T)

        # Gradient of Frobenius norm
        frob_grad = 2 * self.alpha * A

        # Gradient of L1 norm (subgradient)
        l1_grad = self.beta * np.sign(A)

        # Gradient of temporal consistency
        temporal_grad = np.zeros_like(A)
        if A_prev is not None:
            temporal_grad = 2 * self.gamma * (A - A_prev)

        # Sum all gradients
        grad = recon_grad + frob_grad + l1_grad + temporal_grad

        # Flatten gradient, excluding diagonal elements
        grad_flat = np.zeros(n * (n - 1))
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    grad_flat[idx] = grad[i, j]
                    idx += 1

        return grad_flat

    def learn_graph_structure(
        self, time_windows: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Learn the graph structure from KPI data for specified time windows.

        Parameters:
        -----------
        time_windows : List[Tuple[str, str]], optional
            List of time window tuples (start_date, end_date). If None, uses the entire data.

        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary mapping time window labels to adjacency matrices.
        """
        if self.kpi_data is None:
            raise ValueError("Data must be loaded first.")

        # If no time windows provided, use the entire dataset
        if time_windows is None:
            time_windows = [
                ("all", self.kpi_data.index.min(), self.kpi_data.index.max())
            ]

        self.adjacency_matrices = {}
        self.laplacian_matrices = {}
        self.graph_series = {}

        A_prev = None  # Initial previous adjacency matrix

        for window_label, start_date, end_date in time_windows:
            # Extract data for the current time window
            window_data = self.kpi_data.loc[start_date:end_date]
            X = window_data.values.T  # Transpose to get [n_kpis x n_observations]

            n = self.n_kpis
            n_flat = n * (n - 1)  # Number of non-diagonal elements

            # Initial guess for optimization (flattened non-diagonal elements)
            A_init = np.zeros(n_flat)

            # Define constraints to ensure non-negativity
            constraints = [{"type": "ineq", "fun": lambda x: x}]  # A_ij >= 0

            # Optimize the objective function
            result = minimize(
                fun=self._objective_function,
                x0=A_init,
                args=(X, A_prev),
                jac=self._objective_gradient,
                constraints=constraints,
                method="SLSQP",
                options={"maxiter": 200, "disp": True},
            )

            # Reshape the optimized parameters to get the adjacency matrix
            A_opt = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        A_opt[i, j] = max(0, result.x[idx])  # Ensure non-negativity
                        idx += 1

            # Compute the Laplacian matrix
            D_opt = np.diag(np.sum(A_opt, axis=1))
            L_opt = D_opt - A_opt

            # Store the results
            self.adjacency_matrices[window_label] = A_opt
            self.laplacian_matrices[window_label] = L_opt

            # Create NetworkX graph
            G = nx.DiGraph()
            for i, kpi1 in enumerate(self.kpi_names):
                G.add_node(kpi1)
                for j, kpi2 in enumerate(self.kpi_names):
                    if i != j and A_opt[i, j] > 0:
                        G.add_edge(kpi1, kpi2, weight=A_opt[i, j])

            self.graph_series[window_label] = G

            # Update previous adjacency matrix for temporal consistency
            A_prev = A_opt

        print(
            f"Graph structure learning completed for {len(self.graph_series)} time windows."
        )
        return self.adjacency_matrices

    def detect_communities(
        self, method: str = "walktrap", n_communities: Optional[int] = None
    ) -> Dict:
        """
        Detect communities in KPI networks.

        Parameters:
        -----------
        method : str
            Method for community detection ('walktrap', 'louvain', or 'spectral').
        n_communities : int, optional
            Number of communities for spectral clustering. Only used if method='spectral'.

        Returns:
        --------
        Dict : Dictionary containing community assignments for each time window.
        """
        if not self.graph_series:
            raise ValueError("Graph structure must be learned first.")

        self.communities = {}

        for window_label, G in self.graph_series.items():
            if method == "walktrap":
                # For walktrap, convert to undirected graph with weights
                G_undir = G.to_undirected()
                # Use community_louvain as an approximation to walktrap
                partition = community_louvain.best_partition(G_undir)

            elif method == "louvain":
                # Convert to undirected graph with weights
                G_undir = G.to_undirected()
                partition = community_louvain.best_partition(G_undir)

            elif method == "spectral":
                if n_communities is None:
                    raise ValueError(
                        "n_communities must be specified for spectral clustering."
                    )

                # Get Laplacian matrix
                L = self.laplacian_matrices[window_label]

                # Compute eigenvectors corresponding to the smallest eigenvalues
                # Fixed: Using subset_by_index instead of eigvals
                eigenvalues, eigenvectors = eigh(
                    L, subset_by_index=(0, n_communities - 1)
                )

                # Skip first eigenvector (constant vector)
                embedding = eigenvectors[:, 1:n_communities]

                # Apply K-means clustering
                kmeans = KMeans(n_clusters=n_communities, random_state=42)
                labels = kmeans.fit_predict(embedding)

                # Create partition dictionary
                partition = {kpi: label for kpi, label in zip(self.kpi_names, labels)}

            else:
                raise ValueError(f"Unsupported community detection method: {method}")

            self.communities[window_label] = partition

        print(
            f"Community detection completed for {len(self.communities)} time windows."
        )
        return self.communities

    def compute_centrality_measures(self) -> Dict:
        """
        Compute various centrality measures for KPIs.

        Returns:
        --------
        Dict : Dictionary containing centrality measures for each KPI.
        """
        if not self.graph_series:
            raise ValueError("Graph structure must be learned first.")

        self.centrality_measures = {}

        for window_label, G in self.graph_series.items():
            # Compute various centrality measures
            degree_centrality = nx.degree_centrality(G)
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight="weight")

            # For betweenness centrality, create undirected graph if needed
            try:
                betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
            except:
                G_undir = G.to_undirected()
                betweenness_centrality = nx.betweenness_centrality(
                    G_undir, weight="weight"
                )

            self.centrality_measures[window_label] = {
                "degree": degree_centrality,
                "in_degree": in_degree_centrality,
                "out_degree": out_degree_centrality,
                "eigenvector": eigenvector_centrality,
                "betweenness": betweenness_centrality,
            }

        print(
            f"Centrality measures computed for {len(self.centrality_measures)} time windows."
        )
        return self.centrality_measures

    def identify_key_influencers(
        self, centrality_type: str = "eigenvector", top_n: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Identify key influencer KPIs based on centrality measures.

        Parameters:
        -----------
        centrality_type : str
            Type of centrality to use ('degree', 'in_degree', 'out_degree', 'eigenvector', 'betweenness').
        top_n : int
            Number of top influencers to return.

        Returns:
        --------
        Dict[str, List[Tuple[str, float]]] : Dictionary mapping time windows to lists of (KPI, centrality) tuples.
        """
        if not self.centrality_measures:
            self.compute_centrality_measures()

        key_influencers = {}

        for window_label, centrality_dict in self.centrality_measures.items():
            if centrality_type not in centrality_dict:
                raise ValueError(f"Centrality type {centrality_type} not available.")

            # Sort KPIs by centrality value
            sorted_kpis = sorted(
                centrality_dict[centrality_type].items(),
                key=lambda x: x[1],
                reverse=True,
            )

            # Return top_n KPIs
            key_influencers[window_label] = sorted_kpis[:top_n]

        return key_influencers

    def predict_missing_values(
        self, data_with_missing: pd.DataFrame, lambda_reg: float = 0.1
    ) -> pd.DataFrame:
        """
        Predict missing values in KPI data using graph-based optimization.

        Parameters:
        -----------
        data_with_missing : pd.DataFrame
            DataFrame containing KPI data with missing values.
        lambda_reg : float
            Regularization parameter for graph smoothness.

        Returns:
        --------
        pd.DataFrame : DataFrame with predicted values for missing entries.
        """
        if not self.laplacian_matrices:
            raise ValueError("Graph structure must be learned first.")

        # Use the most recent Laplacian matrix
        window_label = list(self.laplacian_matrices.keys())[-1]
        L = self.laplacian_matrices[window_label]

        # Create a mask of observed values (1 for observed, 0 for missing)
        mask = ~data_with_missing.isna()
        mask_values = mask.values

        # Initialize with mean imputation
        X_init = data_with_missing.copy()
        for col in X_init.columns:
            X_init[col].fillna(X_init[col].mean(), inplace=True)

        X_values = X_init.values

        # Define the objective function for missing value optimization
        def obj_func(x_flat):
            # Reshape the flattened array
            X_pred = X_values.copy()
            X_pred[~mask_values] = x_flat

            # Reconstruction error term (only for observed values)
            recon_error = np.sum((mask_values * (X_values - X_pred)) ** 2)

            # Graph smoothness term
            smoothness = lambda_reg * np.trace(X_pred.T @ L @ X_pred)

            return recon_error + smoothness

        # Get initial values for missing entries
        x0 = X_values[~mask_values]

        # Optimize
        result = minimize(
            fun=obj_func, x0=x0, method="L-BFGS-B", options={"maxiter": 100}
        )

        # Update the missing values with optimized values
        X_pred = X_values.copy()
        X_pred[~mask_values] = result.x

        # Return as DataFrame
        predicted_df = pd.DataFrame(
            X_pred, index=data_with_missing.index, columns=data_with_missing.columns
        )

        print(f"Missing value prediction completed.")
        return predicted_df

    def run_full_analysis(
        self,
        periods: List[int] = [12],
        n_communities: int = 3,
        centrality_type: str = "eigenvector",
    ) -> Dict:
        """
        Run the complete KPI network analysis pipeline.

        Parameters:
        -----------
        periods : List[int]
            List of seasonal periods for multi-seasonal time series decomposition.
        n_communities : int
            Number of communities to detect.
        centrality_type : str
            Type of centrality to use for key influencer identification.

        Returns:
        --------
        Dict : Summary of analysis results.
        """
        if self.kpi_data is None:
            raise ValueError("Data must be loaded first.")

        print("Starting full KPI network analysis...")

        # Step 1: Time series decomposition
        print("Step 1: Performing multi-seasonal time series decomposition...")
        self.decompose_time_series(periods=periods)

        # Step 2: Learn graph structure
        print("Step 2: Learning graph structure...")
        self.learn_graph_structure()

        # Step 3: Detect communities
        print("Step 3: Detecting communities...")
        self.detect_communities(method="spectral", n_communities=n_communities)

        # Step 4: Compute centrality measures
        print("Step 4: Computing centrality measures...")
        self.compute_centrality_measures()

        # Step 5: Identify key influencers
        print("Step 5: Identifying key influencers...")
        key_influencers = self.identify_key_influencers(centrality_type=centrality_type)

        # Prepare summary of results
        results_summary = {
            "n_kpis": self.n_kpis,
            "time_periods": len(self.kpi_data),
            "periods_analyzed": periods,
            "communities": self.communities,
            "key_influencers": key_influencers,
        }

        print("KPI network analysis completed successfully.")
        return results_summary
