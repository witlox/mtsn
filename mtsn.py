import logging

from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import MSTL


class MTSN:
    """
    A class implementing the mathematical framework for network-based analysis of KPIs.
    Handles temporal dependencies, community detection, and key influencer identification.
    Uses MSTL for multiple seasonal time series decomposition and advanced graph learning techniques.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.1,
        gamma: float = 0.1,
        residual_alpha: float = 0.2,
        adaptive_residual: bool = True,
    ):
        """
        Initialize the KPI Network Analyzer with graph-specific optimization parameters.

        Parameters:
        -----------
        alpha : float
            Regularization parameter for the Frobenius norm in graph learning.
        beta : float
            Regularization parameter for the L1 norm in graph learning.
        gamma : float
            Regularization parameter for temporal consistency.
        residual_alpha : float
            Interpolation parameter for residual aggregation operators.
        adaptive_residual : bool
            Whether to use adaptive residual connections.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.residual_alpha = residual_alpha
        self.adaptive_residual = adaptive_residual
        self.kpi_data = None
        self.decomposed_data = {}
        self.graph_series = {}
        self.adjacency_matrices = {}
        self.laplacian_matrices = {}
        self.communities = {}
        self.centrality_measures = {}
        self.aggregation_values = {}

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
        if date_column not in data.columns:
            self.kpi_data = data[kpi_columns].copy()
        else:
            self.kpi_data = data.set_index(date_column)[kpi_columns].copy()
        self.kpi_names = kpi_columns
        self.n_kpis = len(kpi_columns)

        logging.info(
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
            if self.kpi_data[kpi].isnull().sum() > 0 or self.kpi_data[kpi].hasnans:
                # Handle missing values with simple interpolation for decomposition
                series = self.kpi_data[kpi].interpolate(method="polynomial", order=2)
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

        logging.info(
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

    def _virgo_initialization(
        self, fan_in: int, fan_out: int, graph_structure: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Implements Virgo initialization for graph neural networks to reduce variance instability.

        Parameters:
        -----------
        fan_in : int
            Input dimension
        fan_out : int
            Output dimension
        graph_structure : np.ndarray, optional
            Adjacency matrix to inform initialization

        Returns:
        --------
        np.ndarray : Initialized weights
        """
        if graph_structure is not None:
            # Compute spectral properties of the graph
            try:
                # Compute largest eigenvalue of the graph Laplacian
                L = np.diag(np.sum(graph_structure, axis=1)) - graph_structure
                eigenvalues = eigh(L, subset_by_index=(L.shape[0] - 1, L.shape[0] - 1))[
                    0
                ]
                largest_eigenvalue = max(eigenvalues[0], 1.0)  # Ensure positive
                scaling = np.sqrt(2.0 / (fan_in * largest_eigenvalue))
            except Exception as e:
                logging.exception(e)
                scaling = np.sqrt(2.0 / fan_in)  # Fallback to He initialization
        else:
            scaling = np.sqrt(2.0 / fan_in)  # Default to He initialization

        # Initialize with this scaling factor
        return np.random.normal(0, scaling, size=(fan_in, fan_out))

    def _compute_aggregation_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute aggregation values for each node/KPI for adaptive residual.

        Parameters:
        -----------
        X : np.ndarray
            KPI measurements matrix

        Returns:
        --------
        np.ndarray : Aggregation values for each KPI
        """
        n = X.shape[0]
        agg_values = np.zeros(n)

        # Compute variance of each KPI
        for i in range(n):
            # Normalize the values
            normalized = (X[i] - np.mean(X[i])) / (np.std(X[i]) + 1e-8)
            # Compute aggregation value as variance of normalized values
            agg_values[i] = np.var(normalized)

        # Scale to range [0, 1]
        if np.max(agg_values) - np.min(agg_values) > 0:
            agg_values = (agg_values - np.min(agg_values)) / (
                np.max(agg_values) - np.min(agg_values)
            )
        else:
            agg_values = np.ones_like(agg_values) * 0.5

        return agg_values

    def _objective_function(
        self,
        A_flat: np.ndarray,
        X: np.ndarray,
        A_prev: Optional[np.ndarray] = None,
        lambda_structure: float = 0.1,
    ) -> float:
        """
        Enhanced objective function for graph learning with structural properties.

        Parameters:
        -----------
        A_flat : np.ndarray
            Flattened adjacency matrix.
        X : np.ndarray
            KPI measurements matrix.
        A_prev : np.ndarray, optional
            Previous adjacency matrix for temporal consistency.
        lambda_structure : float
            Weight for structure regularization term.

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

        # Add structural term to encourage community structure
        # Compute the normalized graph Laplacian trace
        D = np.diag(
            np.sum(A, axis=1) + 1e-10
        )  # Add small epsilon to avoid division by zero
        D_diag = np.diag(D) + 1e-10
        # Ensure values are non-negative before sqrt
        D_diag_safe = np.maximum(D_diag, 0)
        # Compute sqrt only for positive values
        D_sqrt_inv_diag = np.zeros_like(D_diag)
        mask = D_diag_safe > 0
        D_sqrt_inv_diag[mask] = 1.0 / np.sqrt(D_diag_safe[mask])
        D_sqrt_inv = np.diag(D_sqrt_inv_diag)

        L_normalized = np.eye(n) - D_sqrt_inv @ A @ D_sqrt_inv
        structure_reg = lambda_structure * np.trace(L_normalized)

        return recon_error + frob_reg + l1_reg + temporal_reg + structure_reg

    def _objective_gradient(
        self,
        A_flat: np.ndarray,
        X: np.ndarray,
        A_prev: Optional[np.ndarray] = None,
        lambda_structure: float = 0.1,
    ) -> np.ndarray:
        """
        Gradient of enhanced objective function with structural regularization.

        Parameters:
        -----------
        A_flat : np.ndarray
            Flattened adjacency matrix.
        X : np.ndarray
            KPI measurements matrix.
        A_prev : np.ndarray, optional
            Previous adjacency matrix for temporal consistency.
        lambda_structure : float
            Weight for structure regularization term.

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

        # Gradient of structural term
        # Compute degree matrix and its inverse square root
        D = np.diag(np.sum(A, axis=1) + 1e-10)

        # Enhanced diagonal normalization with numerical safeguards
        d = np.diag(D)

        # Multi-stage stabilization process
        d_safe = np.nan_to_num(d, nan=1e-10)  # Handle NaN values
        d_safe = np.maximum(d_safe, 1e-10)  # Enforce minimum positive threshold
        d_safe = np.where(d_safe > 1e8, 1e8, d_safe)  # Prevent overflow in reciprocal

        D_sqrt_inv = np.diag(1.0 / np.sqrt(d_safe))

        # Compute gradient of normalized Laplacian trace w.r.t. A
        D_grad = np.zeros_like(A)
        for i in range(n):
            D_grad[:, i] = 1.0

        D_sqrt_inv_grad = np.zeros_like(D_sqrt_inv)
        for i in range(n):
            if D[i, i] > 1e-10:
                D_sqrt_inv_grad[i, i] = -0.5 * (D[i, i] ** (-1.5))

        L_normalized_grad = (
            -D_sqrt_inv_grad @ A @ D_sqrt_inv
            - D_sqrt_inv @ A @ D_sqrt_inv_grad
            - D_sqrt_inv @ D_sqrt_inv
        )
        structure_grad = lambda_structure * L_normalized_grad

        # Sum all gradients
        grad = recon_grad + frob_grad + l1_grad + temporal_grad + structure_grad

        # Flatten gradient, excluding diagonal elements
        grad_flat = np.zeros(n * (n - 1))
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    grad_flat[idx] = grad[i, j]
                    idx += 1

        return grad_flat

    def _apply_residual_aggregation(self, A: np.ndarray) -> np.ndarray:
        """
        Apply residual aggregation operator to adjacency matrix.

        Parameters:
        -----------
        A : np.ndarray
            Adjacency matrix

        Returns:
        --------
        np.ndarray : Residual aggregation matrix
        """
        n = A.shape[0]

        if self.adaptive_residual:
            # Compute adaptive residual based on aggregation values
            residual_matrix = np.zeros_like(A)
            for i in range(n):
                # Compute local deviation for this node
                local_features = A[i] @ self.current_features
                feature_diff = np.linalg.norm(local_features - self.current_features[i])

                # Adaptive beta based on feature difference
                beta = 1.0 / (1.0 + np.exp(-feature_diff + 5.0))

                # Apply adaptive residual: (1-beta)*I + beta*A
                for j in range(n):
                    if i == j:
                        residual_matrix[i, j] = 1 - beta
                    else:
                        residual_matrix[i, j] = beta * A[i, j]
        else:
            # Apply fixed interpolation: (1-alpha)*I + alpha*A
            residual_matrix = (1 - self.residual_alpha) * np.eye(
                n
            ) + self.residual_alpha * A

        return residual_matrix

    def learn_graph_structure(
        self,
        time_windows: Optional[List[Tuple[str, str]]] = None,
        lambda_structure: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """
        Learn the graph structure from KPI data for specified time windows with optimized methods.

        Parameters:
        -----------
        time_windows : List[Tuple[str, str]], optional
            List of time window tuples (start_date, end_date). If None, uses the entire data.
        lambda_structure : float
            Weight for structure regularization term.

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
        self.aggregation_values = {}

        A_prev = None  # Initial previous adjacency matrix

        for window_label, start_date, end_date in time_windows:
            # Extract data for the current time window
            window_data = self.kpi_data.loc[start_date:end_date]

            # interpolate missing values
            window_data = window_data.interpolate(method="polynomial", order=2)

            # remove left over NaN
            window_data.dropna(inplace=True)

            X = window_data.values.T  # Transpose to get [n_kpis x n_observations]

            # Store current features for adaptive residual
            self.current_features = X.mean(axis=1)

            # Compute aggregation values for adaptive residual
            self.aggregation_values[window_label] = self._compute_aggregation_values(X)

            n = self.n_kpis
            n_flat = n * (n - 1)  # Number of non-diagonal elements

            # Use Virgo initialization instead of zeros
            if A_prev is not None:
                # Use previous adjacency structure to inform initialization
                A_init_matrix = self._virgo_initialization(n, n, A_prev)
                # Flatten non-diagonal elements
                A_init = np.zeros(n_flat)
                idx = 0
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            A_init[idx] = max(0, A_init_matrix[i, j])
                            idx += 1
            else:
                # First window, initialize with typical scaling
                A_init = np.random.normal(0, np.sqrt(2.0 / (n * n)), size=n_flat)
                A_init = np.abs(A_init)  # Ensure non-negative

            # Define constraints to ensure non-negativity
            constraints = [{"type": "ineq", "fun": lambda x: x}]  # A_ij >= 0

            # Optimize the objective function with structural regularization
            result = minimize(
                fun=self._objective_function,
                x0=A_init,
                args=(X, A_prev, lambda_structure),
                jac=self._objective_gradient,
                constraints=constraints,
                method="SLSQP",
                options={"maxiter": 300, "disp": True},
            )

            # Reshape the optimized parameters to get the adjacency matrix
            A_opt = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        A_opt[i, j] = max(0, result.x[idx])  # Ensure non-negativity
                        idx += 1

            # Apply residual aggregation operator
            A_residual = self._apply_residual_aggregation(A_opt)

            # Compute the Laplacian matrix using the residual-aggregated adjacency
            D_opt = np.diag(np.sum(A_residual, axis=1))
            L_opt = D_opt - A_residual

            # Apply graph normalization to prevent oversmoothing
            # (BatchNorm-like operation on the adjacency matrix)
            A_norm = A_residual.copy()
            for j in range(n):
                col_mean = np.mean(A_norm[:, j])
                col_std = np.std(A_norm[:, j]) + 1e-8
                A_norm[:, j] = (A_norm[:, j] - col_mean) / col_std

            # Store the results (both original and normalized)
            self.adjacency_matrices[window_label] = A_residual
            self.laplacian_matrices[window_label] = L_opt

            # Create NetworkX graph
            G = nx.DiGraph()
            for i, kpi1 in enumerate(self.kpi_names):
                G.add_node(kpi1)
                for j, kpi2 in enumerate(self.kpi_names):
                    if i != j and A_residual[i, j] > 0:
                        G.add_edge(kpi1, kpi2, weight=A_residual[i, j])

            self.graph_series[window_label] = G

            # Update previous adjacency matrix for temporal consistency
            A_prev = A_residual

        logging.info(
            f"Graph structure learning completed for {len(self.graph_series)} time windows."
        )
        return self.adjacency_matrices

    def detect_communities(
        self, method: str = "walktrap", n_communities: Optional[int] = None
    ) -> Dict:
        """
        Detect communities in KPI networks using enhanced methods.

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

        logging.info(
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

            def perturbed_eigenvector_centrality(G, epsilon=1e-6):
                A = nx.to_numpy_array(G)
                # Add weak connections between components
                A_perturbed = A + epsilon * np.ones(A.shape)
                G_perturbed = nx.from_numpy_array(A_perturbed)
                return nx.eigenvector_centrality_numpy(G_perturbed)

            eigenvector_centrality = perturbed_eigenvector_centrality(G)

            # For betweenness centrality, create undirected graph if needed
            try:
                betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
            except Exception as e:
                logging.exception(e)
                G_undir = G.to_undirected()
                betweenness_centrality = nx.betweenness_centrality(
                    G_undir, weight="weight"
                )

            # Add aggregation values as a centrality measure
            aggregation_centrality = {
                kpi: value
                for kpi, value in zip(
                    self.kpi_names, self.aggregation_values[window_label]
                )
            }

            self.centrality_measures[window_label] = {
                "degree": degree_centrality,
                "in_degree": in_degree_centrality,
                "out_degree": out_degree_centrality,
                "eigenvector": eigenvector_centrality,
                "betweenness": betweenness_centrality,
                "aggregation": aggregation_centrality,
            }

        logging.info(
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
            Type of centrality to use ('degree', 'in_degree', 'out_degree', 'eigenvector', 'betweenness', 'aggregation').
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

        # Ensure the Laplacian matrix has the correct dimensions
        if L.shape[0] != data_with_missing.shape[1]:
            raise ValueError(
                "Laplacian matrix dimensions do not match the number of KPIs."
            )

        # Create a mask of observed values (1 for observed, 0 for missing)
        mask = ~data_with_missing.isna()
        mask_values = mask.values

        # Initialize with mean imputation
        X_init = data_with_missing.copy()
        for col in X_init.columns:
            X_init[col] = X_init[col].fillna(X_init[col].mean())

        X_values = X_init.values

        # Define the objective function for missing value optimization
        def obj_func(x_flat):
            # Reshape the flattened array
            X_pred = X_values.copy()
            X_pred[~mask_values] = x_flat

            # Reconstruction error term (only for observed values)
            recon_error = np.sum((mask_values * (X_values - X_pred)) ** 2)

            # Graph smoothness term using the Laplacian
            smoothness = 0
            for t in range(X_pred.shape[0]):
                x_t = X_pred[t]  # Shape (n_kpis,)
                smoothness += x_t @ L @ x_t  # Scalar
            smoothness *= lambda_reg

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

        logging.info(f"Missing value prediction completed.")
        return predicted_df

    def run_full_analysis(
        self,
        periods: List[int] = [12],
        n_communities: int = 3,
        centrality_type: str = "eigenvector",
        lambda_structure: float = 0.1,
    ) -> Dict:
        """
        Run the complete KPI network analysis pipeline with optimized algorithms.

        Parameters:
        -----------
        periods : List[int]
            List of seasonal periods for multi-seasonal time series decomposition.
        n_communities : int
            Number of communities to detect.
        centrality_type : str
            Type of centrality to use for key influencer identification.
        lambda_structure : float
            Weight for structure regularization in graph learning.

        Returns:
        --------
        Dict : Summary of analysis results.
        """
        if self.kpi_data is None:
            raise ValueError("Data must be loaded first.")

        logging.info("Starting full KPI network analysis with optimized algorithms...")

        # Step 1: Time series decomposition
        logging.info("Step 1: Performing multi-seasonal time series decomposition...")
        self.decompose_time_series(periods=periods)

        # Step 2: Learn graph structure with optimized methods
        logging.info(
            "Step 2: Learning graph structure with Virgo initialization and residual aggregation..."
        )
        self.learn_graph_structure(lambda_structure=lambda_structure)

        # Step 3: Detect communities
        logging.info("Step 3: Detecting communities...")
        self.detect_communities(method="spectral", n_communities=n_communities)

        # Step 4: Compute centrality measures
        logging.info("Step 4: Computing centrality measures...")
        self.compute_centrality_measures()

        # Step 5: Identify key influencers
        logging.info("Step 5: Identifying key influencers...")
        key_influencers = self.identify_key_influencers(centrality_type=centrality_type)

        # Prepare summary of results
        results_summary = {
            "n_kpis": self.n_kpis,
            "time_periods": len(self.kpi_data),
            "periods_analyzed": periods,
            "communities": self.communities,
            "key_influencers": key_influencers,
        }

        logging.info(
            "KPI network analysis completed successfully with optimized algorithms."
        )
        return results_summary
