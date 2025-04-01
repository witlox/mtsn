import logging
import numpy as np
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mtsn import MTSN


def compute_confidence_interval(
    data: np.ndarray, metric_func: callable, num_samples: int = 1000
):
    bootstrap_samples = [
        metric_func(np.random.choice(data, size=len(data), replace=True))
        for _ in range(num_samples)
    ]
    lower_bound = np.percentile(bootstrap_samples, 2.5)
    upper_bound = np.percentile(bootstrap_samples, 97.5)
    return lower_bound, upper_bound


def validate_communities(true_labels: List[int], predicted_labels: List[int]):
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)

    return {"ARI": ari_score, "NMI": nmi_score}


def visualize_kpi_network(
    mtsn,
    window_label: str = "all",
    threshold: float = 0.1,
    show_communities: bool = True,
    centrality_size: str = "eigenvector",
    layout: str = "spring",
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Visualize the KPI network with communities and centrality information.

    Parameters:
    -----------
    mtsn : MTSN
        Instance of the Multi-Temporal KPI Network Analyzer.
    window_label : str
        Label of the time window to visualize.
    threshold : float
        Threshold for edge weights to include in visualization.
    show_communities : bool
        Whether to color nodes by community.
    centrality_size : str
        Centrality measure to use for node sizes.
    layout : str
        Layout algorithm to use ('spring', 'kamada_kawai', 'circular').
    figsize : Tuple[int, int]
        Figure size.
    """
    logger = logging.getLogger(__name__)
    G = mtsn.graph_series[window_label]

    # Create a copy of the graph with filtered edges
    G_viz = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data["weight"] > threshold:
            G_viz.add_edge(u, v, weight=data["weight"])

    for node in G.nodes():
        G_viz.add_node(node)

    plt.figure(figsize=figsize)

    # Node colors based on communities
    node_colors = None
    if show_communities and window_label in mtsn.communities:
        community_dict = mtsn.communities[window_label]
        node_colors = [community_dict[node] for node in G_viz.nodes()]

    # Node sizes based on centrality
    node_sizes = None
    if window_label in mtsn.centrality_measures:
        if centrality_size in mtsn.centrality_measures[window_label]:
            centrality_dict = mtsn.centrality_measures[window_label][centrality_size]
            node_sizes = [
                1000 * (0.1 + np.log1p(centrality_dict[node])) for node in G_viz.nodes()
            ]
        else:
            logger.warning(
                f"Warning: Centrality measure {centrality_size} not found. Using default sizes."
            )
            node_sizes = [300] * len(G_viz.nodes())
    else:
        node_sizes = [300] * len(G_viz.nodes())

    # Set layout
    if layout == "spring":
        pos = nx.spring_layout(G_viz, k=0.5, iterations=100, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G_viz)
    elif layout == "circular":
        pos = nx.circular_layout(G_viz)
    else:
        pos = nx.spring_layout(G_viz, seed=42)

    # Draw the network
    nx.draw_networkx_nodes(
        G_viz,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        cmap=plt.cm.viridis,
    )

    # Draw edges with width proportional to weight
    for u, v, data in G_viz.edges(data=True):
        width = 10 * (1 + data["weight"])
        nx.draw_networkx_edges(
            G_viz,
            pos,
            edgelist=[(u, v)],
            width=width,
            alpha=0.5,
            arrows=True,
            arrowsize=25,
        )

    # Add labels
    nx.draw_networkx_labels(G_viz, pos, font_size=10, font_family="sans-serif")

    plt.title(f"KPI Network - {window_label}")
    plt.axis("off")

    # If communities are shown, add a colorbar
    if show_communities and node_colors:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(min(node_colors), max(node_colors))
        )
        sm.set_array([])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(sm, label="Community", cax=cax)

    plt.tight_layout()
    plt.show()


def visualize_seasonal_patterns(mtsn, num_kpis: int = 5) -> None:
    """
    Visualize seasonal patterns of top KPIs, including individual seasonal components.

    Parameters:
    -----------
    mtsn : MTSN
        Instance of the Multi-Temporal KPI Network Analyzer.
    num_kpis : int
        Number of KPIs to visualize.
    """
    # Select top KPIs by variance in seasonal component
    seasonal_variance = {}
    for kpi, components in mtsn.decomposed_data.items():
        seasonal_variance[kpi] = np.var(components["seasonal"])

    top_kpis = sorted(seasonal_variance.items(), key=lambda x: x[1], reverse=True)[
        :num_kpis
    ]
    top_kpi_names = [kpi for kpi, _ in top_kpis]

    # Plot the combined seasonal components
    plt.figure(figsize=(12, 8))

    for kpi in top_kpi_names:
        seasonal = mtsn.decomposed_data[kpi]["seasonal"]
        plt.plot(range(len(seasonal)), seasonal, label=kpi)

    plt.title("Combined Seasonal Components of Top KPIs")
    plt.xlabel("Time")
    plt.ylabel("Seasonal Component")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot individual seasonal components for the first KPI
    if top_kpi_names:
        first_kpi = top_kpi_names[0]
        seasonal_components = mtsn.decomposed_data[first_kpi]["seasonal_components"]

        # Check if we have multiple seasonal components
        if (
            isinstance(seasonal_components, pd.DataFrame)
            and len(seasonal_components.columns) > 0
        ):
            plt.figure(figsize=(12, 8))

            for column in seasonal_components.columns:
                plt.plot(
                    range(len(seasonal_components)),
                    seasonal_components[column],
                    label=f"Period {column}",
                )

            plt.title(f"Individual Seasonal Components for {first_kpi}")
            plt.xlabel("Time")
            plt.ylabel("Component Value")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()

    # Plot the original time series with trend
    fig, axes = plt.subplots(num_kpis, 1, figsize=(12, 3 * num_kpis), sharex=True)

    for i, kpi in enumerate(top_kpi_names):
        axes[i].plot(
            mtsn.decomposed_data[kpi]["original"], label="Original", color="blue"
        )
        axes[i].plot(mtsn.decomposed_data[kpi]["trend"], label="Trend", color="red")
        axes[i].set_title(f"KPI: {kpi}")
        axes[i].legend()
        axes[i].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def visualize_centrality_distribution(
    mtsn,
    window_label: str = "all",
    centrality_type: str = "eigenvector",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Visualize the distribution of centrality measures across KPIs.

    Parameters:
    -----------
    mtsn : MTSN
        Instance of the Multi-Temporal KPI Network Analyzer.
    window_label : str
        Label of the time window to visualize.
    centrality_type : str
        Type of centrality to visualize.
    figsize : Tuple[int, int]
        Figure size.
    """
    if not mtsn.centrality_measures:
        mtsn.compute_centrality_measures()

    centrality_dict = mtsn.centrality_measures[window_label][centrality_type]

    # Sort KPIs by centrality value
    sorted_items = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    kpis, values = zip(*sorted_items)

    plt.figure(figsize=figsize)
    bars = plt.bar(kpis, values)

    # Color bars by value
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(values[i] / max(values)))

    plt.title(
        f"{centrality_type.capitalize()} Centrality Distribution - {window_label}"
    )
    plt.xlabel("KPI")
    plt.ylabel(f"{centrality_type.capitalize()} Centrality")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.7, axis="y")
    plt.tight_layout()
    plt.show()


def run_full_analysis(
    mtsn,
    periods: List[int] = [12],
    n_communities: int = 3,
    centrality_type: str = "eigenvector",
    lambda_structure: float = 0.1,
):
    """
    Run the complete KPI network analysis pipeline with optimized algorithms.

    Parameters:
    -----------
    mtsn : MTSN
        Instance of the Multi-Temporal KPI Network Analyzer.
    periods : List[int]
        List of seasonal periods for multi-seasonal time series decomposition.
    n_communities : int
        Number of communities to detect.
    centrality_type : str
        Type of centrality to use for key influencer identification.
    lambda_structure : float
        Weight for structure regularization in graph learning.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting full KPI network analysis with optimized algorithms...")

    # Step 1: Time series decomposition
    mtsn.decompose_time_series(periods=periods)

    # Step 2: Learn graph structure with optimized methods
    mtsn.learn_graph_structure(lambda_structure=lambda_structure)

    # Step 3: Detect communities
    mtsn.detect_communities(method="spectral", n_communities=n_communities)

    # Step 4: Compute centrality measures
    mtsn.compute_centrality_measures()

    # Step 5: Identify key influencers
    key_influencers = mtsn.identify_key_influencers(centrality_type=centrality_type)

    # Prepare summary of results
    results_summary = {
        "n_kpis": mtsn.n_kpis,
        "time_periods": len(mtsn.kpi_data),
        "periods_analyzed": periods,
        "communities": mtsn.communities,
        "key_influencers": key_influencers,
    }

    logger.info(
        "KPI network analysis completed successfully with optimized algorithms."
    )
    return results_summary
