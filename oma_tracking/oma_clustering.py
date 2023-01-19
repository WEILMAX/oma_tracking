"""
oma_clustering.py is a module for clustering modes using DBSCAN algorithm.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
import pandas as pd


@dataclass
class ModeClusterer:
    """ModeClusterer is a class for clustering mode parameters using DBSCAN algorithm.

    Attributes:
        eps (float): The maximum distance between two samples
            for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples (or total weight) in a neighborhood,
            for a point to be considered as a core point. This includes the point itself.
        freq_multiplier (float): The multiplier for the frequency feature.
        damping_multiplier (float): The multiplier for the damping feature.
        size_multiplier (float): The multiplier for the size feature.
        cols (list): The columns of the dataframe that should be used for clustering.
        min_modal_size (float): The minimum size of a mode to be considered for clustering.
        max_modal_damping (float): The maximum damping of a mode to be considered for clustering.
        dbsc (DBSCAN): The DBSCAN object that stores the result of the clustering.
        dbscan_data (pd.DataFrame): The dataframe that is used for clustering.
    """

    eps: float = 5
    min_samples: int = 100
    freq_multiplier: float = 40
    damping_multiplier: float = 1
    size_multiplier: float = 0.5
    index_divider: float = 20000
    cols: list = field(default_factory=lambda: ["frequency", "size", "damping"])
    min_modal_size: float = 5
    max_modal_damping: float = 5

    def __post_init__(self):
        self.dbsc = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.dbscan_data: pd.DataFrame = pd.DataFrame()

    def fit(self, modal_data: pd.DataFrame):
        """Fit the modal_data to the DBSCAN algorithm
        for the time period between start_time and end_time.

        Args:
            modal_data (pd.DataFrame): The modal data to be fitted.
        """
        if not check_columns(self.cols, modal_data):
            raise ValueError(
                "The modal data does not contain all the required columns."
            )
        # feature selection
        dbscan_data = modal_data[self.cols]
        # remove clusters with small size and very high damping as these are non-physical
        dbscan_data = dbscan_data[modal_data["size"] > self.min_modal_size]
        dbscan_data = dbscan_data[dbscan_data["damping"] < self.max_modal_damping]

        # Remove timestamps as index to allow for time gaps in monitoring
        indices = dbscan_data.index
        dbscan_data.reset_index(inplace=True)
        dbscan_data = dbscan_data[self.cols]

        ## feature construction
        # Multiply the features to increase or decrease the importance
        dbscan_data["damping"] = (dbscan_data["damping"] + 1) * self.damping_multiplier
        dbscan_data["frequency"] = dbscan_data["frequency"] * self.freq_multiplier
        dbscan_data["size"] = dbscan_data["size"] * self.size_multiplier
        # Include the index dimension to the clustering
        dbscan_data["time_diff"] = (
            dbscan_data.index.astype(float) - dbscan_data.index.values[0].astype(float)
        ) / self.index_divider

        self.dbsc = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(dbscan_data)

        # Reset the features to their original values
        dbscan_data["frequency"] = dbscan_data["frequency"] / self.freq_multiplier
        dbscan_data["damping"] = dbscan_data["damping"] / self.damping_multiplier - 1
        dbscan_data["size"] = dbscan_data["size"] / self.size_multiplier
        dbscan_data.set_index(indices, inplace=True)
        dbscan_data["labels"] = self.dbsc.labels_

        self.dbscan_data = dbscan_data

    def predict(self, min_cluster_size: int = 500) -> pd.DataFrame:
        """Predict the clusters of the fitted data
        that have more clusters than the min_cluster_size.

        Args:
            min_cluster_size (int, optional): The minimum number of modes in a cluster.
                Defaults to 500.

        Returns:
            pd.DataFrame: The dataframe with the predicted clusters.
        """

        clustered_data = self.dbscan_data.copy()
        # Remove clusters with less than min_cluster_size samples
        lbls = []
        for label in self.dbscan_data["labels"].unique():
            cnt = len(self.dbscan_data[self.dbscan_data["labels"] == label])
            if cnt > min_cluster_size:
                lbls.append(label)

        clustered_data = self.dbscan_data[self.dbscan_data["labels"].isin(lbls)][
            self.dbscan_data[self.dbscan_data["labels"].isin(lbls)]["labels"] >= 0
        ]
        # Reset the modes to start from 0
        codes, uniques = pd.factorize(clustered_data["labels"])
        clustered_data["labels"] = codes
        return clustered_data


def check_columns(cols: list, data: pd.DataFrame) -> bool:
    """
    Check if all elements of a list are in the columns of a dataframe.

    Args:
        columns (list): The list of columns to be checked
        data (pd.DataFrame): The dataframe to be checked against

    Returns:
        bool: True if all elements of the list are in the columns of the dataframe, False otherwise
    """
    return all(col in data.columns for col in cols)
