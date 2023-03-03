"""
oma_clustering.py is a module for clustering modes using DBSCAN algorithm.
To install the hdbscan package, run:
    conda install -c conda-forge hdbscan
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field

import hdbscan
import pandas as pd
from sklearn.cluster import DBSCAN

from oma_tracking.utils import check_columns


def data_selection(
    modal_data: pd.DataFrame,
    cols: list,
    min_size: float,
    max_damping: float
    ):
    # feature selection
    dbscan_data = modal_data[cols]
    # remove clusters with small size and very high damping as these are non-physical
    dbscan_data = dbscan_data[dbscan_data['size'] > min_size]
    dbscan_data = dbscan_data[dbscan_data['damping'] < max_damping]

    return dbscan_data


def column_multiplier(
    modal_data: pd.DataFrame,
    cols: list,
    multipliers: dict[str,float],
    index_divider: float
    ):
    multiplied_data = modal_data.copy()
    # Remove timestamps as index to allow for time gaps in monitoring
    multiplied_data.reset_index(inplace=True)
    multiplied_data = multiplied_data[cols]
    for key in multipliers:
        if 'damping' in key:
            multiplied_data[key] = (multiplied_data[key] + 1)
        multiplied_data[key] = multiplied_data[key] * multipliers[key]
    # Include the index dimension to the clustering
    multiplied_data["time_diff"] = (
        multiplied_data.index.astype(float) - multiplied_data.index.values[0].astype(float)
    ) / index_divider
    return multiplied_data


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
    multipliers: dict[str,float] = \
        field(default_factory=lambda: {"frequency": 40, "size": 0.5, "damping": 1})
    index_divider: float = 20000
    cols: list[str] = \
        field(default_factory=lambda: ["frequency", "size", "damping"])
    min_size: float = 5.0
    max_damping: float = 5.0

    def __post_init__(self):
        self.dbsc = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.dbscan_data: pd.DataFrame = pd.DataFrame()

    def fit(
        self,
        modal_data: pd.DataFrame,
        **kwargs
        ):
        """Fit the modal_data to the DBSCAN algorithm
        for the time period between start_time and end_time.

        Args:
            modal_data (pd.DataFrame): The modal data to be fitted.
        """
        if not check_columns(self.cols, modal_data):
            raise ValueError(
                "The modal data does not contain all the required columns."
            )
        dbscan_data = \
            data_selection(
                modal_data,
                self.cols,
                self.min_size,
                self.max_damping
            )

        ## feature construction
        multiplied_dbscan_data = \
            column_multiplier(
                dbscan_data,
                self.cols,
                self.multipliers,
                self.index_divider
            )

        self.dbsc = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(multiplied_dbscan_data)
        dbscan_data["labels"] = self.dbsc.labels_
        self.dbscan_data = dbscan_data

    def predict(self, min_cluster_size: int = 1000) -> pd.DataFrame:
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



@dataclass()
class ModeClusterer_HDBSCAN:
    """ModeClusterer is a class for clustering mode parameters using HDBSCAN algorithm.

    Attributes:
        freq_multiplier (float): The multiplier for the frequency feature.
        damping_multiplier (float): The multiplier for the damping feature.
        size_multiplier (float): The multiplier for the size feature.
        cols (list): The columns of the dataframe that should be used for clustering.
        min_modal_size (float): The minimum size of a mode to be considered for clustering.
        max_modal_damping (float): The maximum damping of a mode to be considered for clustering.
        dbsc (DBSCAN): The DBSCAN object that stores the result of the clustering.
        dbscan_data (pd.DataFrame): The dataframe that is used for clustering.
    """
    min_cluster_size: int = 100
    cluster_selection_epsilon: float = 0
    multipliers: dict[str,float] = \
        field(default_factory=lambda: {"frequency": 40, "size": 0.5, "damping": 1})
    index_divider: float = 20000
    cols: list[str] = \
        field(default_factory=lambda: ["frequency", "size", "damping"])
    min_size: float = 5.0
    max_damping: float = 5.0


    def __post_init__(self):
        self.dbsc = \
            hdbscan.HDBSCAN(
                min_cluster_size = self.min_cluster_size,
                cluster_selection_epsilon = self.cluster_selection_epsilon,
            )
        self.hdbscan_data: pd.DataFrame = pd.DataFrame()

    def fit(
        self,
        modal_data: pd.DataFrame,
        **kwargs
        ):
        """Fit the modal_data to the HDBSCAN algorithm
        for the time period between start_time and end_time.

        Args:
            modal_data (pd.DataFrame): The modal data to be fitted.
        """
        if not check_columns(self.cols, modal_data):
            raise ValueError(
                "The modal data does not contain all the required columns."
            )
        hdbscan_data = \
            data_selection(
                modal_data,
                self.cols,
                self.min_size,
                self.max_damping
            )

        ## feature construction
        multiplied_dbscan_data = \
            column_multiplier(
                hdbscan_data,
                self.cols,
                self.multipliers,
                self.index_divider
            )

        self.dbsc = \
            hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                **kwargs
            ).fit(multiplied_dbscan_data)
        hdbscan_data["labels"] = self.dbsc.labels_
        self.hdbscan_data = hdbscan_data

    def predict(self, min_cluster_size: int = 500) -> pd.DataFrame:
        """Predict the clusters of the fitted data
        that have more clusters than the min_cluster_size.

        Args:
            min_cluster_size (int, optional): The minimum number of modes in a cluster.
                Defaults to 500.

        Returns:
            pd.DataFrame: The dataframe with the predicted clusters.
        """

        clustered_data = self.hdbscan_data.copy()
        # Remove clusters with less than min_cluster_size samples
        lbls = []
        for label in self.hdbscan_data["labels"].unique():
            cnt = len(self.hdbscan_data[self.hdbscan_data["labels"] == label])
            if cnt > min_cluster_size:
                lbls.append(label)

        clustered_data = self.hdbscan_data[self.hdbscan_data["labels"].isin(lbls)][
            self.hdbscan_data[self.hdbscan_data["labels"].isin(lbls)]["labels"] >= 0
        ]
        # Reset the modes to start from 0
        codes, uniques = pd.factorize(clustered_data["labels"])
        clustered_data["labels"] = codes
        return clustered_data
