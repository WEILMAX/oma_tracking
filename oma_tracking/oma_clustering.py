"""
oma_clustering.py is a module for clustering modes using DBSCAN algorithm.
To install the hdbscan package, run:
    conda install -c conda-forge hdbscan
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field
from typing import Union
#import hdbscan
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN

from oma_tracking.utils import check_columns


def data_selection(
    modal_data: pd.DataFrame,
    cols: list,
    min_size: float,
    max_damping: float
    ):
    # feature selection
    dbscan_data = modal_data#[cols]
    # remove clusters with small size and very high damping as these are non-physical
    dbscan_data = dbscan_data[dbscan_data['size'] > min_size]
    if 'damping' in dbscan_data.columns:
        dbscan_data = dbscan_data[dbscan_data['damping'] < max_damping]
    elif 'mean_damping' in dbscan_data.columns:
        dbscan_data = dbscan_data[dbscan_data['mean_damping'] < max_damping]
    else: 
        raise KeyError(
            "The modal data does not contain the column 'damping' or 'mean_damping'."
        )

    return dbscan_data


def column_multiplier(
    modal_data: pd.DataFrame,
    cols: list,
    multipliers: dict[str,float],
    index_divider: Union[float, None] = None
    ):
    multiplied_data = modal_data.copy()
    # Remove timestamps as index to allow for time gaps in monitoring
    multiplied_data.reset_index(inplace=True)
    multiplied_data = multiplied_data[cols]
    for key in multipliers:
        if 'damping' in key:
            multiplied_data[key] = (multiplied_data[key] + 1)
        multiplied_data[key] = multiplied_data[key] * multipliers[key]
    # Include the index dimension to the clustering if index_divider is not None
    if index_divider is not None:
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
    index_divider: Union[float, None] = None
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
        frequency_range: Union[tuple[float, float], None] = None,
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
        frequency_col = 'mean_frequency'
        if frequency_col not in modal_data.columns:
            frequency_col = 'frequency'
            if frequency_col not in modal_data.columns:
                raise ValueError("No frequency data found in dataframe. Columns 'mean_frequency' or 'frequency' required.")
        if frequency_range is not None:
            modal_data = \
                modal_data.copy().loc[
                    (modal_data[frequency_col] >= frequency_range[0]) &
                    (modal_data[frequency_col] <= frequency_range[1])
                ]
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

        self.dbsc = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(multiplied_dbscan_data[self.cols])
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
    
    def predict_with_noise(self, min_cluster_size: int = 500) -> pd.DataFrame:
        """Predict the clusters of the fitted data
        that have more clusters than the min_cluster_size.
        Keep the noise as -1.

        Args:
            min_cluster_size (int, optional): The minimum number of modes in a cluster.
                Defaults to 500.

        Returns:
            pd.DataFrame: The dataframe with the predicted clusters and the noise as -1.
        """

        clustered_data = self.dbscan_data.copy()
        # Remove clusters with less than min_cluster_size samples
        lbls = []
        for label in self.dbscan_data["labels"].unique():
            cnt = len(self.dbscan_data[self.dbscan_data["labels"] == label])
            if cnt > min_cluster_size:
                lbls.append(label)

        # Replace all labels that aren't in lbls with -1
        clustered_data.loc[~clustered_data["labels"].isin(lbls), "labels"] = -1
        # Reset the modes bigger than 0 to start from 0 keeping the previous order but filling the missing gaps
        # and keep the noise as -1
        non_noise_clusters = clustered_data[clustered_data["labels"] >= 0]
        codes, uniques = pd.factorize(non_noise_clusters["labels"])
        non_noise_clusters["labels"] = codes
        clustered_data[clustered_data["labels"] >= 0] = non_noise_clusters
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
    min_samples: Union[int, None] = None
    multipliers: dict[str,float] = \
        field(default_factory=lambda: {"frequency": 40, "size": 0.5, "damping": 1})
    index_divider: Union[float, None] = None
    cols: list[str] = \
        field(default_factory=lambda: ["frequency", "size", "damping"])
    min_size: float = 5.0
    max_damping: float = 5.0

    def __post_init__(self):
        self.hdbsc = \
            HDBSCAN(
                min_cluster_size = self.min_cluster_size,
                min_samples = self.min_samples
            )
        self.hdbscan_data: pd.DataFrame = pd.DataFrame()
        

    def fit(
        self,
        modal_data: pd.DataFrame,
        frequency_range: Union[tuple[float, float], None] = None,
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
        frequency_col = 'mean_frequency'
        if frequency_col not in modal_data.columns:
            frequency_col = 'frequency'
            if frequency_col not in modal_data.columns:
                raise ValueError("No frequency data found in dataframe. Columns 'mean_frequency' or 'frequency' required.")
        if frequency_range is not None:
            modal_data = \
                modal_data.copy().loc[
                    (modal_data[frequency_col] >= frequency_range[0]) &
                    (modal_data[frequency_col] <= frequency_range[1])
                ]
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

        self.hdbsc = \
            HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                **kwargs
            ).fit(multiplied_dbscan_data[self.cols])
        hdbscan_data["labels"] = self.hdbsc.labels_
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
    
    def predict_with_noise(self, min_cluster_size: int = 500) -> pd.DataFrame:
        """Predict the clusters of the fitted data
        that have more clusters than the min_cluster_size.
        Keep the noise as -1.

        Args:
            min_cluster_size (int, optional): The minimum number of modes in a cluster.
                Defaults to 500.

        Returns:
            pd.DataFrame: The dataframe with the predicted clusters and the noise as -1.
        """

        clustered_data = self.hdbscan_data.copy()
        # Remove clusters with less than min_cluster_size samples
        lbls = []
        for label in self.hdbscan_data["labels"].unique():
            cnt = len(self.hdbscan_data[self.hdbscan_data["labels"] == label])
            if cnt > min_cluster_size:
                lbls.append(label)

        # Replace all labels that aren't in lbls with -1
        clustered_data.loc[~clustered_data["labels"].isin(lbls), "labels"] = -1

        # Reset the modes to start from -1
        codes, uniques = pd.factorize(clustered_data["labels"])
        clustered_data["labels"] = codes
        return clustered_data