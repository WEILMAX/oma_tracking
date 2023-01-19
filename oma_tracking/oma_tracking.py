"""
oma_tracking.py is a module for tracking modes after clustering is done.
Multiple methods are given to track modes.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ModeTracking:
    """
    Class for classifying OMA modes based on the closest clustered mode
    """

    clustered_modes: pd.DataFrame
    oma_modes: pd.DataFrame

    def classify_indices(self):
        """
        Classify the indices of the OMA modes based on the closest clustered mode

        Returns:
            (pd.DataFrame): The OMA modes dataframe with an additional 'label' column indicating the closest clustered mode
        """

        def closest_mode(frequency):
            distances = abs(frequency - self.clustered_modes["mean_frequency"])
            if (distances <= self.clustered_modes["max_distance"]).any():
                closest_index = distances[
                    distances <= self.clustered_modes["max_distance"]
                ].idxmin()
                return self.clustered_modes.loc[closest_index, "labels"]
            else:
                return "undefined"

        self.oma_modes["labels"] = self.oma_modes["frequency"].apply(closest_mode)
        return self.oma_modes
