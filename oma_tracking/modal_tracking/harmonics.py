"""
harmonics.py is a module for detecting, isolating and removing 
harmonic modes from an Opeational Modal Analysis (OMA) dataset.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import List, Union
import datetime

def theoretical_harmonic(
    data: pd.DataFrame,
    p_orders: list[int]
    ) -> pd.DataFrame:
    """
    theoretical_harmonic is a function that takes in a pandas dataframe 
    that contains the rpm data and a p_order and returns a dataframe of 
    theoretical harmonic data.
    harmonic = (p/60) * rpm 
    ---------------------------------------------------------------------
    Args:
        data (pd.DataFrame): A dataframe that contains the rpm data.
        p_order (int): Order of the harmonic
    ---------------------------------------------------------------------
    Returns:
        pd.DataFrame
            A dataframe of the harmonic mode based on the rpm.
    """
    # Get the rpm data from the dataframe.
    rpm_data = data.filter(regex='rpm')
    if rpm_data.empty:
        raise ValueError("No rpm data found in dataframe.")
    # check that the rpm data has a datetime index
    if type(rpm_data.index) is not pd.DatetimeIndex:
        raise ValueError("The rpm data does not have a datetime index.")
    # Create a new dataframe with the same index as the rpm data
    harmonic_data = pd.DataFrame(index = rpm_data.index)
    # Calculate the harmonic data for each p_order
    for p_order in p_orders:
        harmonic_data[f'harmonic_{p_order}p'] = (p_order/60) * rpm_data
    return harmonic_data


@dataclass
class HarmonicDetector:
    scada_data: pd.DataFrame
    modal_data: pd.DataFrame
    p_orders: List[int] = field(default_factory=list)
    distances: pd.DataFrame = field(init=False)
    min_rpm: float = 6.0
    max_distance: float = 0.1

    def __post_init__(self):
        return None
        
    def get_distance_calculator_data(
        self
        ) -> pd.DataFrame:
        """Get the rpm and frequency data from
        the scada and modal dataframes.
        This data is later used to 

        Args:
            scada_data (pd.DataFrame): _description_
            modal_data (pd.DataFrame): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        frequency_data = self.modal_data.filter(regex='frequency')
        if not frequency_data.filter(regex='std').empty:
            columns_to_remove = \
                frequency_data.filter(regex='std').columns
            frequency_data = frequency_data.drop(columns_to_remove, axis=1)
        if frequency_data.empty:
            raise ValueError("No frequency data found in dataframe.")
        rpm_data = self.scada_data.filter(regex='rpm').loc[frequency_data.index]
        if rpm_data.empty:
            raise ValueError("No rpm data found in dataframe.")
        distance_calculator_data = pd.concat([rpm_data, frequency_data], axis=1)
        return distance_calculator_data

    def distances_data(
        self
        ) -> pd.DataFrame:
        """Calculate the distances between the
        theoretical harmonic and the measured
        frequency.

        Returns:
            pd.DataFrame: Distances to the theoretical harmonic
        """
        self.distances = pd.DataFrame()
        for p_order in self.p_orders:
            self.distances[f'distance_{p_order}p'] = \
                self.distance_to_harmonic(p_order)
        return self.distances
    

    def get_plot_distance_data(self,
        frequency_range: tuple[float, float] = (0, 2),
        max_damping: float = 10.0
        ) -> pd.DataFrame:
        plt_data = \
            pd.concat([self.modal_data, self.distances_data()],
            axis=1)
        plt_data = plt_data[plt_data['damping'] < max_damping]
        plt_data = plt_data[plt_data['frequency'] < frequency_range[1]]
        plt_rpm = self.scada_data.filter(like='rpm').loc[plt_data.index]
        plt_data = pd.concat([plt_data, plt_rpm], axis=1)
        return plt_data

    def plot_distances_p(
        self,
        p_harmonic: int,
        figsize: tuple[int, int] = (10,6),
        frequency_range: tuple[float, float] = (0, 2),
        max_damping: float = 10.0,
        ) -> None:
        """Plot the distances between the
        theoretical harmonic and the measured
        frequency.

        Args:
            
        """
        # Prepare the data
        plt_data = \
            self.get_plot_distance_data(
                frequency_range,
                max_damping
            ).dropna()
        # Generate the figure
        plt.figure(figsize = figsize)
        plt.scatter(
            plt_data.filter(regex='rpm'),
            plt_data['frequency'],
            alpha=0.1,
            c=plt_data[f'distance_{p_harmonic}p'],
            cmap = 'nipy_spectral'
        )
        plt.grid(True, color='k', linestyle='--', linewidth=0.5)
        cbar = plt.colorbar(alpha=1)
        cbar.solids.set_alpha(1.0)
        #plt.ylim(frequency_range[0], frequency_range[1])
        plt.xlabel('RPM')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Distance to {p_harmonic}p')
        plt.show()

    def plot_distances(
        self,
        figsize: tuple[int, int] = (10,6),
        figsize2: tuple[int, int] = (30,8),
        frequency_range: tuple[float, float] = (0, 2),
        max_damping: float = 10.0,
        direction: str = 'SS'
        ) -> None:
        """Plot the distances between the
        theoretical harmonic and the measured
        frequency.

        Args:
            
        """
        # Prepare the data
        plt_data = \
            self.get_plot_distance_data(
                frequency_range,
                max_damping
            ).dropna()
        # Generate the rpm-freq figure
        plt.figure(figsize = figsize)
        plt.scatter(
            plt_data.filter(regex='rpm'),
            plt_data['frequency'],
            alpha=0.1,
            color='grey',
            label = 'tracked modes'
        )
        min_rpm_plt_data = plt_data[(plt_data.filter(regex='rpm') > self.min_rpm).values]
        for p_order in self.p_orders:
            p_order_data = min_rpm_plt_data[min_rpm_plt_data[f'distance_{p_order}p'] < self.max_distance]
            plt.scatter(
                p_order_data.filter(regex='rpm'),
                p_order_data['frequency'],
                alpha=0.1,
                label=f'{p_order}p'
            )
        plt.grid(True, color='k', linestyle='--', linewidth=0.5)
        plt.ylim(frequency_range[0], frequency_range[1])
        plt.xlabel('RPM')
        plt.ylabel('Frequency (Hz)')
        plt.title(direction + ' harmonics detected' )
        legend = plt.legend()
        for handle in legend.legendHandles:
            handle.set_alpha(1.0)
        plt.show()

        # Generate the freq timeseries figure
        plt.figure(figsize = figsize2)
        plt.scatter(
            plt_data.index,
            plt_data['frequency'],
            alpha=0.1,
            color='grey',
            label = 'tracked modes'
        )
        min_rpm_plt_data = plt_data[(plt_data.filter(regex='rpm') > self.min_rpm).values]
        for p_order in self.p_orders:
            p_order_data = min_rpm_plt_data[min_rpm_plt_data[f'distance_{p_order}p'] < self.max_distance]
            plt.scatter(
                p_order_data.index,
                p_order_data['frequency'],
                alpha=0.1,
                label=f'{p_order}p'
            )
        plt.grid(True, color='k', linestyle='--', linewidth=0.5)
        plt.ylim(frequency_range[0], frequency_range[1])
        plt.xlabel('Timestamp')
        plt.ylabel('Frequency (Hz)')
        plt.title(direction + ' harmonics detected' )
        legend = plt.legend()
        for handle in legend.legendHandles:
            handle.set_alpha(1.0)
        plt.show()

    def distance_to_harmonic(
        self,
        p_harmonic:int
        ) -> pd.Series:
        """Calculate the distance between the
        theoretical harmonic and the measured
        frequency.

        Args:
            p_harmonic (int): order of the harmonic

        Returns:
            pd.Series: Distances to the theoretical harmonic
        """
        data = self.get_distance_calculator_data()
        distances = pd.Series(
            np.abs(
                (
                    p_harmonic/60 \
                    * data.filter(regex='rpm').values[:,0]
                ) \
                - data['frequency'].values
            ),
            index = data.index,
        )
        return distances
    
    def remove_harmonics(
        self
    ) -> pd.DataFrame:
        """Remove the harmonics from the
        modal data.

        Returns:
            pd.DataFrame: Modal data without harmonics
        """
        data = self.get_plot_distance_data()
        data.reset_index(inplace=True)
        min_rpm_data = data[(data.filter(regex='rpm') > self.min_rpm).values]
        harmonics = \
            min_rpm_data[ \
                min_rpm_data.filter(regex='distance').min(axis=1) \
                < self.max_distance \
                ]
        harmonics_removed = data.drop(harmonics.index)
        harmonics_removed.set_index('timestamp', inplace=True)
        return harmonics_removed
    
    def plot_harmonics_removed(
        self,
        plot_theoretical_harmonic = False,
        ylim: tuple[float, float] = (0, 2),
        xlim: Union[tuple[datetime.datetime, datetime.datetime], None] = \
            None
        ) -> None:
        """Plot the modal data without harmonics.
        """
        harmonics_removed = self.remove_harmonics()
        data = self.get_plot_distance_data()
        plt.figure(figsize=(30,8))
        plt.scatter(
            data.index,
            data['frequency'],
            alpha=1.0,
            color='grey',
            label = 'tracked modes'
        )
        plt.scatter(
            harmonics_removed.index,
            harmonics_removed['frequency'],
            alpha=0.1,
            label = 'harmonics removed',
            color='purple'
        )
        if plot_theoretical_harmonic:
            theoretical_harmonics = \
                theoretical_harmonic(
                    data,
                    p_orders=self.p_orders
                )
            for p_order in self.p_orders:
                plt.plot(
                    theoretical_harmonics[f'harmonic_{p_order}p'],
                    label=f'theoretical {p_order}p harmonic',
                    alpha=0.8
                )
        plt.grid(True, color='k', linestyle='--', linewidth=0.5)
        plt.ylim(ylim[0], ylim[1])
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        plt.xlabel('Timestamp')
        plt.ylabel('Frequency (Hz)')
        plt.title('Harmonics removed')
        legend = plt.legend()
        for handle in legend.legendHandles:
            handle.set_alpha(1.0)
        plt.show()
