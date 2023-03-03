import matplotlib.pyplot as plt
import pandas as pd


def draw_modal_frequency_boxplots(
    simulation_data:  dict[str, pd.DataFrame],
    tracked_modes: pd.DataFrame,
    parked_data: pd.DataFrame,
    rated_data: pd.DataFrame,
    location: str,
    mode: str,
    ) -> None:
    fig, ax = plt.subplots(figsize=(20, 5))
    i = 1
    x_keys = []
    plt.boxplot(tracked_modes[mode].dropna(), positions=[i], widths=0.6)
    x_keys.append('tracked mode')
    i += 1
    plt.boxplot(tracked_modes[mode].loc[tracked_modes[mode].index.intersection(parked_data.index)].dropna(), positions=[i], widths=0.6)
    x_keys.append('tracked mode parked')
    i += 1
    plt.boxplot(tracked_modes[mode].loc[tracked_modes[mode].index.intersection(rated_data.index)].dropna(), positions=[i], widths=0.6)
    x_keys.append('tracked mode rated')
    i += 1
    for key in simulation_data.keys():
        if mode in simulation_data[key].columns:
            plt.boxplot(simulation_data[key][mode], positions=[i], widths=0.6)
            i += 1
            x_keys.append(key)
    
    ax.set_xticks(range(1, len(x_keys) + 1))
    ax.set_xticklabels(x_keys, rotation=90)
    ax.set_title(location + ' ' + mode)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Simulation')
    ax.grid(True, color='k', linestyle='--', linewidth=0.5)
    plt.show()

def draw_modal_frequency_violinplots(
    simulation_data: dict[str, pd.DataFrame],
    tracked_modes: pd.DataFrame,
    parked_data: pd.DataFrame,
    rated_data: pd.DataFrame,
    location: str,
    mode: str,
    ) -> None:
    fig, ax = plt.subplots(figsize=(20, 5))
    i = 1
    x_keys = []
    plt.violinplot(tracked_modes[mode].dropna(), positions=[i], widths=0.6)
    x_keys.append('tracked mode')
    i += 1
    plt.violinplot(tracked_modes[mode].loc[tracked_modes[mode].index.intersection(parked_data.index)].dropna(), positions=[i], widths=0.6)
    x_keys.append('tracked mode parked')
    i += 1
    plt.violinplot(tracked_modes[mode].loc[tracked_modes[mode].index.intersection(rated_data.index)].dropna(), positions=[i], widths=0.6)
    x_keys.append('tracked mode rated')
    i += 1
    for key in simulation_data.keys():
        if mode in simulation_data[key].columns:
            plt.violinplot(simulation_data[key][mode], positions=[i], widths=0.6)
            i += 1
            x_keys.append(key)
    
    ax.set_xticks(range(1, len(x_keys) + 1))
    ax.set_xticklabels(x_keys, rotation=90)
    ax.set_title(location + ' ' + mode)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Simulation')
    ax.grid(True, color='k', linestyle='--', linewidth=0.5)
    plt.show()