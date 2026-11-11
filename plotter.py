import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def plot_metrics(radius_list, v2v, averages_peak_reach, averages_peak_area, averages_time):
    # Path to save the plot
    project_dir = Path(__file__).resolve().parent
    images_dir = project_dir/"images"
    images_dir.mkdir(parents=True, exist_ok=True)  # Create the images directory if it doesn't exist yet
    save_path = images_dir/f"v2v_{v2v}.png"

    fig, axes = plt.subplots(len(radius_list), 2, figsize=(14, 6*len(radius_list)), subplot_kw={'projection': '3d'}, squeeze=False)
    fig.suptitle(f"V2V: {v2v}", fontsize=14, fontweight='bold')
    for i in range(len(radius_list)):
        radius = radius_list[i]
        # Create 3D plots with constant v2v and radius
        filtered_peak_reach_averages = {k: averages_peak_reach[k] for k, _ in averages_peak_reach.items() if
                                        k[3] == radius and k[0] == v2v}
        filtered_peak_area_averages = {k: averages_peak_area[k] for k, _ in averages_peak_area.items() if
                                       k[3] == radius and k[0] == v2v}
        filtered_time_averages = {k: averages_time[k] for k, _ in averages_time.items() if
                                  k[3] == radius and k[0] == v2v}
        x_peak_reach = []
        y_peak_reach = []
        z_peak_reach = []
        x_time = []
        y_time = []
        z_time = []
        x_peak_area = []
        y_peak_area = []
        z_peak_area = []
        for key, value in filtered_peak_reach_averages.items():
            x_peak_reach.append(key[1])  # num_robots
            y_peak_reach.append(key[2])  # num_refuelers
            z_peak_reach.append(value)
        for key, value in filtered_time_averages.items():
            x_time.append(key[1])
            y_time.append(key[2])
            z_time.append(value)
        for key, value in filtered_peak_area_averages.items():
            x_peak_area.append(key[1])
            y_peak_area.append(key[2])
            z_peak_area.append(value)
        plot1 = axes[i, 0].scatter(x_peak_reach, y_peak_reach, z_peak_reach, c='blue', label='Peak Reach')
        axes[i, 0].scatter(x_peak_area, y_peak_area, z_peak_area, c='green', label='Peak Area')
        axes[i, 0].legend()
        axes[i, 0].set_xlabel('Num Robots')
        axes[i, 0].set_ylabel('Num Refuelers')
        axes[i, 0].set_zlabel('x solo reach or world')
        axes[i, 0].set_title(f'Peak Reach and Area Coverage (r={radius})')
        plot2 = axes[i, 1].scatter(x_time, y_time, z_time, c=z_time, cmap='viridis', s=40)
        axes[i, 1].set_xlabel('Num Robots')
        axes[i, 1].set_ylabel('Num Refuelers')
        axes[i, 1].set_zlabel('Time Past Solo per capita (s)')
        axes[i, 1].set_title(f'Time Past Solo Reach per Robot (r={radius})')
        fig.colorbar(plot2, ax=axes[i, 1], shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
