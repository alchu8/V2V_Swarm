import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def plot_peak_reach(radius_list, v2v, averages_peak_reach, averages_peak_area, averages_time):
    # Path to save the plot
    project_dir = Path(__file__).resolve().parent
    images_dir = project_dir/"images"
    images_dir.mkdir(parents=True, exist_ok=True)  # Create the images directory if it doesn't exist yet
    save_path = images_dir/f"peak_reach_v2v_{v2v}.png"

    fig, axes = plt.subplots(len(radius_list), 2, figsize=(14, 6*len(radius_list)), subplot_kw={'projection': '3d'}, squeeze=False)
    # fig.suptitle('v2v: ' + str(v2v))
    fig.suptitle(f"Swarm Utility & Exploration Analysis (V2V: {v2v})", fontsize=14, fontweight='bold')
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
        fig.colorbar(plot1, ax=axes[i, 0], shrink=0.5, aspect=10, pad=0.1)
        plot2 = axes[i, 1].scatter(x_time, y_time, z_time, c=z_time, cmap='viridis', s=40)
        axes[i, 1].set_xlabel('Num Robots')
        axes[i, 1].set_ylabel('Num Refuelers')
        axes[i, 1].set_zlabel('Time Past Solo per capita (s)')
        axes[i, 1].set_title(f'Time Past Solo Reach per Robot (r={radius})')
        fig.colorbar(plot2, ax=axes[i, 1], shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_run(result):
    """Static post-mortem plot of a single run."""
    # Path to save the plot
    project_dir = Path(__file__).resolve().parent
    images_dir = project_dir/"images"
    images_dir.mkdir(parents=True, exist_ok=True)  # Create the images directory if it doesn't exist yet
    save_path = images_dir/"run.png"

    config = result['config']
    L = config.world_size
    depot = np.array([L / 2, L / 2])
    solo_reach = result['solo_reach']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(depot[0] + solo_reach * np.cos(theta),
            depot[1] + solo_reach * np.sin(theta),
            'k--', alpha=0.3, label=f'Solo-reach (1/2alpha = {solo_reach:.0f})')
    ax.scatter(*depot, s=200, c='darkgreen', marker='o', label='Depot', zorder=10)

    pos = result['final_pos']
    mode = result['final_mode']
    for m, (c, lab) in enumerate(zip(['steelblue', 'firebrick', 'forestgreen', 'dimgray'],
                                     ['Wandering', 'Heading home', 'Docked', 'Lost (immobilized)'])):
        mask = mode == m
        if mask.any():
            ax.scatter(pos[mask, 0], pos[mask, 1], c=c, s=60, alpha=0.7, label=lab)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Final positions')
    ax.grid(alpha=0.3)

    ax = axes[1]
    if result['soc_trace']:
        steps = np.array([s[0] for s in result['soc_trace']])
        socs = np.array([s[1] for s in result['soc_trace']])
        for i in range(socs.shape[1]):
            ax.plot(steps, socs[:, i], alpha=0.3, linewidth=0.8)
        ax.plot(steps, socs.mean(axis=1), 'k-', linewidth=2, label='Mean SoC')
    ax.axhline(config.low_soc_threshold, color='red', linestyle='--', alpha=0.5, label='Seek-charge threshold')
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('State of charge')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title('SoC traces (gray = individuals; black = mean)')
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'N={config.num_robots}  alpha={config.alpha:.4f}  '
        f'V2V={"on" if config.v2v_enabled else "off"}  |  '
        f'Tasks done={result["tasks_done"]}  '
        f'Peak reach={result["peak_reach"]:.2f}x  '
        f'Time @ outer 80%={result["fraction_at_outer80"]*100:.1f}%',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def mystery_plot():
    # fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # fig.suptitle('v2v: ' + str(v2v))
    # ax_mpr = axes[0,0]
    # ax_tps = axes[0,1]
    # ax_td = axes[1,0]
    # ax_te = axes[1,1]
    # ax_mpr.plot(num_robots_list, mean_peak_reach)
    # ax_mpr.set_xlabel('fleet size')
    # ax_mpr.set_ylabel('mean peak reach')
    # ax_mpr.set_title('mean peak reach')
    # ax_tps.plot(num_robots_list, time_past_solo)
    # ax_tps.set_xlabel('fleet size')
    # ax_tps.set_ylabel('time past solo')
    # ax_tps.set_title('time past solo')
    # ax_td.plot(num_robots_list, tasks_done)
    # ax_td.set_xlabel('fleet size')
    # ax_td.set_ylabel('tasks done')
    # ax_td.set_title('tasks done')
    # ax_te.plot(num_robots_list, tasks_expired)
    # ax_te.set_xlabel('fleet size')
    # ax_te.set_ylabel('tasks expired')
    # ax_te.set_title('tasks expired')
    # figs.append((fig, axes))
    # plt.tight_layout()
    # plt.show()
    # fig = plt.figure(figsize=(14, 6))
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # scatter = ax1.scatter(x, y, z, c=z, cmap='viridis', s=40)

    # plt.tight_layout()
    # plt.show()
    pass
