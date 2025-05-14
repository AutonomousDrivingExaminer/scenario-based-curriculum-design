import json
from glob import glob

import matplotlib
import numpy as np
import optree
from matplotlib import pyplot as plt
from scenic.domains.driving.roads import ManeuverType

matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
matplotlib.rcParams["axes.titlesize"] = 25
matplotlib.rcParams["axes.labelsize"] = 25
matplotlib.rcParams["axes.labelpad"] = 10
matplotlib.rcParams["xtick.labelsize"] = 20
matplotlib.rcParams["ytick.labelsize"] = 20
matplotlib.rcParams["legend.fontsize"] = 25


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def read_data(path):
    params = {}
    param_logs = glob(path)
    for param_log in param_logs:
        with open(param_log) as f:
            level_params = json.load(f)
        epoch = int(param_log.split("-")[-1].split(".")[0])
        params[epoch] = level_params
    return params


def main():

    fig, axs = plt.subplots(1, 3)


    fig.set_figwidth(21)
    fig.set_figheight(5)
    titles = ["Maneuver Type", r"Mean Target Speed $\uparrow$", r"Mean Number of NPCs $\uparrow$"]
    algos = ["DCD"]
    for algo_idx, algo in enumerate(["dcd"]):
        data = read_data(f"results/env_params/{algo}/train-*.json")
        data, epochs, stats = compute_stats(data)
        # Plotting
        epochs = np.array(epochs) * 2400
        plot_args = [
            (plot_maneuvers, [stats]),
           # (plot_ignore_others, [stats]),
            #(plot_ignore_traffic_lights, [stats]),
            (plot_numerics, [stats["mean_target_speed"], "Mean Target Speed", (0, 60)]),
            (plot_numerics, [stats["mean_num_npcs"], "Mean Number of NPCs", (0,5)]),
        ]
        for i, (fn, args) in enumerate(plot_args):
            if algo_idx == 0:
                axs[i].set_title(titles[i])
            if i == 0:
                axs[i].set_ylabel(algos[algo_idx])
            ax = axs[i]
            ax.set_xlim(0, 150_000)
            ax.set_xticklabels([f'{int(x) // 1000}K'.format() for x in ax.get_xticks().tolist()])
            fn(ax, epochs, *args)

    #plt.show()
    plt.savefig("ued_env_params.pdf")


def compute_stats(data):
    values = {}
    for epoch, config in data.items():
        config = optree.tree_map(lambda *xs: list(xs), *config)
        epoch_params = config["params"]
        num_npcs = []
        for i in range(6):
            counts = len(list(filter(lambda x: x == i, epoch_params["NUM_NPCS"])))
            num_npcs.append(counts)
        num_npcs = np.array(num_npcs) / len(epoch_params["NUM_NPCS"])

        # np.mean(epoch_params["NUM_NPCS"]), np.std(epoch_params["NUM_NPCS"])
        adv_params = epoch_params["NPC_PARAMS"]
        target_speed = []
        for i in range(1, 7):
            counts = len(list(filter(lambda x: x == i * 10, adv_params["target_speed"])))
            target_speed.append(counts)
        target_speed = np.array(target_speed) / len(adv_params["target_speed"])

        maneuver_types = []
        for m in range(1, len(ManeuverType)):
            counts = len(list(filter(lambda x: x == m, epoch_params["MANEUVER_TYPE"])))
            maneuver_types.append(counts)
        maneuver_type = np.array(maneuver_types) / len(epoch_params["MANEUVER_TYPE"])
        ignore_vehicles = 1 - sum(adv_params["ignore_vehicles"]) / len(adv_params["ignore_vehicles"])
        ignore_traffic_lights = sum(adv_params["ignore_traffic_lights"]) / len(adv_params["ignore_traffic_lights"])

        values[epoch] = {
            "num_npcs": num_npcs,
            "target_speed": target_speed,
            "maneuver_type": maneuver_type,
            "ignore_vehicles": ignore_vehicles,
            "ignore_traffic_lights": ignore_traffic_lights,
            "mean_target_speed": (np.mean(adv_params["target_speed"]), np.std(adv_params["target_speed"])),
            "mean_num_npcs": (np.mean(epoch_params["NUM_NPCS"]), np.std(epoch_params["NUM_NPCS"])),
        }
    epochs = sorted(values.keys())
    stats = [values[t] for t in epochs]
    stats = optree.tree_map(lambda *xs: np.array(xs), *stats)
    stats = optree.tree_map(lambda x: (x[1:] + x[:-1]) / 2, stats)
    stats = optree.tree_map(lambda x: x.tolist(), stats)
    epochs = epochs[1:]
    return data, epochs, stats


def plot_maneuvers(ax, epochs, stats, *args):
    stats = stats["maneuver_type"]
    names = [ManeuverType(m).name.lower() for m in range(1, 4)]
    colors = ["red", "green", "blue"]
    shares = list(zip(*stats))
    ax.stackplot(epochs, *shares, labels=names, alpha=0.4)
    ax.legend(loc='lower right')


def plot_ignore_others(ax, epochs, stats):
    data = stats["ignore_vehicles"]
    ax.plot(epochs, data)
    ax.set_ylim(0, 1)
    ax.fill_between(epochs, np.array(data), 0, alpha=0.2)


def plot_ignore_traffic_lights(ax, epochs, stats):
    data = stats["ignore_traffic_lights"]
    ax.plot(epochs, data)
    ax.set_ylim(0, 1)
    ax.fill_between(epochs, np.array(data), 0, alpha=0.2)


def plot_numerics(ax, epochs, stats, title, limits):
    mean, std = stats
    ax.plot(epochs, mean)
    ax.set_ylim(*limits)
    ax.fill_between(epochs, np.array(mean) - np.array(std), np.array(mean) + np.array(std), alpha=0.2)


if __name__ == '__main__':
    main()
