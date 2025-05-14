import matplotlib
import numpy as np
import wandb
from matplotlib import pyplot as plt

matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
matplotlib.rcParams["axes.titlesize"] = 25
matplotlib.rcParams["axes.labelsize"] = 25
matplotlib.rcParams["axes.labelpad"] = 10
matplotlib.rcParams["xtick.labelsize"] = 15
matplotlib.rcParams["ytick.labelsize"] = 20
matplotlib.rcParams["legend.fontsize"] = 25

FIGSIZE = (21, 5)
ALPHA_STD = 0.25
MAX_X = 150_000
BINNING = 1


def plot_route_completion(ax, data, color=None, label=""):
    epochs = min([len(d["episode_reward"]) for d in data])
    route_completion = np.stack(d["episode_reward"][:epochs] for d in data)
    total_timesteps = np.stack([d["total_timesteps"][:epochs] for d in data])

    # std_route_completion = data["train/std_route_completion"]
    ts = np.arange(epochs)

    # binning
    mean_route_completion = np.array(
        [np.mean(route_completion[:, i:i + BINNING]) for i in range(0, epochs, BINNING)])
    std_route_completion = np.array(
        [np.std(route_completion[:, i:i + BINNING]) for i in range(0, epochs, BINNING)])
    ts = np.array(
        [np.mean(total_timesteps[:, i:i + BINNING]) for i in range(0, epochs, BINNING)]).astype(int)

    fig = ax.plot(ts, mean_route_completion, color=color, label=label)
    fig = ax.fill_between(ts, mean_route_completion - std_route_completion,
                          mean_route_completion + std_route_completion, alpha=ALPHA_STD)
    return fig


def plot_eval_route_completion(ax, data, color=None, label=""):
    key = "eval/route_completion"
    epochs = min([len(d["total_timesteps"]) for d in data])
    total_timesteps = [d["total_timesteps"][:epochs] for d in data]
    route_completion = [d[key][:epochs] for d in data]
    total_timesteps = np.stack(
        [ts[c.notnull()] for ts, c in zip(total_timesteps, route_completion)])
    route_completion = np.stack([c[c.notnull()] for c in route_completion])

    bins = 2
    mean = np.array([np.mean(route_completion[:, i:i + bins]) for i in range(0, epochs, bins)])
    std = np.array([np.std(route_completion[:, i:i + bins]) for i in range(0, epochs, bins)])
    ts = np.array([np.mean(total_timesteps[:, i:i + bins]) for i in range(0, epochs, bins)]).round().astype(int)

    fig = ax.plot(ts, mean, color=color, label=label)
    fig = ax.fill_between(ts, mean - std, mean + std, alpha=ALPHA_STD)
    return fig


def plot_rewards_completion(ax, data, color=None, label="", eval=False):
    epochs = min([len(d["episode_reward"]) for d in data])
    rewards = np.stack([d["episode_reward"][:epochs] for d in data])
    total_timesteps = np.stack([d["total_timesteps"][:epochs] for d in data])
    ts = np.arange(epochs)

    # binning
    mean_rewards = np.array([np.mean(rewards[:, i:i + BINNING]) for i in range(0, epochs, BINNING)])
    std_rewards = np.array([np.std(rewards[:, i:i + BINNING]) for i in range(0, epochs, BINNING)])
    ts = np.array(
        [np.mean(total_timesteps[:, i:i + BINNING]) for i in range(0, epochs, BINNING)]).astype(int)

    fig = ax.plot(ts, mean_rewards, color=color, label=label)
    fig = ax.fill_between(ts, mean_rewards - std_rewards, mean_rewards + std_rewards,
                          alpha=ALPHA_STD)
    return fig


def plot_all_collisions(ax, data, color=None, label=""):
    epochs = min([len(d["episode_reward"]) for d in data])
    total_timesteps = np.stack([d["total_timesteps"][:epochs] for d in data])
    vehicles = [d["infractions/COLLISION_VEHICLE"][:epochs] for d in data]
    static = [d["infractions/COLLISION_STATIC"][:epochs] for d in data]
    collisions = np.stack([v + s for v, s in zip(vehicles, static)])


    xs, ys = [], []
    for i in range(0, epochs, BINNING):
        mean = np.mean(collisions[:, i:i + BINNING])
        std = np.std(collisions[:, i:i + BINNING])
        ys.append((mean, std))
        xs.append(np.mean(total_timesteps[:, i:i + BINNING]))

    stats = np.array(ys)
    mean, std = stats[:, 0], stats[:, 1]
    fig = ax.plot(xs, mean, color=color, label=label)
    fig = ax.fill_between(xs, mean - std, mean + std, alpha=ALPHA_STD)
    return fig


plotting_fns = {
    "route_completion": plot_route_completion,
    "episode_reward": plot_rewards_completion,
    "all_collisions": plot_all_collisions,
    "eval_route_completion": plot_eval_route_completion,
}


def main():
    labels_dict = {
        "dr": "DR",
        "plr": "PLR",
        "dcd": "DCD",

        "episode_reward": "Episode Reward (Train)",
        "route_completion": "Route Completion (Train)",
        "all_collisions": "Collision Rate",
        "eval_route_completion": "Route Completion (Eval)",
    }

    colors = {
        "dr": "#1f77b4",
        "plr": "#ff7f0e",
        "dcd": "#2ca02c",
    }

    limits = {
        "episode_reward": (0, 150),
        "route_completion": (0, 100),
        "all_collisions": (0, 1),
        "eval_route_completion": (0, 100),
    }


    run_ids = {
        "dr": ["k5441nkb", "otzmvo9k", "d6vqs3sl"],
        "plr": ["bkkrwti4", "k1udo4im", "gdpkakyc"],
        "dcd": ["arllvz2c", "64c6apjn", "jr3afidy"],
    }

    ids_to_plot = ["dr", "plr", "dcd"]
    metrics = ["episode_reward", "route_completion", "eval_route_completion", ]

    api = wandb.Api()
    fig, axes = plt.subplots(1, len(metrics), figsize=FIGSIZE, sharex=True)

    figs = []

    for i, (algo, run_ids) in enumerate(run_ids.items()):
        if algo not in ids_to_plot:
            continue
        runs = [api.run(f"adex-team/adex/{run_id}") for run_id in run_ids]
        data = [run.history() for run in runs]

        for metric, ax in zip(metrics, axes):
            plotting_fn = plotting_fns[metric]
            figs.append(plotting_fn(ax, data, color=colors[algo], label=labels_dict[algo]))
            ax.set_xticklabels([f'{int(x)//1000}K'.format() for x in ax.get_xticks().tolist()])
            ax.set_xlabel("Timesteps")
            ax.set_title(labels_dict[metric])
            ax.set_xlim(0, MAX_X)
            ax.set_ylim(*limits[metric])

    # legend outside of plot
    plt.subplots_adjust(top=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center',
                     bbox_to_anchor=(.5, -0.1), ncol=len(labels), fancybox=True, shadow=True)

    plt.savefig(f"ued_scores.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    main()
