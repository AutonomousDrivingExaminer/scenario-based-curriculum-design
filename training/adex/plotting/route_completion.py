import wandb
from matplotlib import pyplot as plt
from srunner.scenariomanager.traffic_events import TrafficEventType
from wandb.apis.public import Run


def plot_route_completion(ax, data):
    ts = data["total_timesteps"]
    eval_completion = data["eval/route_completion"]
    ts = ts[eval_completion.notnull()]
    eval_completion = eval_completion[eval_completion.notnull()]
    return ax.plot(ts, eval_completion)


def plot_rewards_completion(ax, data):
    ts = data["total_timesteps"]
    rewards = data["eval/episode_reward"]
    ts = ts[rewards.notnull()]
    rewards = rewards[rewards.notnull()]
    fig = ax.plot(ts, rewards)
    return fig

def plot_infractions(data):
    ts = data["total_timesteps"]
    infractions = [
        TrafficEventType.COLLISION_STATIC,
        TrafficEventType.COLLISION_VEHICLE,
        TrafficEventType.WRONG_WAY_INFRACTION,
        TrafficEventType.ON_SIDEWALK_INFRACTION,
        TrafficEventType.ROUTE_DEVIATION
    ]
    fig, ax = plt.subplots(1, len(infractions))
    ls = []
    for i, infraction in enumerate(infractions):
        infraction = f"infractions/{infraction.name}"
        infraction = data[infraction]
        l = ax[i].plot(ts, infraction)
        ls.append(l)

    plt.show()


def main():
    run_ids = {
        "dcd": "arllvz2c",
        "plr": "bkkrwti4",
        "dr": "k5441nkb"
    }
    api = wandb.Api()
    run: Run = api.run(f"adex-team/adex/{run_ids['dcd']}")
    data = run.history()
    fig, ax = plt.subplots(1, 3)
    fig.set_figwidth(10)
    fig.set_figheight(3)

    figs = []
    ax[0].set_ylim(0, 100)
    ax[0].set_xlabel("Timesteps")
    ax[0].set_title("Route Completion")
    ax[1].set_ylim(0, 150)
    ax[1].set_xlabel("Timesteps")
    ax[1].set_title("Rewards")
    ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("Timesteps")
    ax[2].set_title("Infractions")

    for i, (algo, run_id) in enumerate(run_ids.items()):
        run: Run = api.run(f"adex-team/adex/{run_id}")
        data = run.history()
        figs.append(plot_rewards_completion(ax[1], data))
        figs.append(plot_route_completion(ax[0], data))
        plot_rewards_completion(ax[1], data)
    lgd = fig.legend(figs,  labels=run_ids, ncols=3, loc='upper center', bbox_to_anchor=(0.5,-0.05))

    plt.savefig("ued_result.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plot_infractions(data)



if __name__ == '__main__':
    main()
