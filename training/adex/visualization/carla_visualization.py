from __future__ import annotations

import math

import carla
from srunner.scenarios.basic_scenario import BasicScenario

from adex_gym.wrappers.carla_visualization import VisualizationCallback


def make_route_visualization(agent: str,
                             color: tuple[int, int, int] = None) -> VisualizationCallback:
    color = color or (0, 5, 0)

    def visualize_route(scenario: BasicScenario, world: carla.World) -> None:
        config = [cfg for cfg in scenario.config.ego_vehicles if cfg.rolename == agent][0]
        map = world.get_map()
        for tf, _ in config.route:
            wpt = map.get_waypoint(tf.location)
            wpt_t = wpt.transform
            begin = wpt_t.location + carla.Location(z=0.2)
            angle = math.radians(wpt_t.rotation.yaw)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            world.debug.draw_arrow(
                begin=begin,
                end=end,
                arrow_size=0.05,
                color=carla.Color(*color),
                life_time=100
            )

    return visualize_route
