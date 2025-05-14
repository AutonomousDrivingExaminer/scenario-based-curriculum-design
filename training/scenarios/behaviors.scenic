from __future__ import annotations
import carla
from agents.navigation.basic_agent import BasicAgent
from scenic.simulators.carla.utils import utils
from scenic.simulators.carla.actions import SetThrottleAction, SetSteerAction, SetBrakeAction


behavior BasicAgentBehavior(opt_dict: dict = {}):
    vehicle = self.carlaActor
    target_speed = opt_dict.get('target_speed', 30)
    controller = BasicAgent(vehicle, target_speed=target_speed, opt_dict=opt_dict)
    
    if self.route is not None:
        world = vehicle.get_world()
        map = world.get_map()
        current_waypoint = map.get_waypoint(vehicle.get_location())
        end_lane = self.route[-1]
        target_location = utils.scenicToCarlaLocation(end_lane.centerline[-1], z=0)
        target_waypoint = map.get_waypoint(target_location)
        plan = controller.trace_route(current_waypoint, target_waypoint)
        controller.set_global_plan(plan)

    while True:
        control = controller.run_step()
        take (
            SetThrottleAction(control.throttle),
            SetSteerAction(control.steer),
            SetBrakeAction(control.brake)
        )
