from __future__ import annotations
param map = localPath('./maps/Town05.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town05'
model scenic.simulators.carla.model

param DISTANCE_TO_INTERSECTION = Range(-25, -5)
param NUM_ADVERSARIES = 0
param NUM_NPCS = 0

class RouteFollowingCar(Car):
    route: list[Lane]

def is_4way_intersection(inter) -> bool:
    left_turns = filter(lambda i: i.type == ManeuverType.LEFT_TURN, inter.maneuvers)
    all_single_lane = all(len(lane.adjacentLanes) == 1 for lane in inter.incomingLanes)
    return len(left_turns) >=4 and inter.is4Way and all_single_lane


vehicles = []
four_way_intersections = filter(lambda i: i.is4Way and i.isSignalized, network.intersections)
intersection = Uniform(*four_way_intersections)
maneuvers = intersection.maneuvers

maneuver = Uniform(*maneuvers)
route = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]
ego = RouteFollowingCar following roadDirection from maneuver.startLane.centerline[-1] for globalParameters.DISTANCE_TO_INTERSECTION,
    with route route,
    with rolename f"student",
    with color Color(0,1,0),
    with blueprint "vehicle.tesla.model3"

for i in range(globalParameters.NUM_ADVERSARIES):
    maneuver = Uniform(*maneuvers)
    maneuvers.remove(maneuver)
    vehicle = Car following roadDirection from maneuver.startLane.centerline[-1] for globalParameters.DISTANCE_TO_INTERSECTION,
        with rolename f"adv_{i}",
        with color Color(1,0,0)
    vehicles.append(vehicle)

for i in range(globalParameters.NUM_NPCS):
    lane = Uniform(*intersection.incomingLanes)
    distance = resample(globalParameters.DISTANCE_TO_INTERSECTION)
    vehicle = Car following roadDirection from lane.centerline[-1] for distance,
        with rolename f"npc_{i}",
        with color Color(0,0,1),
        with behavior AutopilotBehavior()
    vehicles.append(vehicle)

monitor TrafficLights:
    freezeTrafficLights()
    while True:
        setClosestTrafficLightStatus(ego, "green")
        wait

