from __future__ import annotations
param map = localPath('./maps/Town05.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town05'
model scenic.simulators.carla.model

param DISTANCE_TO_INTERSECTION = Range(-25, -5)
param NUM_AGENTS = 4

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
maneuvers = filter(lambda m: m.type != ManeuverType.U_TURN, maneuvers)


for i in range(globalParameters.NUM_AGENTS):
    maneuver = Uniform(*maneuvers)
    route = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]
    maneuvers.remove(maneuver)
    vehicle = RouteFollowingCar following roadDirection from maneuver.startLane.centerline[-1] for globalParameters.DISTANCE_TO_INTERSECTION,
        with route route,
        with rolename f"vehicle_{i}",
        with color Color(0,1,0),
        with blueprint "vehicle.tesla.model3"
    vehicles.append(vehicle)

ego = vehicles[0]


monitor TrafficLights:
    freezeTrafficLights()
    while True:
        setClosestTrafficLightStatus(ego, "green")
        wait


terminate after 15 seconds