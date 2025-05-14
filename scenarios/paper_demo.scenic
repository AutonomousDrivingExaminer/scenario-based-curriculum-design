param MANEUVER = LEFT_TURN
param NUM_ADVERSARIES = 2
param NUM_PEDESTRIANS = 3
param ADV_PARAMS = {}
...
intersection = Uniform(filter(...))
start_maneuver = Uniform(filter(...))
route ...
ego = RouteFollowingCar at ...,
    with route route,
    with rolename "student",
    with blueprint "vehicle.tesla.model3"
...
for i in range(NUM_ADVERSARIES):
    maneuver = Uniform(...)
    route = Uniform(...)
    adv = RouteFollowingCar at ...,
        with route route,
        with rolename f"adv_{i}"
terminate after 20 seconds