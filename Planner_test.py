import irsim
from irsim.lib.path_planners.a_star import AStarPlanner
from irsim.lib.path_planners.probabilistic_road_map import PRMPlanner
from irsim.lib.path_planners.rrt import RRT
from irsim.lib.path_planners.rrt_star import RRTStar


# @pytest.mark.parametrize(
#     "planner, resolution",
#     [
#         (AStarPlanner, 0.3),
#         (RRTStar, 0.3),
#         (RRT.py, 0.3),
#         (PRMPlanner, 0.3),
#     ],
# )
def path_planners(planner, resolution):
    env = irsim.make(
        "test.yaml", save_ani=False, full=False, display=True
    )
    env_map = env.get_map()
    planner = planner(env_map, resolution)
    robot_info = env.get_robot_info()
    robot_state = env.get_robot_state()
    trajectory = planner.planning(robot_state, robot_info.goal)
    env.draw_trajectory(trajectory, traj_type="r-")
    env.end()
path_planners(AStarPlanner, 1)