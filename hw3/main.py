from __future__ import division
from os import link
import sim
import pybullet as p
import random
import numpy as np
import math
import time

MAX_ITERS = 10000
delta_q = 0.1

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)


def rrt_semi_random_sample(q_goal, steer_goal_p):
    if random.random() < steer_goal_p:
        return q_goal.copy()
    else:
        bias = np.random.uniform(-np.pi/8, np.pi/8, size=len(q_goal)).tolist()
        return [q_goal[i] + bias[i] for i in range(len(q_goal))]


def rrt_nearest(V, q_rand):
    nearest = V[0]
    nearest_idx = 0
    nearest_dist = float("inf")
    for i, v in enumerate(V):
        dist_square = sum(np.power(v[j] - q_rand[j], 2) for j in range(len(v)))
        if dist_square < nearest_dist:
            nearest, nearest_idx, nearest_dist = v, i, dist_square

    return nearest, nearest_idx, np.sqrt(nearest_dist)


# TODO: I drop the delta_q and use a different metric
def rtt_steer(q_rand, q_nearest, delta_q):
    q_new = q_nearest.copy()
    for i in range(len(q_rand)):
        q_new[i] = (1 - 0.2) * q_nearest[i] + 0.2 * q_rand[i]

    return q_new


def rrt_get_path(adj, q_init_idx, q_goal_idx, path):
    path.append(q_init_idx)
    if q_init_idx == q_goal_idx:
        return path

    for v in adj[q_init_idx]:
        if v not in path:
            new_path = rrt_get_path(adj, v, q_goal_idx, path)
            return new_path

    return None


def rrt_find_path(V, E, q_init_idx, q_goal_idx):
    print("-----------")
    for e in E: print(str(e[0]) + " " + str(e[1]))

    # convert edges to adjacent list
    adj = []
    for i in range(len(V)):
        successors = []
        for e in E:
            if e[0] == i: successors.append(e[1])
        adj.append(successors)
    # for i in range(len(adj)):
    #     print(str(i) + ":")
    #     print(adj[i])

    # BFS to find a path
    path = rrt_get_path(adj, q_init_idx, q_goal_idx, [])
    # print("============")
    # if path is None:
    #     print("Cannot find a path")
    # else:
    #     for p in path: print(p)
    return None if not path else [V[i] for i in path]


def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: series of joint angles
    """
    # ========== PART 3 =========
    # TODO: Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    V, E = [q_init], []
    for i in range(MAX_ITERS):
        q_rand = rrt_semi_random_sample(q_goal, steer_goal_p)
        q_nearest, q_nearest_idx, _ = rrt_nearest(V, q_rand)
        q_new = rtt_steer(q_rand, q_nearest, delta_q)
        if not env.check_collision(q_new):
            V.append(q_new)
            q_new_idx = len(V) - 1
            E.append((q_nearest_idx, q_new_idx))
            if sum(np.power(q_new[j] - q_goal[j], 2) for j in range(len(q_new))) < np.power(delta_q, 2):
                V.append(q_goal)
                q_goal_idx = len(V) - 1
                E.append((q_new_idx, q_goal_idx))
                path = rrt_find_path(V, E, 0, q_goal_idx)
                return path

    return None


def get_grasp_position_angle(object_id):
    position, grasp_angle = np.zeros((3, 1)), 0
    # ========= PART 2============
    # Get position and orientation (yaw in radians) of the gripper for grasping
    position, tmp = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(tmp)[2]
    # ==================================
    return position, grasp_angle


if __name__ == "__main__":
    random.seed(1)
    object_shapes = [
        "assets/objects/cube.urdf",
    ]
    env = sim.PyBulletSim(object_shapes = object_shapes)
    num_trials = 3

    # PART 1: Basic robot movement
    # Implement env.move_tool function in sim.py. More details in env.move_tool description
    passed = 0
    for i in range(num_trials):
        # Choose a reachable end-effector position and orientation
        random_position = env._workspace1_bounds[:, 0] + 0.15 + \
            np.random.random_sample((3)) * (env._workspace1_bounds[:, 1] - env._workspace1_bounds[:, 0] - 0.15)
        random_orientation = np.random.random_sample((3)) * np.pi / 4 - np.pi / 8
        random_orientation[1] += np.pi
        random_orientation = p.getQuaternionFromEuler(random_orientation)
        marker = sim.SphereMarker(position=random_position, radius=0.03, orientation=random_orientation)
        # Move tool
        env.move_tool(random_position, random_orientation)
        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
        link_marker = sim.SphereMarker(link_state[0], radius=0.03, orientation=link_state[1], rgba_color=[0, 1, 0, 0.8])
        # Test position
        delta_pos = np.max(np.abs(np.array(link_state[0]) - random_position))
        delta_orn = np.max(np.abs(np.array(link_state[1]) - random_orientation))
        if  delta_pos <= 1e-3 and delta_orn <= 1e-3:
            passed += 1
        env.step_simulation(1000)
        # Return to robot's home configuration
        env.robot_go_home()
        del marker, link_marker
    print(f"[Robot Movement] {passed} / {num_trials} cases passed")

    # PART 2: Grasping
    passed = 0
    env.load_gripper()
    for _ in range(num_trials):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)

        # Test for grasping success (this test is a necessary condition, not sufficient):
        object_z = p.getBasePositionAndOrientation(object_id)[0][2]
        if object_z >= 0.2:
            passed += 1
        env.reset_objects()
    print(f"[Grasping] {passed} / {num_trials} cases passed")

    # PART 3: RRT Implementation
    passed = 0
    for _ in range(num_trials):
        # grasp the object
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            # get a list of robot configuration in small step sizes
            path_conf = rrt(env.robot_home_joint_config,
                            env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5, env)
            if path_conf is None:
                print(
                    "no collision-free path is found within the time budget. continuing ...")
            else:
                # TODO: Execute the path while visualizing the location of joint 5 (see Figure 2 in homework manual)
                # - For visualization, you can use sim.SphereMarker
                # ===============================================================================
                for i in range(len(path_conf)):
                    if i != 0:
                        env.move_joints(path_conf[i])
                        visualize_path(path_conf[i - 1], path_conf[i], env)
                # ===============================================================================
                print("Path executed. Dropping the object")

                # TODO: Drop the object
                # - Hint: open gripper, wait, close gripper
                # ===============================================================================
                env.open_gripper()
                env.close_gripper()
                # ===============================================================================

                # TODO: Retrace the path to original location
                # ===============================================================================

                # ===============================================================================
            p.removeAllUserDebugItems()

        env.robot_go_home()

        # Test if the object was actually transferred to the second bin
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and\
            object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and\
            object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()

    print(f"[RRT Object Transfer] {passed} / {num_trials} cases passed")