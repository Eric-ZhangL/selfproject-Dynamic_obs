import irsim
import numpy as np

# env = irsim.make('./env/moving_mb.yaml') # initialize the environment with the configuration file
env = irsim.make('/home/zhangl/DRL_project/Dynamic_obs/self_env/ir_sim_test/robot_world_mbv3.yaml') # initialize the environment with the configuration file
# env = irsim.make('env/moving.yaml') # initialize the environment with the configuration file

    
for step in range(300):
    env.step()



    # goal= env.robot_list[1]._goal
    # tx= goal[0][1]
    # print(">>> 调试信息：读取到的 goal =", tx)
    # goal1 = env.robot_list[0].done()
    # goal2 = env.robot_list[1].done()
    # goal3 = env.robot_list[2].done() 
    # env.robot_list[2].set_state([1,1,2,0])

    # robot_state0 = env.robot_list   
    # robot_state0 = env.robot_list[0]._goal
    # robot_state1 = env.robot_list[1]._state
    # robot_state2 = env.robot_list[2]._state
    # # robot_state1 = env.get_robot_state()
    # print(step)
    # print(goal1)
    # print(goal2)
    # print(goal3)
    # print(env.robot_list[0].done())
    # print(robot_state1)
    env.render() # render the environment

    if env.done(): 
        break # check if the simulation is done
    actions = [np.array([0.3, 0.3]),0]
    env.step(actions)  #
env.end() # close the environment