import gym
import skfuzzy as fuzz

import numpy as np
env = gym.make('MountainCarContinuous-v0')

def func_fuzzy(observation):
    # Generate universe variables
    x_Position = np.arange(-1.2, 0.7, 0.1)
    x_Velocity = np.arange(-0.07, 0.08, 0.01)
    x_action  = np.arange(-1, 1.1, 0.1)

    # Generate fuzzy membership functions
    Position_far = fuzz.trimf(x_Position, [-1, -1, -0.5])
    Position_Vfar = fuzz.trimf(x_Position, [-1.2, -1.2, -1])
    Position_Vnear = fuzz.trimf(x_Position, [0.4, 0.6, 0.6])
    Position_near = fuzz.trimf(x_Position, [-0.5, 0, 0.4])
    Velocity_lo = fuzz.trimf(x_Velocity, [-0.07, -0.07, 0])
    Velocity_md = fuzz.trimf(x_Velocity, [-0.07, 0, 0.07])
    Velocity_hi = fuzz.trimf(x_Velocity, [0, 0.07, 0.07])
    action_Vright = fuzz.trimf(x_action, [-1, -1, -0.8])
    action_right = fuzz.trimf(x_action, [-0.2, -0.1, 0])
    action_left = fuzz.trimf(x_action, [0, 0.1, 0.2])
    action_Vleft = fuzz.trimf(x_action, [0.8, 1, 1])

    # We need the activation of our fuzzy membership functions at observation values.
    Position_level_far = fuzz.interp_membership(x_Position, Position_far, observation[0])
    Position_level_Vfar = fuzz.interp_membership(x_Position, Position_Vfar, observation[0])
    Position_level_Vnear = fuzz.interp_membership(x_Position, Position_Vnear, observation[0])
    Position_level_near = fuzz.interp_membership(x_Position, Position_near, observation[0])
    Velocity_level_lo = fuzz.interp_membership(x_Velocity, Velocity_lo, observation[1])
    Velocity_level_md = fuzz.interp_membership(x_Velocity, Velocity_md, observation[1])
    Velocity_level_hi = fuzz.interp_membership(x_Velocity, Velocity_hi, observation[1])

    # Now we take our rules and apply them.
    active_rule1 = np.fmin(Position_level_near, Velocity_level_hi)
    r1 = np.fmin(active_rule1, action_left)

    active_rule2 = np.fmin(Position_level_near, Velocity_level_lo)
    r2 = np.fmin(active_rule2, action_right)

    active_rule3 = np.fmin(Position_level_far, Velocity_level_lo)
    r3 = np.fmin(active_rule3, action_right)

    active_rule4 = np.fmin(Position_level_far, Velocity_level_hi)
    r4 = np.fmin(active_rule4, action_left)

    active_rule1 = np.fmin(Position_level_Vnear, Velocity_level_hi)
    r5 = np.fmin(active_rule1, action_Vleft)

    active_rule4 = np.fmin(Position_level_Vfar, Velocity_level_hi)
    r6 = np.fmin(active_rule4, action_Vleft)


    # Aggregate all output membership functions together
    aggregated = np.fmax(r1,np.fmax(r2, np.fmax(r3,np.fmax(r4,np.fmax(r5,r6)))))

    # Calculate defuzzified result
    Action = fuzz.defuzz(x_action, aggregated, 'mom')

    return Action

for i_episode in range(1):
    observation = env.reset()
    action = 0.8
    for t in range(1000):
        env.render()
        observation, reward, done, info = env.step([action])
        action=func_fuzzy(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

