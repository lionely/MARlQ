Q-learning Function: ql_box, boxSize==3
Epsilon Policy:
        epsilon_floor = 0.3
        epsilons = [0.7] * (int(TOTAL_DIST/100)+1)
        lastDist = pu.getLastDist(funcName)
        epsilon_lastDist_i = int(lastDist/100)
        #TODO: There might be a more effective way to do this!
        for i in range(epsilon_lastDist_i):
            if (epsilon_lastDist_i - i) > 4:
                epsilons[i] = epsilon_floor
            else:
                epsilons[i] -= 0.1 * (epsilon_lastDist_i - i)

        # Choose which epsilon depending on distance

            epsilon_index = int(info['distance']/100)
            #print(epsilon_index)
            epsilon = epsilons[epsilon_index]

Standing Penalty: True, 0.10

Action Dictionary:
        action_dict = {'up': [1, 0, 0, 0, 0, 0],
                   'L': [0, 1, 0, 0, 0, 0],
                   'down': [0, 0, 1, 0, 0, 0],
                   'R': [0, 0, 0, 1, 0, 0],
                   'JUMP': [0, 0, 0, 0, 1, 0],
                   'R_JUMP1': [0, 0, 0, 1, 1, 0],
                   'R_JUMP2': [0, 0, 0, 1, 1, 0],
                   'R_JUMP3': [0, 0, 0, 1, 1, 0]}