import time
import random


def performance_benchmark(env):
    print("Starting performance benchmark")
    cycles = 0
    turn = 0
    env.reset()
    start = time.time()
    end = 0

    while True:
        cycles += 1
        for agent in env.agent_iter(env.num_agents):  # step through every agent once with observe=True
            obs, reward, done, info = env.last()
            if done:
                action = None
            elif 'legal_moves' in env.infos[agent]:
                action = random.choice(env.infos[agent]['legal_moves'])
            else:
                action = env.action_spaces[agent].sample()
            env.step(action)
            turn += 1

            if all(env.dones.values()):
                env.reset()

        if time.time() - start > 5:
            end = time.time()
            break

    length = end - start

    turns_per_time = turn / length
    cycles_per_time = cycles / length
    print(str(turns_per_time) + " turns per second")
    print(str(cycles_per_time) + " cycles per second")
    print("Finished performance benchmark")
