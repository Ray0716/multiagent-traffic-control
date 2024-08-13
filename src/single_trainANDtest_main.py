import aienvs
from multi_DQRN import AgentDQN
import yaml
import logging
import pdb
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import numpy as np
import os
import pdb
import csv
from collections import OrderedDict

def saver(results, index, save_dir):
    if index is not None:
        index = str(index)
        save_dir = os.path.join(save_dir, index)
        os.makedirs(save_dir, exist_ok=True)
    for key in results.keys():
        name = str(key) + ".csv"
        filename = os.path.join(save_dir, name)
        outfile = open(filename, 'w')
        writer = csv.writer(outfile)
        writer.writerows(map(lambda x:[x], results[key]))

def saverSingle(data, name, save_dir):
    name = str(name)
    filename = os.path.join(save_dir, name+'.csv')
    outfile = open(filename, 'w')
    writer = csv.writer(outfile)
    writer.writerows(map(lambda x:[x], data))

def makeChannelled(item, index: int, repeat: int = 0):
    item = np.expand_dims(item, index)
    if repeat:
        item = np.concatenate( (item,)*repeat, axis = index )
    return item.copy()

def evaluateAgent(agent, env, num_sim, test_index, save_dir):
    result_types = ['score', 'delay', 'waitingtime', 'traveltime', 'speed']
    test_result = {result_type : [] for result_type in result_types}
    result_types = result_types[:-2]

    env.reset_test_cntr()
    env.testSeeds(True)
    observation = env.reset(mode=0, single=-1)
    obs_channelled = makeChannelled(observation, 0)
    for i in range(num_sim):
        terminate = False
        print(f"[Info] (evaluateAgent): Running evaluation simulation #{i+1}")

        result_per_sim = {result_type : [] for result_type in result_types}
        while not terminate:
            action = agent.choose_action(obs_channelled, test=True)
            next_observation, reward, terminate, truncate, info = env.step(action)
            next_obs_channelled = makeChannelled(next_observation, 0)

            result_per_sim['score'].append(reward['result'])
            result_per_sim['delay'].append(reward['total_delay'])
            result_per_sim['waitingtime'].append(reward['total_waiting'])

            observation = next_observation
            obs_channelled = next_obs_channelled

        try:
            observation, average_train_times, average_travel_time, average_speed = env.reset(mode=2, single=-1)
            obs_channelled = makeChannelled(observation, 0)

            test_result['traveltime'].append(average_travel_time)
            test_result['speed'].append(average_speed)
            for result_type in result_types :
                mean_result = np.mean( np.array(result_per_sim[result_type]) )
                test_result[result_type].append(mean_result)
            print(f"[Info] (evaluateAgent): Traveltime: {test_result['traveltime']}")
            print(f"[Info] (evaluateAgent): Speed: {test_result['speed']}")
            print(f"[Info] (evaluateAgent): Test result: {test_result}")
            saver(test_result, test_index, save_dir)
        except:
            raise Exception("[Error] (evaluateAgent): couldn't collect results.\nExiting...")
            observation = env.reset(mode=0, single=-1)
            obs_channelled = makeChannelled(observation, 0)

    env.testSeeds(False)

def getSingleCarResults(parameters, repeat: int = 100):
    env = SumoGymAdapter(parameters)
    total_speeds = []
    total_traveltimes = []

    actions = [OrderedDict({"0" : action_tmp}) for action_tmp in parameters["route_actions_single"]]
    for _ in range(repeat):
        speeds = []
        traveltimes = []

        for route_idx in [0, 1, 2, 3]:
            action = actions[route_idx]

            observation = env.reset(mode=0, single=route_idx)
            obs_channelled = makeChannelled(observation, 0)

            terminate = False

            while not terminate:
                next_observation, _, terminate, _, _ = env.step(action)

                next_obs_channelled = makeChannelled(next_observation, 0)

                observation = next_observation
                obs_channelled = next_obs_channelled

            _, average_travel_time, average_speed = env.reset(mode=1, single=-1)
            traveltimes.append(average_travel_time)
            speeds.append(average_speed)

        total_traveltimes.append(traveltimes)
        total_speeds.append(speeds)

    return total_traveltimes, total_speeds

if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Starting test_traffic_new")

    CURRENT_DIR = os.getcwd()
    print(f"\n[Info] (main): Current dir: {CURRENT_DIR}")
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "../Output/Results")
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\n[Info] (main): Save dir: {SAVE_DIR}")

    with open("configs/new_config.yaml", 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)['parameters']
        except yaml.YAMLError as exc:
            print(exc)

    # traveltimes_single, speeds_single = getSingleCarResults(parameters)
    # SINGLE_DIR = os.path.join(os.path.dirname(SAVE_DIR), "Single")
    # os.makedirs(SINGLE_DIR, exist_ok=True)
    # np.savetxt(os.path.join(SINGLE_DIR, "traveltime.csv"), np.array(traveltimes_single))
    # np.savetxt(os.path.join(SINGLE_DIR, "speed.csv"), np.array(speeds_single))

    env = SumoGymAdapter(CURRENT_DIR, SAVE_DIR, parameters)

    #load_checkpoint = os.path.isfile('tmp/q_eval/deepqnet.ckpt')

    TOTAL_TIME_STEPS = int(1e5)
    INPUT_SHAPE = [1, 84, 84]
    DISCOUNT = 0.99
    LR = 25e-5
    EXPLORATION_RATE = 1e-1
    MEM_SIZE = int(3e4)
    NUM_BATCHES = 32
    FREEZE_INTERVAL = int(3e3)
    EVALUATE_STEPS = int(1e3)
    EVALUATE_SIMULATIONS = 2

    agent = AgentDQN(input_shape=INPUT_SHAPE, num_actions=2, num_agents = 1, lr=LR, gamma=DISCOUNT, epsilon=1.0, epsilon_min=EXPLORATION_RATE,
                     mem_size=MEM_SIZE, batch_size=NUM_BATCHES)

    #if load_checkpoint:
     #   agent.load_models()

    print("[Info] (main): Loading up the agent's memory with random gameplay")
    observation = env.reset(mode=0, single=-1)
    obs_channelled = makeChannelled(observation, 0)
    obs_shape = (1, *observation.shape)
    episode_index = 1
    print(f"[Info] (main): Running episode #{episode_index}")
    print(f"[Info] (main): Replay buffer size: {agent.memorySize()}")
    for _ in range(MEM_SIZE):
        action = env.action_space.sample()
        next_observation, reward, terminate, truncate, info = env.step(action)

        next_obs_channelled = makeChannelled(next_observation, 0)

        agent.remember(obs_channelled, action, reward, next_obs_channelled, int(terminate))

        observation = next_observation
        obs_channelled = next_obs_channelled

        if terminate:
            observation = env.reset(mode=0, single=-1)
            obs_channelled = makeChannelled(observation, 0)
            episode_index += 1
            print(f"[Info] (main): Running episode #{episode_index}")
            print(f"[Info] (main): Replay buffer size: {agent.memorySize()}")

    print("[Info] (main): Done with random gameplay. Game on.")
    print(f"[Info] (main): Finished #{episode_index} episodes during replay buffer setup")
    print(f"[Info] (main): Replay buffer size: {agent.memorySize()}")
    print(f"[Info] (main): Learning state: {agent._start_learn}")
    print("\n\n")

    train_time_steps_score = []
    train_time_steps_delay = []
    train_time_steps_waitingtime = []
    train_episode_score = []
    train_travel_time = []
    train_speed = []

    episode_index = 0
    evaluate_index = 0
    time_step = 0
    while time_step != TOTAL_TIME_STEPS:
        print(f"[Info] (main): Running episode #{episode_index}")
        episode_index += 1

        observation = env.reset(mode=0, single=-1)
        obs_channelled = makeChannelled(observation, 0)

        score = 0
        terminate = False
        while not terminate:
            time_step += 1
            # print(f"Running iteration #{time_step}")
            action = agent.choose_action(obs_channelled, False)
            next_observation, reward, terminate, truncate, info = env.step(action)
            next_obs_channelled = makeChannelled(next_observation, 0)
            agent.remember(obs_channelled, action, reward, next_obs_channelled, int(terminate))
            train_time_steps_score.append(reward['result'])
            train_time_steps_delay.append(reward['total_delay'])
            train_time_steps_waitingtime.append(reward['total_waiting'])
            score +=reward['result']
            observation = next_observation
            obs_channelled = next_obs_channelled
            agent.learn()

            if time_step % FREEZE_INTERVAL == 0:
                agent.syncModels()

            if time_step % EVALUATE_STEPS == 0:
                average_train_times, average_travel_time, average_speed = env.reset(mode=1, single=-1)
                train_travel_time.append(average_travel_time)
                train_speed.append(average_speed)
                print(f"train_travel_time: {train_travel_time}")
                print(f"train_speed: {train_speed}")
                terminate = False

                evaluate_index = time_step//EVALUATE_STEPS
                print(f"[Info] (main): Starting evaluation at evaluate_index: {evaluate_index}, time_step: {time_step}, EVALUATE_STEPS: {EVALUATE_STEPS}, EVALUATE_SIMULATIONS: {EVALUATE_SIMULATIONS}")
                evaluateAgent(agent, env, EVALUATE_SIMULATIONS, evaluate_index, SAVE_DIR)

                break

            if time_step == TOTAL_TIME_STEPS:
                break

        if terminate:
            average_train_times, average_travel_time, average_train_speed = env.reset(mode=1, single=-1)
            train_travel_time.append(average_travel_time)
            train_speed.append(average_train_speed)
            print(f"train_travel_time: {train_travel_time}")
            print(f"train_speed: {train_speed}")


    print(f"[Info] (main): Total number of train episodes: {episode_index}")
    print(f"[Info] (main): Total number of evaluations: {evaluate_index}")

    print(f"[Info] (main): Save data...")
    saverSingle(data=train_time_steps_score, name='train_time_steps_score_reward', save_dir=SAVE_DIR)
    saverSingle(data=train_time_steps_delay, name='train_time_steps_score_delay', save_dir=SAVE_DIR)
    saverSingle(data=train_time_steps_waitingtime, name='train_time_steps_score_waitingtime', save_dir=SAVE_DIR)
    saverSingle(data=train_episode_score, name='train_episode_score', save_dir=SAVE_DIR)
    saverSingle(data=train_travel_time, name='train_travel_time', save_dir=SAVE_DIR)
    saverSingle(data=train_speed, name='train_speed', save_dir=SAVE_DIR)
    print(f"[Info] (main): Closing env...")
    env.close()
