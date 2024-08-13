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

def saver(data, name, save_dir):
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

if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Starting test_traffic_new")

    CURRENT_DIR = os.getcwd()
    print(f"\nCurrent dir: {CURRENT_DIR}")
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "../Output/Results")
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\nSave dir: {SAVE_DIR}")

    with open("configs/new_config.yaml", 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)['parameters']
        except yaml.YAMLError as exc:
            print(exc)

    env = SumoGymAdapter(parameters)

    #load_checkpoint = os.path.isfile('tmp/q_eval/deepqnet.ckpt')

    TOTAL_TIME_STEPS = int(1e6)
    INPUT_SHAPE = [1, 84, 84]
    DISCOUNT = 0.99
    LR = 25e-5
    EXPLORATION = 1e-1
    MEM_SIZE = int(3e4)
    NUM_BATCHES = 32
    FREEZE_INTERVAL = int(3e4)

    agent = AgentDQN(input_shape=INPUT_SHAPE, num_actions=2, lr=LR, gamma=DISCOUNT, epsilon=1.0, epsilon_min=EXPLORATION,
                     mem_size=MEM_SIZE, batch_size=NUM_BATCHES, freeze_interval=FREEZE_INTERVAL)

    #if load_checkpoint:
     #   agent.load_models()

    print("Loading up the agent's memory with random gameplay")
    observation = env.reset(full=False)
    obs_channelled = makeChannelled(observation, 0)
    obs_shape = (1, *observation.shape)
    while agent.memorySize() != MEM_SIZE:
        action = env.action_space.sample()
        next_observation, reward, terminate, truncate, info = env.step(action)

        next_obs_channelled = makeChannelled(next_observation, 0)

        agent.remember(obs_channelled, action, reward, next_obs_channelled, int(terminate))

        observation = next_observation
        obs_channelled = next_obs_channelled

        if terminate:
            observation = env.reset(full=False)
        # print('MEMORY_COUNTER: ', agent.memorySize())

    print("Done with random game play. Game on.")

    train_time_steps_score = []
    train_time_steps_delay = []
    train_time_steps_waitingtime = []
    train_episode_score = []
    train_travel_time = []

    episode_index = 0
    time_step = 0
    while True:
        print(f"Running episode #{episode_index}")
        episode_index += 1
        terminate = False
        if episode_index != 1:
            try:
                observation, average_train_times, average_train_time, _ = env.reset(full=True)
                train_travel_time.append(average_train_time)
                print(f"train_travel_time: {train_travel_time}")
            except:
                observation = env.reset(full=False)
        else:
            observation = env.reset(full=False)

        obs_channelled = makeChannelled(observation, 0)
        score = 0
        while (not terminate) and time_step < TOTAL_TIME_STEPS:
            time_step += 1
            # print(f"Running iteration #{time_step}")
            if time_step % FREEZE_INTERVAL == 0:
                agent.syncModels()
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
            if time_step == TOTAL_TIME_STEPS:
                print(f"Gathering results for episode: {episode_index}")
                try:
                    observation, average_train_times, average_train_time, _ = env.reset(full=True)
                    train_travel_time.append(average_train_time)
                    train_episode_score.append(score)
                except:
                    print(f"[Error] (main): couldn't get results from env.reset({i}). Episode: {episode_index}")
                    observation = env.reset(full=False)
                break

    print(f"[Info] (main): Save data...")
    saver(data=train_time_steps_score, name='train_time_steps_score_reward', save_dir=SAVE_DIR)
    saver(data=train_time_steps_delay, name='train_time_steps_score_delay', save_dir=SAVE_DIR)
    saver(data=train_time_steps_waitingtime, name='train_time_steps_score_waitingtime', save_dir=SAVE_DIR)
    saver(data=train_episode_score, name='train_episode_score', save_dir=SAVE_DIR)
    saver(data=train_travel_time, name='train_travel_time', save_dir=SAVE_DIR)
    print(f"[Info] (main): Closing env...")
    env.close()
