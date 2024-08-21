import aienvs

import matplotlib.pyplot as plt


from multi_DQRN import AgentDQN
import yaml
import logging
import pdb
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import numpy as np
import os
import csv
from torch import save as torchSave
from matplotlib import pyplot as plt
from utils import *


SAMPLE_FREQUENCY = 20 # every 5 timesteps it samples

timestep_list = []
reward_list = []
speed_list = []
num_v_list = []

if __name__ == '__main__':

    CURRENT_DIR = os.getcwd()
    if (CURRENT_DIR != "src"):
        print("Again! Run cd src")
    CONFIG_DIR = os.path.join(CURRENT_DIR, "configs")
    CONFIG_FILE = os.path.join(CONFIG_DIR, "vertical_config.yaml")
    parameters = None
    with open(CONFIG_FILE, 'r') as stream:
        parameters = yaml.safe_load(stream)['parameters']

    CONGESTION = str(parameters['car_pr'])
    
    CONGESTION = CONGESTION.replace(".", "p")

    LOG_DIR = os.path.join(CURRENT_DIR, "../Logs", parameters['scene'], "Train", CONGESTION)
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "run.log")

    log_fmt = "[%(levelname)s] (%(name)s): %(message)s"
    log_handlers = [logging.FileHandler(LOG_FILE), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=log_handlers)
    
    logger = logging.getLogger("main")

    logger.info(f"Congestion setting : {CONGESTION}")

    logger.info(getLineDash())
    logger.info(getLineDash())
    logger.info("Single Training Run")
    logger.info(getLineDash())
    logger.info(getLineDash())

    printNewLine(2)

    OUTPUT_DIR = os.path.join(CURRENT_DIR, "../Output", parameters['scene'], "Train", CONGESTION)
    OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    RESULT_DIR = os.path.join(OUTPUT_DIR, "Results")
    os.makedirs(RESULT_DIR, exist_ok=True)
    MODEL_DIR = os.path.join(OUTPUT_DIR, "Model")
    os.makedirs(MODEL_DIR, exist_ok=True)
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "Checkpoint")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    logger.info(getLineDash())
    logger.info(f"Current dir: {CURRENT_DIR}")
    logger.info(f"Config dir: {CONFIG_DIR}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Model dir: {MODEL_DIR}")
    logger.info(f"Checkpoint dir: {CHECKPOINT_DIR}")

    # CHECKPOINT_NAME = "chkpt-{time_step:02d}-{speed:.2f}-{reward:.2f}.tar"
    CHECKPOINT_NAME = "chkpt-{time_step:02d}.pt.tar"
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME )

    MODEL_NAME = 'model.pt'
    MODEL_FILE = os.path.join(MODEL_DIR, MODEL_NAME)

    logger.info(f"Checkpoint file pattern: {CHECKPOINT_FILE}")
    logger.info(f"Model file: {MODEL_FILE}")
    logger.info(getLineDash())

    printNewLine(2)

    TOTAL_TIME_STEPS = int(1e6) # how long it trian for
    STACK_FRAMES = 1
    INPUT_SHAPE = [STACK_FRAMES, 84, 84]
    DISCOUNT = 0.99
    LR = 25e-5
    EXPLORATION_RATE = 1e-1
    MEM_SIZE = int(1000) # changed from 1000 to 100
    NUM_BATCHES = 32
    FREEZE_INTERVAL = int(3e3)
    EVALUATE_STEPS = int(1e4) # how often save checkpoint

    env = SumoGymAdapter(CURRENT_DIR, OUTPUT_DIR, parameters)
    env_stepLength = env.getStepLength()
    logger.info(f"environment's step length is : {env_stepLength}")

    num_agents_perFactor = len(parameters["lightPositions"].keys())
    num_actions = env.getActionSize()

    agent = AgentDQN(input_shape=INPUT_SHAPE, num_actions=num_actions, num_agents=num_agents_perFactor, lr=LR, gamma=DISCOUNT, epsilon=1.0, epsilon_min=EXPLORATION_RATE,
                     mem_size=MEM_SIZE, batch_size=NUM_BATCHES)

    time_step = 0
    torchSave({
            'time_step': time_step,
            'model_state_dict': agent.getModel().state_dict(),
            'optimizer_state_dict': agent.getOptimizer().state_dict(),
            'speed': -1,
            'reward': -100,
            }, CHECKPOINT_FILE.format(time_step=time_step))
    torchSave(agent.getModel(), MODEL_FILE)

    logger.info(getLineDash())
    logger.info("Filling up the agent's replay memory with random gameplay")
    logger.info(getLineDash())
    printNewLine(2)

    observation, _ = env.reset(single=-1)
    obs_channelled = makeChannelled(observation, 0, 0)

    stacked_cntr = 1
    obs_stacked = np.zeros(INPUT_SHAPE)
    obs_stacked = stackFrames(STACK_FRAMES, obs_channelled, obs_stacked)

    episode_index = 1
    logger.info(getLineDash())
    logger.info(f"Running memory episode #{episode_index}")
    logger.info(f"Replay memory buffer size: {agent.memorySize()}")
    logger.info(getLineDash())
    printNewLine(2)

    for _ in range(MEM_SIZE):
        action = agent.choose_action(obs_stacked, greedy_only=False)
        next_observation, reward, terminate, truncate, reward_info = env.step(action)

        next_obs_channelled = makeChannelled(next_observation, 0, 0)
        next_obs_stacked = stackFrames(STACK_FRAMES, next_obs_channelled, obs_stacked)

        if stacked_cntr == STACK_FRAMES:
            agent.remember(obs_stacked, action, reward, next_obs_stacked, int(terminate))
        else:
            stacked_cntr += 1

        obs_stacked = next_obs_stacked

        if terminate:
            logger.info(getLineDash())
            print(f"[Info] (main): Running memory episode #{episode_index}")
            print(f"[Info] (main): Replay buffer size: {agent.memorySize()}")
            logger.info(getLineDash())
            printNewLine(2)

            episode_index += 1

            observation, run_stats = env.reset(single=-1)
            obs_channelled = makeChannelled(observation, 0, 0)

            stacked_cntr = 1
            obs_stacked = np.zeros(INPUT_SHAPE)
            obs_stacked = stackFrames(STACK_FRAMES, obs_channelled, obs_stacked)

            total_speed, total_max_speed, total_travel_time, total_wait_time, total_time_loss = run_stats
            avg_speed = sum(total_speed)/len(total_speed)
            avg_max_speed = sum(total_max_speed)/len(total_max_speed)
            avg_travel_time = sum(total_travel_time)/len(total_travel_time)
            avg_wait_time = sum(total_wait_time)/len(total_wait_time)
            avg_time_loss = sum(total_time_loss)/len(total_time_loss)

            logger.info(getLineDash())
            logger.info(f"avg_speed: {avg_speed}")
            logger.info(f"avg_max_speed: {avg_max_speed}")
            logger.info(f"avg_travel_time: {avg_travel_time}")
            logger.info(f"avg_wait_time: {avg_wait_time}")
            logger.info(f"avg_time_loss: {avg_time_loss}")
            logger.info(getLineDash())
            printNewLine(2)

    logger.info(getLineDash())
    logger.info("Replay memory buffer has been filled.")
    logger.info(f"Replay memory buffer size: {agent.memorySize()}")
    logger.info(f"Finished #{episode_index} episodes during replay buffer setup")
    logger.info(getLineDash())
    printNewLine(2)


    avg_score = []
    avg_vehicles = []
    avg_timesteps = []
    avg_speed = []
    avg_max_speed = []
    avg_travel_time = []
    avg_wait_time = []
    avg_time_loss = []

    logger.info(getLineDash())
    logger.info("Starting training")
    logger.info(getLineDash())
    printNewLine(2)

    episode_index = 0
    eval_index = 0
    observation, _ = env.reset(single=-1)
    obs_channelled = makeChannelled(observation, 0, 0)

    stacked_cntr = 1
    obs_stacked = np.zeros(INPUT_SHAPE)
    obs_stacked = stackFrames(STACK_FRAMES, obs_channelled, obs_stacked)

    while time_step != TOTAL_TIME_STEPS:
        episode_index += 1
        logger.info(getLineDash())
        logger.info(f"Running episode #{episode_index}\nTime step: {time_step}")
        logger.info(getLineDash())
        printNewLine(2)

        simulation_score = []
        simulation_speed = []
        simulation_accel = []
        simulation_delay = []
        simulation_waitingtime = []
        simulation_vehicles = []

        time_step_simulation = 0
        terminate = False
        while not terminate:
            time_step_simulation += 1
            time_step += 1
            logger.debug(f"Running iteration #{time_step}")

            action = agent.choose_action(obs_stacked, greedy_only=False)

            next_observation, reward, terminate, truncate, reward_info = env.step(action)
            next_obs_channelled = makeChannelled(next_observation, 0, 0)
            next_obs_stacked = stackFrames(STACK_FRAMES, next_obs_channelled, obs_stacked)

            if stacked_cntr == STACK_FRAMES:
                agent.remember(obs_stacked, action, reward, next_obs_stacked, int(terminate))
            else:
                stacked_cntr += 1

            obs_stacked = next_obs_stacked

            num_vehicles = reward_info['vehicles']
            if num_vehicles != 0:
                simulation_score.append(reward)



                print(f"timestep: {time_step}, reward: {reward}, waiting time {reward_info['waiting']/num_vehicles}") #added =------------------------------------------------
                if time_step % SAMPLE_FREQUENCY == 0:
                    reward_list.append(reward)
                    timestep_list.append(time_step)


                simulation_speed.append(reward_info['speed']/num_vehicles)
                simulation_accel.append(reward_info['accel']/num_vehicles)
                simulation_delay.append(reward_info['delay']/num_vehicles)
                simulation_waitingtime.append(reward_info['waiting']/num_vehicles)
                simulation_vehicles.append(num_vehicles)

            if time_step % FREEZE_INTERVAL == 0:
                agent.syncModels()

            agent.learn()

            if time_step % EVALUATE_STEPS == 0:
                eval_index += 1
                sim_avg_speed = -1
                if reward_info['vehicles'] != 0:
                    sim_avg_speed = reward_info['speed']/num_vehicles
                torchSave({
                        'time_step': time_step,
                        'model_state_dict': agent.getModel().state_dict(),
                        'optimizer_state_dict': agent.getOptimizer().state_dict(),
                        'speed': sim_avg_speed,
                        'reward': reward,
                        }, CHECKPOINT_FILE.format(time_step=time_step))

            if time_step == TOTAL_TIME_STEPS:
                break

        if terminate:
            observation, run_stats = env.reset(single=-1)
            obs_channelled = makeChannelled(observation, 0, 0)

            stacked_cntr = 1
            obs_stacked = np.zeros(INPUT_SHAPE)
            obs_stacked = stackFrames(STACK_FRAMES, obs_channelled, obs_stacked)

            total_speed, total_max_speed, total_travel_time, total_wait_time, total_time_loss = run_stats
            avg_score.append(sum(simulation_score)/len(simulation_score))
            avg_timesteps.append(time_step_simulation)
            avg_vehicles.append(len(total_speed))
            avg_speed.append(sum(total_speed)/len(total_speed))
            avg_max_speed.append(sum(total_max_speed)/len(total_max_speed))
            avg_travel_time.append(sum(total_travel_time)/len(total_travel_time))
            avg_wait_time.append(sum(total_wait_time)/len(total_wait_time))
            avg_time_loss.append(sum(total_time_loss)/len(total_time_loss))

            logger.info(getLineDash())
            logger.info(f"avg_score: {avg_score}")


            logger.info(f"avg_vehicles: {avg_vehicles}")
            logger.info(f"avg_timesteps: {avg_timesteps}")
            logger.info(f"avg_speed: {avg_speed}")
            logger.info(f"avg_max_speed: {avg_max_speed}")
            logger.info(f"avg_travel_time: {avg_travel_time}")
            logger.info(f"avg_wait_time: {avg_wait_time}")
            logger.info(f"avg_time_loss: {avg_time_loss}")
            logger.info(getLineDash())
            printNewLine(2)

            # saveData(data=simulation_score, filename='train_score', save_dir=RESULT_DIR)
            # saveData(data=simulation_accel, filename='train_accel', save_dir=RESULT_DIR)
            # saveData(data=simulation_speed, filename='train_speed', save_dir=RESULT_DIR)
            # saveData(data=simulation_delay, filename='train_delay', save_dir=RESULT_DIR)
            # saveData(data=simulation_waitingtime, filename='train_waitingtime', save_dir=RESULT_DIR)

    logger.info(getLineDash())
    logger.info(f"Total number of train episodes: {episode_index}")
    logger.info(f"Total number of evaluations: {eval_index}")
    logger.info(getLineDash())
    printNewLine(2)

    logger.info(getLineDash())
    logger.info("Saving data...")
    logger.info(getLineDash())
    saveData(data=avg_score, filename='avg_score', save_dir=RESULT_DIR)
    saveData(data=avg_vehicles, filename='avg_vehicles', save_dir=RESULT_DIR)
    saveData(data=avg_timesteps, filename='avg_timesteps', save_dir=RESULT_DIR)
    saveData(data=avg_speed, filename='avg_speed', save_dir=RESULT_DIR)
    saveData(data=avg_max_speed, filename='avg_max_speed', save_dir=RESULT_DIR)
    saveData(data=avg_travel_time, filename='avg_travel_time', save_dir=RESULT_DIR)
    saveData(data=avg_wait_time, filename='avg_wait_time', save_dir=RESULT_DIR)
    saveData(data=avg_time_loss, filename='avg_time_loss', save_dir=RESULT_DIR)

    printNewLine(2)
    logger.info(getLineDash())
    logger.info("Closing env...")
    logger.info(getLineDash())
    env.close()

    print(timestep_list)
    print(reward_list)

    plt.plot(timestep_list, reward_list)
    plt.ylabel('reward')
    plt.xlabel('timestep')
    plt.show()



