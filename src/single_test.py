import aienvs
from multi_DQRN import AgentDQN
import yaml
import logging
import pdb
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import numpy as np
import os
import csv
from torch import load as torchLoad
from matplotlib import pyplot as plt
from utils import *

plt.style.use('ggplot')


timestep_list = []
reward_list = []
speed_list = []
num_v_list = []

SAMPLE_FREQUENCY = 1 # every x timesteps it samples data 


def evaluateAgent(agent, env, num_sim, eval_index, save_dir):
    avg_score = []
    avg_vehicles = []
    avg_timesteps = []
    avg_speed = []
    avg_max_speed = []
    avg_travel_time = []
    avg_wait_time = []
    avg_time_loss = []

    logger = logging.getLogger("evaluateAgent")

    env.reset_test_cntr()
    observation, _ = env.reset(single=-1)
    obs_channelled = makeChannelled(observation, 0, 0)

    obs_stacked = np.zeros(INPUT_SHAPE)
    obs_stacked = stackFrames(STACK_FRAMES, obs_channelled, obs_stacked)

    for i in range(num_sim):
        logger.info(getLineDash())
        logger.info(f"Running evaluation: {eval_index}, simulation {i+1}/{num_sim}")
        logger.info(getLineDash())
        printNewLine(2)

        simulation_score = []
        simulation_speed = []
        simulation_accel = []
        simulation_delay = []
        simulation_waitingtime = []
        simulation_vehicles = []

        time_step = 0
        terminate = False
        while not terminate:
            time_step += 1
            action = agent.choose_action(obs_stacked, greedy_only=True)

            next_observation, reward, terminate, truncate, reward_info = env.step(action)

            next_obs_channelled = makeChannelled(next_observation, 0, 0)
            next_obs_stacked = stackFrames(STACK_FRAMES, next_obs_channelled, obs_stacked)

            obs_stacked = next_obs_stacked

            num_vehicles = reward_info['vehicles']
            if num_vehicles != 0:
                simulation_score.append(reward)
                simulation_speed.append(reward_info['speed']/num_vehicles)
                simulation_accel.append(reward_info['accel']/num_vehicles)
                simulation_delay.append(reward_info['delay']/num_vehicles)
                simulation_waitingtime.append(reward_info['waiting']/num_vehicles)
                simulation_vehicles.append(num_vehicles)

            if time_step % SAMPLE_FREQUENCY == 0: # added ------------------------------------------------------------------------
                timestep_list.append(time_step)
                reward_list.append(reward)
                num_v_list.append(num_vehicles)

                if num_vehicles == 0:
                    speed_list.append(reward_info['speed'])
                else:
                    speed_list.append(reward_info['speed']/num_vehicles)



        if terminate:
            observation, run_stats = env.reset(single=-1)
            obs_channelled = makeChannelled(observation, 0, 0)

            obs_stacked = np.zeros(INPUT_SHAPE)
            obs_stacked = stackFrames(STACK_FRAMES, obs_channelled, obs_stacked)

            total_speed, total_max_speed, total_travel_time, total_wait_time, total_time_loss = run_stats
            avg_vehicles.append(len(total_speed))
            avg_timesteps.append(time_step)
            avg_score.append(sum(simulation_score)/len(simulation_score))
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

    eval_result = {"avg_score" : avg_score, 'avg_vehicles' : avg_vehicles, "avg_timesteps" : avg_timesteps, "avg_speed" : avg_speed, "avg_max_speed" : avg_max_speed, "avg_travel_time" : avg_travel_time, "avg_wait_time" : avg_wait_time, "avg_time_loss" : avg_time_loss}
    logger.info(getLineDash())
    logger.info(f"Saving data for evaluation {eval_index}...")
    logger.info(getLineDash())
    saveDataDict(eval_result, eval_index, save_dir)

if __name__ == '__main__':

    CURRENT_DIR = os.getcwd()

    CONFIG_DIR = os.path.join(CURRENT_DIR, "configs")
    CONFIG_FILE = os.path.join(CONFIG_DIR, "vertical_config.yaml")
    parameters = None
    with open(CONFIG_FILE, 'r') as stream:
        parameters = yaml.safe_load(stream)['parameters']

    CONGESTION = str(parameters['car_pr'])
    CONGESTION = CONGESTION.replace(".", "p")

    LOG_DIR = os.path.join(CURRENT_DIR, "../Logs", parameters['scene'], "Test", CONGESTION)
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "run.log")

    log_fmt = "[%(levelname)s] (%(name)s): %(message)s"
    log_handlers = [logging.FileHandler(LOG_FILE), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=log_handlers)

    logger = logging.getLogger("main")
    logger.info(f"Congestion setting : {CONGESTION}")

    logger.info(getLineDash())
    logger.info(getLineDash())
    logger.info("Single Testing Run")
    logger.info(getLineDash())
    logger.info(getLineDash())

    printNewLine(2)

    OUTPUT_DIR = os.path.join(CURRENT_DIR, "../Output", parameters['scene'], "Test", CONGESTION)
    OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    RESULT_DIR = os.path.join(OUTPUT_DIR, "Results")
    os.makedirs(RESULT_DIR, exist_ok=True)
    MODEL_DIR = os.path.join(OUTPUT_DIR.replace("Test", "Train"), "Model")
    os.makedirs(MODEL_DIR, exist_ok=True)
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR.replace("Test", "Train"), "Checkpoint")
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

    TOTAL_TIME_STEPS = int(3e3)
    STACK_FRAMES = 1
    INPUT_SHAPE = [STACK_FRAMES, 84, 84]
    DISCOUNT = 0.99
    LR = 25e-5
    EXPLORATION_RATE = 1e-1
    MEM_SIZE = int(1)
    NUM_BATCHES = 32
    EVALUATE_STEPS = int(1e4)
    EVALUATE_SIMULATIONS = 1 # changed from 3 to 1

    env = SumoGymAdapter(CURRENT_DIR, OUTPUT_DIR, parameters)
    env_stepLength = env.getStepLength()
    logger.info(f"environment's step length is : {env_stepLength}")
    env.testSeeds(True)

    num_agents_perFactor = len(parameters["lightPositions"].keys())
    num_actions = env.getActionSize()

    agent = AgentDQN(input_shape=INPUT_SHAPE, num_actions=num_actions, num_agents=num_agents_perFactor, lr=LR, gamma=DISCOUNT, epsilon=1.0, epsilon_min=EXPLORATION_RATE,
                     mem_size=MEM_SIZE, batch_size=NUM_BATCHES)
    agent.getModel().eval()

    eval_index = 0
    for time_step in range(TOTAL_TIME_STEPS, -1, -EVALUATE_STEPS):
        eval_index = int(time_step/EVALUATE_STEPS)
        chkpt_file = CHECKPOINT_FILE.format(time_step=time_step)
        if not os.path.exists(chkpt_file):
            logger.info(f"Checkpoint file: {chkpt_file} doesn't exist. Skipping")
            continue
        checkpoint = torchLoad(chkpt_file)

        logger.info(getLineDash())
        logger.info(f"Running evaluation #{eval_index}")
        logger.info(f"Checkpoint file used: {chkpt_file}")
        logger.info(getLineDash())
        printNewLine(2)

        agent.loadModel(checkpoint['model_state_dict'])

        evaluateAgent(agent, env, EVALUATE_SIMULATIONS, eval_index, RESULT_DIR)

    logger.info(getLineDash())
    logger.info(f"Total number of evaluations: {eval_index}")
    logger.info(getLineDash())
    printNewLine(2)

    logger.info(getLineDash())
    logger.info("Closing env...")
    logger.info(getLineDash())
    env.close()

    plt.plot(timestep_list, speed_list)
    plt.ylabel('speed')
    plt.xlabel('timestep')
    plt.show()

    plt.plot(timestep_list, reward_list)
    plt.ylabel('reward')
    plt.xlabel('timestep')
    plt.show()


    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('timestep')
    ax1.set_ylabel('avg speed', color=color)
    ax1.plot(timestep_list, speed_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('reward', color=color)  # we already handled the x-label with ax1
    ax2.plot(timestep_list, reward_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


