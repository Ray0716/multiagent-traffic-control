import glob
import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from LoggedTestCase import LoggedTestCase
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import pdb
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from factor_graph import factor_graph
import csv
import time
from maxplus import maxplus
from torch import load as torchLoad
from utils import *

class experimentor():

    def __init__(self, total_simulation=8):
        logging.info("Starting test_traffic_new")
        #with open("configs/testconfig.yaml", 'r') as stream:
        # with open("configs/eight_config.yaml", 'r') as stream:
        with open("configs/four_config_0p4.yaml", 'r') as stream:
            try:
                parameters=yaml.safe_load(stream)['parameters']
            except yaml.YAMLError as exc:
                print(exc)

        CURRENT_DIR = os.getcwd()
        OUTPUT_DIR = os.path.join(CURRENT_DIR, "../Output", parameters['scene'], "MASTest")
        OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        RESULT_DIR = os.path.join(OUTPUT_DIR, "Results")
        os.makedirs(RESULT_DIR, exist_ok=True)
        MODEL_DIR = os.path.join(OUTPUT_DIR.replace("MASTest", "Train").replace(parameters['scene'], parameters['factor_scene']), "Model")
        os.makedirs(MODEL_DIR, exist_ok=True)
        CHECKPOINT_DIR = os.path.join(OUTPUT_DIR.replace("MASTest", "Train").replace(parameters['scene'], parameters['factor_scene']), "Checkpoint")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"\n[Info] (main): Current dir: {CURRENT_DIR}")
        print(f"\n[Info] (main): Output dir: {OUTPUT_DIR}")
        print(f"\n[Info] (main): Result dir: {RESULT_DIR}")
        print(f"\n[Info] (main): Model dir: {MODEL_DIR}")
        print(f"\n[Info] (main): Checkpoint dir: {CHECKPOINT_DIR}")

        self.result_dir = RESULT_DIR + "_0p4"

        # CHECKPOINT_NAME = "chkpt-{time_step:02d}-{speed:.2f}-{reward:.2f}.tar"
        CHECKPOINT_NAME = "chkpt-{time_step:02d}.pt.tar"
        CHECKPOINT_FILE = os.path.join( CHECKPOINT_DIR, CHECKPOINT_NAME )
        self.chkpt_file = CHECKPOINT_FILE

        MODEL_NAME = 'model.pt'
        MODEL_FILE = os.path.join(MODEL_DIR, MODEL_NAME)

        STACK_FRAMES = 1
        INPUT_SHAPE = [STACK_FRAMES, 84, 84]
        DISCOUNT = 0.99
        LR = 25e-5
        EXPLORATION_RATE = 1e-1
        MEM_SIZE = int(1)
        NUM_BATCHES = 32

        self.i = 0
        self.env = SumoGymAdapter(CURRENT_DIR, OUTPUT_DIR, parameters)
        self.env.testSeeds(True)
        self.total_simulation = total_simulation
        self._parameters = parameters
        for keys in self._parameters['testmodelnr'].keys():
            self.index = keys
        self.modelnumber = self._parameters['testmodelnr'][self.index]
        self.num_agents = len(self._parameters['lightPositions'].keys())

        #************************  DO NOT FORGET TO CHANGE THE PADDER AND CONFIG FILE   ***********************
        self.factor_graph = factor_graph(parameters = self._parameters, input_shape=INPUT_SHAPE, lr=LR, discount=DISCOUNT, exploration_rate=EXPLORATION_RATE, mem_size=MEM_SIZE, num_batches=NUM_BATCHES)
        if self._parameters['coordination_algo'] == 'maxplus':
            self.maxplus = maxplus(regular_factor=self._parameters['factored_agents'], agent_neighbour_combo=self._parameters['agent_neighbour_combo'], max_iter=self._parameters['max_iter'])
            print('USING MAXPLUS ALGORITHM FOR COORDINATION')
        self.result_initialiser()
        self.algo_timer = []
        # self.fileinitialiser()
        self.factored_agent_type = self._parameters['factored_agent_type']
        #self.result_appender('results/six_intersection/0.4/maxplus/trial/150000')


    def tester(self):
        path = os.getcwd()

        eval_index = 48
        for self.model in range(520000, -1, -10000):
            eval_index += 1
            self.env.reset_test_cntr()
            self.env.testSeeds(True)

            if self.index == 'individual':
                filename = 'single' + '_' + str(self._parameters['car_pr']) + '_deepqnet.ckpt-' + str(self.model)
                try:
                    chkpt = os.path.join(*[path, 'tmp', 'one_intersection', str(self._parameters['car_pr']), 'q_eval', filename])
                    self.factor_graph.Q_function_dict['individual'].load_models(chkpt)
                except:
                    chkpt = os.path.join(*[path, 'tmp', 'one_intersection', str(self._parameters['car_pr']), 'trial1', 'q_eval', filename])
                    self.factor_graph.Q_function_dict['individual'].load_models(chkpt)
                print('LOADED CHECKPOINT:', filename)

            elif self.index == 'vertical':
                chkpt_file = self.chkpt_file.format(time_step=self.model)
                checkpoint = torchLoad(chkpt_file)
                self.factor_graph.Q_function_dict['vertical'].loadModel(checkpoint['model_state_dict'])
                print('LOADED CHECKPOINT: ', chkpt_file)

            self.test(eval_index)

    def result_initialiser(self):
        self.test_result = {}
        self.test_result['result'] = []
        self.test_result['num_teleports'] = []
        self.test_result['emergency_stops'] = []
        self.test_result['total_delay'] = []
        self.test_result['total_waiting'] = []
        self.test_result['traveltime'] = []

    def file_finder(self, data):
        path = os.getcwd()
        self.result_path = os.path.join(path, data)
        reward_files = [files for files in glob.glob(os.path.join(path, data, 'result*'))]
        data_dict = {}
        self.modelnumber=None
        if len(reward_files)!=0:
            sorted_reward_files = []
            for i in range(len(reward_files)):
                csv = reward_files[i].split('result')[-1]
                sorted_reward_files.append(int(csv.split('.')[0]))
            sorted_reward_files.sort()
            data_dict['result'] =  'result' + str(sorted_reward_files[-1]) + '.csv'
            data_dict['traveltime']  = 'traveltime' + str(sorted_reward_files[-1]) + '.csv'
            #data_dict['algo_timer'] = 'algo_timer' + str(sorted_reward_files[-1]) + '.csv'
            self.modelnumber = sorted_reward_files[-1] + 10000
        print('MODEL NUMBER BEING USED IS: ', self.modelnumber)
        return data_dict, bool(data_dict)

    def result_appender(self, file_path):
        data_dict, bool_value = self.file_finder(file_path)
        if bool_value == True:
            for keys in data_dict.keys():
                path = os.path.join(self.result_path, data_dict[keys])
                data = pd.read_csv(path, header=None)
                for j in range(len(data)):
                    if keys =='algo_timer':
                        self.algo_timer.append(data.values[j][0])
                    else:
                        self.test_result[keys].append(data.values[j][0])
        else:
            print("NO PREVIOUS SAVED RESULT")

    def store_result(self, reward):
        for keys in reward.keys():
            self.test_result[keys].append(reward[keys])

    def shape(self, ob):
        for keys in ob[0].keys():
            print(keys, ob[0][keys].shape)

    def store_tt(self, tt):
        self.test_result['traveltime'].append(tt)

    def saver(self, data, name, iternumber):
        path = os.getcwd()
        filename = str(name) + str(iternumber) + '.csv'
        pathname = os.path.join(*[path, 'results', self._parameters['scene'], str(self._parameters['car_pr']), self._parameters['coordination_algo'], 'trial', 'zeroiter',filename])
        outfile = open(pathname, 'w')
        writer = csv.writer(outfile)
        writer.writerows(map(lambda x:[x], data))
        outfile.close()

    def fileinitialiser(self):
        path = os.getcwd()
        for key in self.test_result.keys():
            filename = key + '.csv'
            pathname = os.path.join(*[path, 'results', self._parameters['scene'], str(self._parameters['car_pr']), self._parameters['coordination_algo'], 'trial', 'zeroiter', filename])
            if os.path.exists(os.path.dirname(pathname)):
                print('Result directroy already exists: ', pathname)
            else:
                os.makedirs(os.path.dirname(pathname))

    def file_rename(self, name, iternr):
        path = os.getcwd()
        res_dir = os.path.join(path, 'test_result', str(self.result_folder))
        oldname = str(name) + str(iternr-10000)  + '.csv'
        newname = str(name) + str(iternr) + '.csv'
        os.rename(res_dir+ '/' + oldname, res_dir + '/' + newname)

    def reset(self):
        self.factor_graph.reset()

    def qarr_key_changer(self, q_arr):
        q_val = {}
        for keys in q_arr.keys():
            q_val[str(self._parameters['factored_agents'][keys])] = q_arr[keys]
        return q_val

    def take_action(self, state_graph):
        q_arr = self.factor_graph.get_factored_Q_val(state_graph)
        if self._parameters['coordination_algo'] == 'brute':
            start = time.process_time()
            sum_q_value, best_action, sumo_act = self.factor_graph.b_coord(q_arr)
            self.algo_timer.append(time.process_time() - start)
            print(sum_q_value, sumo_act)
        elif self._parameters['coordination_algo'] == 'maxplus':
            self.maxplus.initialise_again()
            q_arr = self.qarr_key_changer(q_arr)
            start = time.process_time()
            payoff, sumo_act = self.maxplus.max_plus_calculator(q_arr)
            self.algo_timer.append(time.process_time() - start)
        else:
            start = time.process_time()
            sumo_act = self.factor_graph.individual_coord(q_arr)
            self.algo_timer.append(time.process_time() - start)
        return sumo_act

    def save(self, data, iternr):
        for key in data.keys():
            result = data[key]
            self.saver(data=result, name=key, iternumber=iternr)

    def saverSingle(self, data, name, save_dir):
        name = str(name)
        filename = os.path.join(save_dir, name+'.csv')
        outfile = open(filename, 'w')
        writer = csv.writer(outfile)
        writer.writerows(map(lambda x:[x], data))

    def makeChannelled(self, item, index: int, transpose: bool, repeat: int = 0):
        item = np.expand_dims(item, index)
        if repeat:
            item = np.concatenate( (item,)*repeat, axis = index )
        item_return = item.copy()
        return item_return.transpose(0, 2, 1) if transpose else item_return

    def makeChannelledGraph(self, ob, index: int):
        ob_dict = {}
        for keys in self._parameters['factored_agents'].keys():
            transpose = self.factored_agent_type[keys] == "horizontal"
            ob_dict[keys] = self.makeChannelled(ob[keys], index=index, transpose=transpose)

        return ob_dict

    def stack_frames(self, stacked_frames, frame, buffer_size, config):
        if stacked_frames is None:
            stacked_frames = np.zeros((buffer_size, *frame.shape))
            for idx, _ in enumerate(stacked_frames):
                if config=='horizontal':
                    stacked_frames[idx, :] = frame.transpose()
                else:
                    stacked_frames[idx, :] = frame
        else:
            stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]
            if config== 'horizontal':
                stacked_frames[buffer_size-1, :] = frame.transpose()
            else:
                stacked_frames[buffer_size-1, :] = frame

        stacked_frame = stacked_frames
        stacked_state = stacked_frames.transpose(1,2,0)[None, ...]

        return stacked_frame, stacked_state

    def stack_state_initialiser(self):
        self.ob_dict = {}

    def stacked_graph(self, ob, initial=True):
        for keys in self._parameters['factored_agents'].keys():
            if initial==True:
                self.ob_dict[keys], self.stacked_state_dict[keys] = self.stack_frames(stacked_frames=None, frame= ob[keys], buffer_size=1, config = self.factored_agent_type[keys])

            else:
                self.ob_dict[keys], self.stacked_state_dict[keys] = self.stack_frames(stacked_frames=self.ob_dict[keys], frame= ob[keys], buffer_size=1, config = self.factored_agent_type[keys])

        return self.ob_dict, self.stacked_state_dict

    def six_padder(self, ob):
        for keys in ob.keys():
            if keys == '0':
                ob[keys] = np.pad(ob[keys], ((15,15),(1,0)), 'constant', constant_values=(0,0))
            elif keys == '1':
                ob[keys] = np.pad(ob[keys], ((15,15),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '2':
                ob[keys] = np.pad(ob[keys], ((15,14),(1,0)), 'constant', constant_values=(0,0))
            elif keys =='3':
                ob[keys] = np.pad(ob[keys], ((15,14),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '4':
                ob[keys] = ob[keys][:84,:]
                ob[keys] = np.pad(ob[keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
            elif keys == '5':
                ob[keys] = ob[keys][:84,:]
                ob[keys] = np.pad(ob[keys], ((0,0),(15,16)), 'constant', constant_values=(0,0))
            elif keys == '6':
                ob[keys] = ob[keys][:84, :]
                ob[keys] = np.pad(ob[keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
        return ob

    def three_padder(self, ob):
        for keys in ob.keys():
            if keys == '1':
                ob[keys] = np.pad(ob[keys], ((0,0),(1,0)), 'constant', constant_values= (0,0))
        return ob

    def four_padder(self, ob):
        for keys in ob.keys():
            if keys == '0':
                ob[keys] = np.pad(ob[keys], ((15,14),(0,0)), 'constant', constant_values=(0,0))
            elif keys == '1':
                ob[keys] = np.pad(ob[keys], ((14,14),(0,0)), 'constant', constant_values=(0,0))
            elif keys == '2':
                ob[keys] = np.pad(ob[keys], ((0,0),(14,14)), 'constant', constant_values=(0,0))
            elif keys =='3':
                ob[keys] = np.pad(ob[keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
        return ob

    def six_ind_padder(self, ob):
        for keys in ob.keys():
            if keys == '1':
                ob[keys] = np.pad(ob[keys], ((0,0),(2,2)), 'constant', constant_values=(0,0))
            elif keys == '2':
                ob[keys] = np.pad(ob[keys], ((0,0),(2,2)), 'constant', constant_values=(0,0))
            elif keys =='3':
                ob[keys] = np.pad(ob[keys], ((1,0),(0,0)), 'constant', constant_values=(0,0))
            elif keys == '4':
                ob[keys] = np.pad(ob[keys], ((1,0),(2,2)), 'constant', constant_values=(0,0))
            elif keys == '5':
                ob[keys] = np.pad(ob[keys], ((1,0),(2,2)), 'constant', constant_values=(0,0))
        return ob

    def eight_padder(self, ob):
        for keys in ob.keys():
            if keys == '0':
                ob[keys] = np.pad(ob[keys], ((15,14),(1,0)), 'constant', constant_values=(0,0))
            elif keys == '1':
                ob[keys] = np.pad(ob[keys], ((15,14),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '2':
                ob[keys] = np.pad(ob[keys], ((15,14),(1,0)), 'constant', constant_values=(0,0))
            elif keys =='3':
                ob[keys] = np.pad(ob[keys], ((15,15),(1,0)), 'constant', constant_values=(0,0))
            elif keys == '4':
                ob[keys] = np.pad(ob[keys], ((15,15),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '5':
                ob[keys] = np.pad(ob[keys], ((15,15),(1,1)), 'constant', constant_values=(0,0))
            elif keys =='6':
                ob[keys] = np.pad(ob[keys], ((0,0),(15,15)), 'constant', constant_values=(0,0))
            elif keys =='7':
                ob[keys] = np.pad(ob[keys], ((0,0),(15,15)), 'constant', constant_values=(0,0))
            elif keys =='8':
                ob[keys] = np.pad(ob[keys], ((0,0),(15,15)), 'constant', constant_values=(0,0))
            elif keys =='9':
                ob[keys] = np.pad(ob[keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
        return ob

    def eight_ind_padder(self, ob):
        for keys in ob[0].keys():
            if (keys == '0' or keys =='1' or keys =='2' or keys=='3'):
                ob[0][keys] = np.pad(ob[0][keys], ((1,0),(0,0)), 'constant', constant_values=(0,0))
        return ob

    def padObservation(self, ob, num_agents):
        match num_agents:
            case 8:
                return self.eight_padder(ob)
            case 6:
                return self.six_padder(ob)
            case 4:
                return self.four_padder(ob)
            case 3:
                return self.three_padder(ob)
            case _:
                raise Exception(f"[Error] (padObservation): invalid num_agents: {num_agents} provided")

    def test(self, eval_index):

        avg_score = []
        avg_vehicles = []
        avg_timesteps = []
        avg_speed = []
        avg_max_speed = []
        avg_travel_time = []
        avg_wait_time = []
        avg_time_loss = []

        self.stack_state_initialiser()
        observation, _ = self.env.reset(single=-1)
        observation = self.padObservation(observation, self.num_agents)
        obs_channelled = self.makeChannelledGraph(observation, 0)

        for i in range(self.total_simulation):

            simulation_score = []

            time_step = 0
            terminate = False
            while not terminate:
                time_step += 1
                action = self.take_action(obs_channelled)

                next_observation, reward, terminate, truncate, reward_info = self.env.step(action)

                next_observation = self.padObservation(next_observation, self.num_agents)
                next_obs_channelled = self.makeChannelledGraph(next_observation, 0)

                observation = next_observation
                obs_channelled = next_obs_channelled

                score = 0
                num_vehicles = 0
                for key in reward_info.keys():
                    score += reward[key]
                    num_vehicles += reward_info[key]['reward']
                if num_vehicles != 0:
                    simulation_score.append(score)

            if terminate:
                self.stack_state_initialiser()
                observation, run_stats = self.env.reset(single=-1)

                observation = self.padObservation(observation, self.num_agents)
                obs_channelled = self.makeChannelledGraph(observation, 0)

                total_speed, total_max_speed, total_travel_time, total_wait_time, total_time_loss = run_stats

                avg_vehicles.append(len(total_speed))
                avg_timesteps.append(time_step)
                avg_score.append(sum(simulation_score)/len(simulation_score))
                avg_speed.append(sum(total_speed)/len(total_speed))
                avg_max_speed.append(sum(total_max_speed)/len(total_max_speed))
                avg_travel_time.append(sum(total_travel_time)/len(total_travel_time))
                avg_wait_time.append(sum(total_wait_time)/len(total_wait_time))
                avg_time_loss.append(sum(total_time_loss)/len(total_time_loss))

                print(getLineDash())
                print(f"avg_score: {avg_score}")
                print(f"avg_vehicles: {avg_vehicles}")
                print(f"avg_timesteps: {avg_timesteps}")
                print(f"avg_speed: {avg_speed}")
                print(f"avg_max_speed: {avg_max_speed}")
                print(f"avg_travel_time: {avg_travel_time}")
                print(f"avg_wait_time: {avg_wait_time}")
                print(f"avg_time_loss: {avg_time_loss}")
                print(getLineDash())
                printNewLine(2)

        eval_result = {"avg_score" : avg_score, 'avg_vehicles' : avg_vehicles, "avg_timesteps" : avg_timesteps, "avg_speed" : avg_speed, "avg_max_speed" : avg_max_speed, "avg_travel_time" : avg_travel_time, "avg_wait_time" : avg_wait_time, "avg_time_loss" : avg_time_loss}
        print(getLineDash())
        print(f"Saving data for evaluation {eval_index}...")
        print(getLineDash())
        saveDataDict(eval_result, eval_index, self.result_dir)
        print(f"[Info] (main): Closing env...")

if __name__=="__main__":
    #************************************************CHANGE PADDER*****************************************
    exp = experimentor(total_simulation=3)
    exp.tester()
