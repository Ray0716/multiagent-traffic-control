import gymnasium as gym
import logging
from gymnasium import spaces
import os
from aienvs.Sumo.LDM import ldm
from aienvs.Sumo.SumoHelper import SumoHelper
from aienvs.Sumo.state_representation import *
import time
import os, sys
os.environ["SUMO_HOME"] = "/opt/homebrew/Cellar/sumo/1.20.0/share/sumo"

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import random
from aienvs.Sumo.SumoHelper import SumoHelper
from aienvs.Environment import Env
import copy
import time
from aienvs.Sumo.TrafficLightPhases import TrafficLightPhases
import yaml
from aienvs.Sumo.statics_control import calculateStats
from gymnasium.spaces import Box
import numpy as np
from collections import OrderedDict

class SumoGymAdapter(Env):
    """
    An adapter that makes Sumo behave as a proper Gym environment.
    At top level, the actionspace and percepts are in a Dict with the
    trafficPHASES as keys.

    @param maxConnectRetries the max number of retries to connect.
        A retry is needed if the randomly chosen port
        to connect to SUMO is already in use.
    """
    _DEFAULT_PARAMETERS = {'gui':True,  # gui or not
                'scene':'four_grid',  # subdirectory in the aienvs/scenarios/Sumo directory where
                'tlphasesfile':'cross.net.xml',  # file
                'box_bottom_corner':(0, 0),  # bottom left corner of the observable frame
                'box_top_corner':(10, 10),  # top right corner of the observable frame
                'resolutionInPixelsPerMeterX': 1,  # for the observable frame
                'resolutionInPixelsPerMeterY': 1,  # for the observable frame
                'y_t': 6,  # yellow time
                'car_pr': 0.5,  # for automatic route/config generation probability that a car appears
                'car_tm': 2,  #  for automatic route/config generation when the first car appears?
                'route_starts' : [],  #  for automatic route/config generation, ask Rolf
                'route_min_segments' : 0,  #  for automatic route/config generation, ask Rolf
                'route_max_segments' : 0,  #  for automatic route/config generation, ask Rolf
                'route_ends' : [],  #  for automatic route/config generation, ask Rolf
                'generate_conf' : True,  # for automatic route/config generation
                'libsumo' : False,  # whether libsumo is used instead of traci
                'waiting_penalty' : 1,  # penalty for waiting
                'new_reward': False,  # some other type of reward ask Miguel
                'lightPositions' : {},  # specify traffic light positions
                'scaling_factor' : 1.0,  # for rescaling the reward? ask Miguel
                'maxConnectRetries':50,  # maximum reattempts to connect by Traci
                }

    def __init__(self, root_dir, output_dir, parameters:dict={}):
        """
        @param path where results go, like "Experiment ID"
        @param parameters the configuration parameters.
        gui: whether we show a GUI.
        scenario: the path to the scenario to use
        """
        logging.debug(parameters)

        self._step_length = 1

        self._parameters = copy.deepcopy(self._DEFAULT_PARAMETERS)
        self._parameters.update(parameters)
        # dirname = os.path.dirname(__file__)
        # tlPhasesFile = os.path.join(dirname, "../../../scenarios/Sumo/", self._parameters['scene'], self._parameters['tlphasesfile'])
        self.scenerio_root_dir = os.path.join(root_dir, "scenarios/Sumo")
        self.scenerio_dir = os.path.join(self.scenerio_root_dir, self._parameters['scene'])
        tlPhasesFile = os.path.join(self.scenerio_dir, self._parameters['tlphasesfile'])
        self._tlphases = TrafficLightPhases(tlPhasesFile)
        self.ldm = ldm(using_libsumo=self._parameters['libsumo'])
        self._takenActions = {str(id): [] for id in self._tlphases.getIntersectionIds()}
        self._yellowTimer = {}
        self._chosen_action = None
        self.seed(42)  # in case no seed is given
        self._action_space = self._getActionSpace()
        print(f"action_space: {self._action_space}")
        print(f"action_space sample: {self.getActionSample()}")
        # self.stats_dir = os.path.join(dirname, "../../../Output/Stats")
        self.out_dir = output_dir
        self.stats_dir = os.path.join(self.out_dir, "Stats")
        trip_dirname = self._parameters['tripinfofolder']
        self.trip_dir = os.path.join(self.stats_dir, trip_dirname)
        self.trip_file = os.path.join(self.trip_dir, "tripinfo.xml")
        if not os.path.isdir(self.trip_dir):
            print(f"Creating output-info dir {self.trip_dir}")
        os.makedirs(self.trip_dir, exist_ok=True)
        # self.stats_control = Control(self.trip_dir)
        self.running_sumo = False
        self.test_seeds = self._parameters.get('test', False)
        self.testseed = self._parameters.get('test_seed', [])
        self.seed_cntr = 0
        #self.factor_graph = self._parameters['factored_agents']
        #self.n_factors = len(list(self.factor_graph.keys()))
        sumo_binary = 'sumo-gui' if self._parameters['gui'] else 'sumo'
        # sumo_binary = 'flatpak run org.eclipse.sumo'
        self.sumo_binary = checkBinary(sumo_binary)
        self.max_retries = self._parameters['maxConnectRetries']
        self.factor_agents = self._parameters['factored_agents']
        self.n_factors = len(list(self.factor_agents.keys()))
        if self.n_factors > 1:
            self.factored_coords = self._parameters['factored_coords']

    def step(self, actions:dict):
        self._set_lights(actions)
        self.ldm.step()
        obs = self._observe()
        done = self.ldm.isSimulationFinished()
        total_result = self._computeGlobalReward()
        if self.n_factors > 1:
            reward = {}
            for key in total_result.keys():
                reward[key] = total_result[key]['reward']
        else:
            reward = total_result['reward']
        '''self.action_switches(actions)
        actual_reward = self.actual_global_reward(global_reward)'''

        # as in openai gym, last one is the info list
        return obs, reward, done, False, total_result

    '''def actual_global_reward(self, global_reward):
        global_reward['result'] += -0.1*self._action_switches['0']
        return global_reward

    def action_switches(self, actions:spaces.Dict):
        self._action_switches = {}
        for intersectionId in actions.keys():
            if len(self._takenActions[intersectionId])==1:
                self._action_switches[intersectionId] = 0
            else:
                prev_action = self._takenActions[intersectionId][-1]
                if prev_action != self._intToPhaseString(intersectionId, actions.get(intersectionId)):
                    self._action_switches[intersectionId] = 1
                else:
                    self._action_switches[intersectionId] = 0
        return self._action_switches'''

    def reset(self, single: int):
        run_stats = None
        if self.running_sumo:
            logging.debug("LDM closed by resetting")
            self.close()
            run_stats = self._getStats()

        logging.info("Starting SUMO environment...")
        self._startSUMO(single)
        self._step_length = self.ldm.getStepLength()
        # TODO: Wouter: make state configurable ("state factory")
        if self.n_factors > 1:
            self._state = FactoredLDMMatrixState(self.ldm, [self._parameters['box_bottom_corner'], self._parameters['box_top_corner']], factored_agents=self.factor_agents, factored_coords=self.factored_coords)
        else:
            self._state = LdmMatrixState(self.ldm, [self._parameters['box_bottom_corner'], self._parameters['box_top_corner']], self._parameters["type"])
        return self._observe(), run_stats

    def _getStats(self):
        logging.info(f"Returning results")
        return calculateStats(self.trip_file, self._step_length)

    def getStepLength(self):
        return self._step_length

    # TODO: change the defaults to something sensible
    def render(self, delay=0.0):
        import colorama
        colorama.init()

        def move_cursor(x, y):
            print ("\x1b[{};{}H".format(y + 1, x + 1))

        def clear():
            print ("\x1b[2J")

        clear()
        move_cursor(100, 100)
        import numpy as np
        np.set_printoptions(linewidth=100)
        print(self._observe())
        time.sleep(delay)

    def seed(self, seed):
        random.seed(seed)
        self._seed = int(time.time())

    def reset_test_cntr(self):
        self.seed_cntr = 0

    def testSeeds(self, on: bool):
        self.test_seeds = on

    def close(self):
        if self.running_sumo:
            self.ldm.close()
            self.running_sumo = False
        else:
            logging.debug("No LDM to close. Perhaps it's the first instance of training")

    @property
    def observation_space(self):
        size = self._state.size()
        return Box(low=0, high=np.inf, shape=(size[0], size[1]), dtype=np.int32)
        # return self._state.update_state()

    @property
    def action_space(self):
        return self._action_space

    ########## Private functions ##########################
    def __del__(self):
        logging.debug("LDM closed by destructor")
        self.close()
        # if 'ldm' in locals():
        #     self.ldm.close()

    def _startSUMO(self, single: int):
        """
        Start the connection with SUMO as a subprocess and initialize
        the traci port, generate route file.
        """
        max_retries = self.max_retries
        retries = 0
        # Try repeatedly to connect
        while True:
            # this cannot be seeded
            self._port = random.SystemRandom().choice(list(range(10000, 20000)))
            if not self.test_seeds:
                self._seed = random.randint(0, 276574)
                logging.info(f"The seed used for route generation: {self._seed}")
            else:
                logging.info(f"The test seeds counter: {self.seed_cntr}")
                if self.seed_cntr >= len(self.testseed):
                    raise Exception(f"[Error] (SumoGymAdapter::_startSUMO): invalid index: {self.seed_cntr} access for testseed list of length: {len(self.testseed)}")
                self._seed = self.testseed[self.seed_cntr]
                logging.info(f"The test seed used for route generation: {self._seed}")
                self.seed_cntr +=1
            self._sumo_helper = SumoHelper(self.scenerio_dir, self._parameters, self._port, int(self._seed), single)
            conf_file = self._sumo_helper.sumocfg_file
            logging.info("Configuration: " + str(conf_file))
            sumoCmd = [self.sumo_binary, "-c", conf_file, "--tripinfo-output", self.trip_file, "--seed", str(self._seed)]
            try:
                self.ldm.start(sumoCmd, self._port)
            except Exception as e:
                # if str(e) == "connection closed by SUMO" and retries > 0:
                if retries != max_retries:
                    print(f"[Error] (_startSUMO): {e}")
                    print(f"[Error] (_startSUMO): retrying {retries}/{max_retries}")
                    retries += 1
                    if self.test_seeds:
                        self.seed_cntr -= 1
                    continue
                else:
                    raise e
            else:
                break

        self.running_sumo = True
        self.ldm.init(waitingPenalty=self._parameters['waiting_penalty'], new_reward=self._parameters['new_reward'])  # ignore reward for now
        # used to set boundaries to compute the network space and it computes the states, i can use this to compute states based on the fatored graphs.
        self.ldm.setResolutionInPixelsPerMeter(self._parameters['resolutionInPixelsPerMeterX'], self._parameters['resolutionInPixelsPerMeterY'])
        self.ldm.setPositionOfTrafficLights(self._parameters['lightPositions'])

        if len(list(self.ldm.getTrafficLights())) != len(self._tlphases.getIntersectionIds()):
            raise Exception("environment traffic lights do not match those in the tlphasesfile "
                    +self._parameters['tlphasesfile'] + str(self.ldm.getTrafficLights())
                    +str(self._tlphases.getIntersectionIds()))

    def _intToPhaseString(self, intersectionId:str, lightPhaseId: int):
        """
        @param intersectionid the intersection(light) id
        @param lightvalue the PHASES value
        @return the intersection PHASES string eg 'rrGr' or 'GGrG'
        """
        logging.debug("lightPhaseId" + str(lightPhaseId))
        return self._tlphases.getPhase(intersectionId, lightPhaseId)

    def _observe(self):
        """
        Fetches the Sumo state and converts in a proper gym observation.
        The keys of the dict are the intersection IDs (roughly, the trafficLights)
        The values are the state of the TLs
        """
        return self._state.update_state()

    def _computeGlobalReward(self):
        """
        Computes the global reward
        """
        return self._state.update_reward()

    def getActionSize(self):
        interscn_id = self._tlphases.getIntersectionIds()[0]
        return self._tlphases.getNrPhases(interscn_id)

    def getActionSample(self):
        return self._action_space.sample()

    def _getActionSpace(self):
        """
        @returns the actionspace: a dict containing <id,phases> where
        id is the intersection id and value is
         all possible actions for each id as specified in tlphases
        """
        return spaces.Dict({inters:spaces.Discrete(self._tlphases.getNrPhases(inters)) \
                            for inters in self._tlphases.getIntersectionIds()})

    def _set_lights(self, actions: OrderedDict):
        """
        Take the specified actions in the environment
        @param actions a list of
        """
        for intersectionId in actions.keys():
            action = self._intToPhaseString(intersectionId, actions.get(intersectionId))
            if len(self._takenActions[intersectionId]) != 0:
                # Retrieve the action that was taken the previous step
                prev_action = self._takenActions[intersectionId][-1]
            else:
                prev_action = action
                self._yellowTimer.update({intersectionId:0})

            # Check if the given action is different from the previous action
            if prev_action != action:
                # Either the this is a true switch or coming grom yellow
                action, self._yellowTimer[intersectionId] = self._correct_action(prev_action, action, self._yellowTimer[intersectionId])

            # Set traffic lights
            self.ldm.setRedYellowGreenState(intersectionId, action)
            self._takenActions[intersectionId].append(action)

    def _correct_action(self, prev_action, action, timer):

        """
        Check what we are going to do with the given action based on the
        previous action.
        """
        # Check if the agent was in a yellow state the previous step
        if 'y' in prev_action:
            # Check if this agent is in the middle of its yellow state
            if timer > 0:
                new_action = prev_action
                timer -= 1
            # Otherwise we can get out of the yellow state
            else:
                new_action = self._chosen_action
                if not isinstance(new_action, str):
                    raise Exception("chosen action is illegal")
        # We are switching from green to red, initialize the yellow state
        else:
            self._chosen_action = action
            if self._parameters['y_t'] > 0:
                new_action = prev_action.replace('G', 'y')
                timer = self._parameters['y_t'] - 1
            else:
                new_action = action
                timer = 0

        return new_action, timer

