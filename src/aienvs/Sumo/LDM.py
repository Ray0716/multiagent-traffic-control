import os
import sys
import logging
import string
import numpy as np
os.environ["SUMO_HOME"] = "/opt/homebrew/Cellar/sumo/1.20.0/share/sumo"


class ldm():
    '''
    An LDM (Local Dynamic Map) module contains the positions and other state attributes of dynamic objects
    in the simulation (vehicles, possibly also traffic lights)
    and adapts the vehicles in a platoon to change their controls accordingly.
    Usage -- as a module: from LDM import ldm (has to be imported after traci.start )
    Then call ldm.init()

    Public methods: getMapSliceByCorners( bottomLeftCoords, topRightCoords )
    getMapSliceByCenter( self, centerCoords, widthInMeters, heightInMeters )
    '''

    def __init__(self, using_libsumo=True):
        if( using_libsumo ):
            import libsumo as SUMO_client
        else:
            import traci as SUMO_client
            self.tc = SUMO_client.constants

        self.SUMO_client = SUMO_client

        # if 'SUMO_HOME' in os.environ:
        #     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        #     sys.path.append(tools)
        # else:
        #     raise ValueError("Environment variable SUMO_HOME is not set, "
        #                      "please declare it (e.g. in ~/.bashrc).")
        # should be added once only, otherwise multiple step listeners are created
        self._lightids={}


    #TODO: Wouter: change all verbose prints to logging
    def init(self, waitingPenalty, new_reward, verbose=0):
        ''' LDM()

        Creates and initializes the Local Dynamic Map
        Call after traci has connected
        '''
        self.__optimize=False #set True to disable non-optimized public functions
        self.netBoundaryMeters=list(self.SUMO_client.simulation.getNetBoundary())
        print(f"netBoundaryMeters simulation: {self.netBoundaryMeters}")
        self.netBoundaryMetersLL=list([self.netBoundaryMeters[0][0]-10, self.netBoundaryMeters[0][1]-10])
        self.netBoundaryMetersUR=list([self.netBoundaryMeters[1][0]+10, self.netBoundaryMeters[1][1]+10])
        self.netBoundaryMeters=list( [tuple(self.netBoundaryMetersLL), tuple(self.netBoundaryMetersUR)] )
        print(f"netBoundaryMeters final: {self.netBoundaryMeters}")

        self._verbose=verbose
        self._lightids=self.SUMO_client.trafficlight.getIDList()
        self._subscribeToTrafficLights()
        self._lightstate={}
        self._tlPositions={}
        self._waitingPenalty = waitingPenalty
        self.new_reward = new_reward
        self._vehicles_id=[]
        self._vehicles_info={}
        self.prev_speed={}
        self._veh_sub_details = (self.tc.VAR_POSITION, self.tc.VAR_SPEED, self.tc.VAR_ALLOWED_SPEED, self.tc.VAR_WAITING_TIME, self.tc.VAR_TIMELOSS, self.tc.VAR_ACCEL)

    def start(self, sumoCmd:list, PORT:9001):
        """
        @param sumoCmd the sumo command for the start, list of init arguments
        """
        logging.info("Sumo command:" + str(sumoCmd))
        # traci needs a port, libsumo doesn't
        if( self.SUMO_client.__name__ == "traci" ):
           self.SUMO_client.start(sumoCmd, port=PORT)
        else:
            self.SUMO_client.start(sumoCmd)

    def step(self):
        '''
        This updates the vehicles' states with information from the simulation
        '''
        try:
            self.SUMO_client.simulationStep()
        except self.SUMO_client.TraCIException as exc:
            logging.error(str(exc) + str(" This is some problem of libsumo, but everything still seems to work correctly"))

        self._resetMap()

        self._vehicles_id = self.getVehiclesID()

        if len(self._vehicles_id) > 0:
            self._updateVehiclesInfo()

            if self._vehicles_id != list(self._vehicles_info.keys()):
                raise Exception(f"[Error] (LDM::step): _vehicles_id: {self._vehicles_id} doesn't match _vehicles_info.keys(): {self._vehicles_info.keys()}")

            self._updateMapWithVehicles()

        tls_info = self.getTrafficLightsInfo()
        self._updateTrafficLights(tls_info)
        self._updateMapWithTrafficLights()

        return True

    def _updateVehiclesInfo(self):
        self._vehicles_info={}
        for veh_id in self._vehicles_id:
            veh_info = self._getVehicleInfo(veh_id)
            self._vehicles_info[veh_id] = veh_info
            logging.debug(f"Vehicle: {veh_id} variable info:\n{veh_info}")

    def _getVehicleInfo(self, veh_id):
        self._addVehicleSubscription(veh_id)
        subscription_result = self.SUMO_client.vehicle.getSubscriptionResults(veh_id)
        return subscription_result

    def _addVehicleSubscription(self, veh_id):
        self.SUMO_client.vehicle.subscribe(veh_id, self._veh_sub_details)

    def _updateTrafficLights(self, lightupdates):
        """
        update the trafficlights cache
        I guess the lightupdate is according to https://sumo.dlr.de/wiki/TraCI/Traffic_Lights_Value_Retrieval
        """
        for lightid in lightupdates:
            lightstate = lightupdates[lightid][self.tc.TL_RED_YELLOW_GREEN_STATE]
            if(self._verbose):
                print("Light " + lightid + "=" + lightstate)
            self._lightstate[lightid] = lightstate;

    def getTrafficLightsInfo(self):
        tls_info = {}
        for lightid in self._lightids:
            tls_info[lightid] = self.SUMO_client.trafficlight.getSubscriptionResults(lightid)
        return tls_info

    def _updateMapWithVehicles(self):
        vehicles_info = self._vehicles_info
        for veh_id in vehicles_info:
            veh_coords = vehicles_info[veh_id][self.tc.VAR_POSITION]
            veh_coords = tuple((round(veh_coords[0]), round(veh_coords[1])))
            if(self._verbose):
                print(f"Vehicle id: {veh_id}, Coordinates: {veh_coords}")

            veh_coords_inArray = self._coordMetersToArray(veh_coords)
            veh_coords_inArray = self._clipCoords(veh_coords_inArray)
            self._arrayMap[veh_coords_inArray[0], veh_coords_inArray[1]] = self._arrayMap[veh_coords_inArray[0], veh_coords_inArray[1]] + 1
            # try:
            #     self._arrayMap[veh_coords_inArray[0], veh_coords_inArray[1]] = self._arrayMap[veh_coords_inArray[0], veh_coords_inArray[1]] + 1
            # except IndexError as error:
            #     print(error)

    def _updateTrafficLight(self, lights, position):
        """
        Add the right value for the state of the traffic light at the right
        position in the matrix.
        Keyword arguments:
            state_matrix -- the matrix in which the values are stores
            light_color  -- a tuple containing the state of the traffic light
            position     -- a tuple containing the traffic light position
            traci        -- instance of TraCI to communicate with SUMO
        """
        for index in range(len(lights)):
            if lights[index] == 'G':
                val = 0.8
            elif lights[index] == 'y':
                val = 0.5
            elif lights[index] == 'r':
                val = 0.2

            arrayPosition = self._coordMetersToArray( position[index])
            self._arrayMap[arrayPosition[0], arrayPosition[1]] += val

    def _updateMapWithTrafficLights(self):
        for lightid in self._lightids:
            if(self._tlPositions.get(lightid) != None):
                self._updateTrafficLight(self._lightstate[lightid], list(self._tlPositions[lightid]))

    def close(self):
        """
        close sumo env
        """
        self.SUMO_client.close()

    def getStepLength(self):
        return self.SUMO_client.simulation.getDeltaT()

    def isSimulationFinished(self):
        """
        @return minimum number of vehicles that are still expected to leave the net (id 0x7d)
        """
        return (self.SUMO_client.simulation.getMinExpectedNumber() <= 0)

    def getRewardByCorners(self, bottomLeftCoords, topRightCoords, local_rewards):
        vehicles_info = self._vehicles_info
        vehicles_info_filtered = vehicles_info.copy()
        if local_rewards:
            for veh_id in vehicles_info:
                position = vehicles_info[veh_id][self.tc.VAR_POSITION]

                if(position[0] < bottomLeftCoords[0]):
                    vehicles_info_filtered.pop(veh_id)
                    continue
                if(position[0] > topRightCoords[0]):
                    vehicles_info_filtered.pop(veh_id)
                    continue
                if(position[1] < bottomLeftCoords[1]):
                    vehicles_info_filtered.pop(veh_id)
                    continue
                if(position[1] > topRightCoords[1]):
                    vehicles_info_filtered.pop(veh_id)
                    continue

        # print(f"return shape: {len(self._computeReward(filteredVehicles))}")
        # reward, reward_info = self._computeReward(filteredVehicles)
        return self._computeReward(vehicles_info_filtered)

    def getRewardByCenter(self, centerCoords, widthInMeters, heightInMeters):
        vehicles_info = self._vehicles_info
        vehicles_info_filtered = vehicles_info.copy()
        for veh_id in vehicles_info:
            position = vehicles_info[veh_id][self.tc.VAR_POSITION]

            if(position[0] < centerCoords[0] - heightInMeters/2.):
                vehicles_info_filtered.pop(veh_id)
                continue
            if(position[0] > centerCoords[0] + heightInMeters/2.):
                vehicles_info_filtered.pop(veh_id)
                continue
            if(position[1] < centerCoords[1] - widthInMeters/2.):
                vehicles_info_filtered.pop(veh_id)
                continue
            if(position[1] > centerCoords[0] + widthInMeters/2.):
                vehicles_info_filtered.pop(veh_id)
                continue

        return self._computeReward(vehicles_info_filtered)

    def getMapSliceByCorners(self, bottomLeftCoords, topRightCoords):
        bottomLeftMatrixCoords = self._coordMetersToArray( bottomLeftCoords)
        topRightMatrixCoords = self._coordMetersToArray( topRightCoords)
        return self._arrayMap[bottomLeftMatrixCoords[0]:(topRightMatrixCoords[0]), bottomLeftMatrixCoords[1]:(topRightMatrixCoords[1])].transpose()[::-1]

    def getMapSliceByCenter(self, centerCoords, widthInMeters, heightInMeters ):
        bottomLeftCoords = (centerCoords[0] - widthInMeters/2., centerCoords[1] - heightInMeters/2.)
        topRightCoords = (centerCoords[0] + widthInMeters/2., centerCoords[1] + heightInMeters/2.)
        return self.getMapSliceByCorners(bottomLeftCoords, topRightCoords )

    def setResolutionInPixelsPerMeter( self, pixelsPerMeterWidth, pixelsPerMeterHeight ):
        self._pixelsPerMeterWidth=pixelsPerMeterWidth
        self._pixelsPerMeterHeight=pixelsPerMeterHeight
        print(f"pixelsPerMeterWidth, pixelsPerMeterHeight: ({self._pixelsPerMeterWidth}, {self._pixelsPerMeterHeight})")
        self._initializeArrayMap()
        self.arrayX_max, self.arrayY_max = self._arrayMap.shape
        print(f"state array shape: {self._arrayMap.shape}")

    def setResolutionInMetersPerPixel( self, metersPerPixelWidth, metersPerPixelHeight ):
        self.setResolutionInPixelsPerMeter( 1./metersPerPixelWidth, 1./metersPerPixelHeight )

    def setPositionOfTrafficLights( self, lightsPositions ):
        for lightID in lightsPositions.keys():
            self.setPositionOfTrafficHeads( lightID, lightsPositions.get(lightID) )

    def setPositionOfTrafficHeads( self, lightID, positionsInMeters ):
        self._tlPositions[lightID] = positionsInMeters

    ######## getting trafficlight info. Maybe move to TrafficLight object #######
    def getTrafficLights(self):
        """
        @return the list[string] of all traffic light ids
        """
        return self._lightids

    def getLightState(self, tlid):
        """
        @param tlid the id of a traffic light
        @return the state of the traffic light with given tlid
        """
        return self._lightstate[tlid]

    ######## getting lane info. Maybe move to Map object #######
    def getControlledLanes(self, lightid:str):
        """
        @param lightid the id of the traffic light
        @return the lanes controlled by the given lightid
        """
        return self.SUMO_client.trafficlight.getControlledLanes(lightid)

    def getLaneMaxSpeed(self, laneid:str):
        """
        @param lane the id of a lane
        @return the maximum speed on the lane
        """
        self.SUMO_client.lane.getMaxSpeed(laneid)

    def getLaneShape(self, laneid:str):
        """
        @param lane the id of a lane
        @return the shape of the lane
        """
        self.SUMO_client.lane.getShape(laneid)

    def getLaneVehicles(self, laneid:str):
        """
        @param lane the id of a lane
        @return the vehicles on this lane
        """
        return self.SUMO_client.lane.getLastStepVehicleIDs(laneid)

    def getVehiclesID(self):
        """
        @return the list[string] of vehicle ids
        """
        return list(self.SUMO_client.vehicle.getIDList())

    def getVehicleLane(self, vehicleid:str):
        """
        @param vehicleid the id of the vehicle
        @return  the lane id where the vehicle is at this time
        """
        return self.SUMO_client.vehicle.getLaneID(vehicleid)

    def getVehicleWaitingTime(self,vehicleid:str):
        """
        @param vehicleid the id of the vehicle
        @return  the waiting time of the vehicle
        """
        return self.SUMO_client.vehicle.getWaitingTime(vehicleid)

    def getVehicleCO2Emission(self, vehicleid:str):
        """
        @param vehicleid the id of the vehicle
        @return vehicle co2 emission
        """
        return self.SUMO_client.vehicle.getCO2Emission(vehicleid)

    def getFuelConsumption(self, vehicleid):
        """
        @param vehicleid the id of the vehicle
        @return vehicle fuel consumption
        """
        return self.SUMO_client.vehicle.getFuelConsumption(vehicleid)

    def getSpeed(self, vehicleid):
        """
        @param vehicleid the id of the vehicle
        @return  the current speed of the vehicle
        """
        return self.SUMO_client.vehicle.getSpeed(vehicleid)

    def getVehicleMaxSpeed(self, vehicleid):
        """
        @param vehicleid the id of the vehicle
        @return  the maximum speed of the vehicle
        """
        return self.SUMO_client.vehicle.getMaxSpeed(vehicleid)

    def getVehicleAllowedSpeed(self, vehicleid):
        """
        @param vehicleid the id of the vehicle
        @return  the allowed speed of the vehicle
        """
        return self._vehicles_info.get(vehicleid).get(self.tc.VAR_ALLOWED_SPEED)

    def getVehiclePosition(self, vehicleid):
        """
        @param vehicleid the id of the vehicle
        @return  the position of the vehicle, unscaled, as in the sumo map
        """
        return self.SUMO_client.vehicle.getPosition(vehicleid)

    def getStartingTeleportNumber(self) :
        """
        @return unknown
        """
        return self.SUMO_client.simulation.getStartingTeleportNumber()
    ########################## private functions ##############################

    def _subscribeToTrafficLights(self):
        logging.info("LightID subscriptions" + str(self._lightids))
        for lightid in self._lightids:
            self.SUMO_client.trafficlight.subscribe(lightid, (self.tc.TL_RED_YELLOW_GREEN_STATE, self.tc.TL_CURRENT_PHASE))

    def _initializeArrayMap( self ):
        if( self._verbose ):
            print( self.netBoundaryMeters[1] )
            print( self.netBoundaryMeters[0] )

        self._arrayMap=np.zeros( self._coordMetersToArray(tuple(( self.netBoundaryMeters[1][0], self.netBoundaryMeters[1][1] )) ) )

    def _resetMap( self ):
        self._arrayMap = np.zeros( self._arrayMap.shape )

    def _clipCoords(self, coords):
        arrayX, arrayY = coords
        arrayX = min(self.arrayX_max-1, arrayX)
        arrayY = min(self.arrayY_max-1, arrayY)
        return [arrayX, arrayY]

    def _coordMetersToArray( self, *coordsInMeters):
        arrayX = round( (coordsInMeters[0][0] - self.netBoundaryMeters[0][0]) * self._pixelsPerMeterWidth - 0.5 )
        arrayY = round( (coordsInMeters[0][1] - self.netBoundaryMeters[0][1]) * self._pixelsPerMeterHeight - 0.5 )
        # print(f"(pixelsPerMeterWidth, pixelsPerMeterHeight): ({self._pixelsPerMeterWidth}, {self._pixelsPerMeterHeight})")
        # print(f"(coordsInMeters[0][0], coordsInMeters[0][1]): ({coordsInMeters[0][0]}, {coordsInMeters[0][1]})")
        # print(f"(netBoundaryMeters[0][0], netBoundaryMeters[0][1]): ({self.netBoundaryMeters[0][0]}, {self.netBoundaryMeters[0][1]})")
        # print(f"(netBoundaryMeters[1][0], netBoundaryMeters[1][1]): ({self.netBoundaryMeters[1][0]}, {self.netBoundaryMeters[1][1]})")
        # print(f"(arrayX, arrayY): ({arrayX}, {arrayY})")
        # raise Exception("[Info] (LDM): check coordsInMeters")
        return [arrayX, arrayY]

    # vehicles are a subset of all subscription results
    def _computeReward(self, vehicles_info):
        total_result = {}
        total_result['reward'] = 0.
        total_result['speed'] = 0.
        total_result['max_speed'] = 0.
        total_result['accel'] = 0.
        total_result['max_accel'] = 0.
        total_result['delay'] = 0.
        total_result['waiting'] = 0.
        total_result['teleports'] = 0.
        total_result['emergency_stops'] = 0.
        total_result['vehicles'] = 0

        if not vehicles_info:
            logging.debug("No vehicles, returning 0 reward")
            return total_result

        for veh_id in vehicles_info:
            veh_info = vehicles_info[veh_id]
            waiting_time = veh_info[self.tc.VAR_WAITING_TIME]

            speed = veh_info[self.tc.VAR_SPEED]
            allowed_speed = veh_info[self.tc.VAR_ALLOWED_SPEED]
            max_accel = veh_info[self.tc.VAR_ACCEL]
            time_loss = veh_info[self.tc.VAR_TIMELOSS]

            if self.new_reward:
                reward = -min(waiting_time, 1.0)
            else:

                previous_speed = self.prev_speed[veh_id] if self.prev_speed.get(veh_id) is not None else 0.

                if( self._verbose ):
                    if self._waitingPenalty:
                        print(veh_id + " waiting_time " + str(waiting_time) + " speed " + str(speed) + " allowed_speed " + str(allowed_speed))
                    else:
                        print(veh_id + " speed " + str(speed) + " allowed_speed " + str(allowed_speed))

                if self._waitingPenalty:
                    if waiting_time<=1:
                        waiting_time = 0.5
                    else:
                        waiting_time = 1

                    #clippedWaitingTime = min(waitingTime*0.5, 1.0)

                    delay = speed/allowed_speed
                    normalised_delay = 1 - delay
                    clippedDelay = max(0, normalised_delay)

                    accel = speed - previous_speed

                    if accel < -4.5:
                        total_result['emergency_stops'] +=1

                    self.prev_speed[veh_id] = speed
                    reward = -0.5*normalised_delay -0.5*waiting_time
                    #reward = - 0.*normalisedDelay - 0.3*clippedWaitingTime

                else:
                    clippedDelay = max(0, 1 - speed / allowedSpeed)
                    reward = -clippedDelay

                if( self._verbose ):
                    if self._waitingPenalty:
                        print(veh_id + " clippedWaitingTime " + str(clippedWaitingTime) + " clippedDelay " + str(clippedDelay) + " reward " + str(reward))
                    else:
                        print(veh_id + " clippedDelay " + str(clippedDelay) + " reward " + str(reward))

            total_result['reward'] += reward
            total_result['speed'] += speed
            total_result['max_speed'] += allowed_speed
            total_result['accel'] += accel
            total_result['max_accel'] += max_accel
            total_result['delay'] += delay
            total_result['waiting'] += waiting_time
            total_result['vehicles'] += 1

        num_teleports = self.getStartingTeleportNumber()

        #total_result['result'] += -0.1*num_teleports -0.2*total_result['emergency_stops']

        total_result['teleports'] += num_teleports
        return total_result

    def _getVehiclePositions(self, vehicles_info):
        resultsFormatted=list(vehicles_info.values())
        positionList = list()

        for vehAttrib in resultsFormatted:
            if(vehAttrib):
                position = (round(vehAttrib[self.tc.VAR_POSITION][0]), round(vehAttrib[self.tc.VAR_POSITION][1]))
                if(self._verbose):
                    print("Position " + str(position))
                positionList.append(position)
        return positionList

    def _getLaneEnds(self, vehicles_info):
        resultsFormatted=list(vehicles_info.values())
        positionList = list()

        for vehAttrib in resultsFormatted:
            position = (round(vehAttrib[self.tc.VAR_POSITION][0]), round(vehAttrib[self.tc.VAR_POSITION][1]))
            if(self._verbose):
                print("Position " + str(position))
            positionList.append(position)
        return positionList

    def setRedYellowGreenState(self, agent:string, state:string ):
        """
        set new state for a traffic  light
        @param agent the agent id
        @param state the new state eg "GrGr"
        """
        self.SUMO_client.trafficlight.setRedYellowGreenState(agent, state)

    def test(self, bottomLeftCoord = (506., 430.), topRightCoord = (516., 500.), centerCoord = (510., 475.), width = 10., height=70. ):
        #mapSlice=str(self.getMapSliceByCorners( bottomLeftCoord, topRightCoord ))
        mapSlice=str(self.getMapSliceByCenter( centerCoord, width, height ))
        #for visualization only
        """
        import colorama
        colorama.init()
        def move_cursor(x,y):
            print ("\x1b[{};{}H".format(y+1,x+1))
        def clear():
            print ("\x1b[2J")
        clear()
        move_cursor(100,100)
        """
        logging.debug(mapSlice)

    def __del__(self):
        """
        close sumo env
        """
        self.SUMO_client.close()


