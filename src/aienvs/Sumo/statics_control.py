import os
import numpy as np
import xml.etree.ElementTree as ET

"""
Calculate the mean of all the values we store per factor and write
the summaries.
"""
def calculateStats(trip_info_file, step_length = 1.):
    success = False
    i = 0

    tree = ET.parse(trip_info_file)
    # while not success:
    #     # Try to copy the output file of SUMO
    #     try:
    #         tree = ET.parse(trip_file)
    #         success = True
    #     except:
    #         print(f"Could not load output file: {trip_info_file}")
    #         success = False

    #         i +=1
    #         if i == 20:
    #             break

    # if not success:
    #     raise Exception(f"[Error] (calculateStats): couldn't open trip_info_file: {trip_info_file}")

    data = tree.getroot()
    total_speed = []
    total_max_speed = []
    total_travel_time = []
    total_wait_time = []
    total_time_loss = []

    for car in data:
        duration = float(car.attrib['duration'])
        speed = float(car.attrib['routeLength'])/duration
        total_speed.append(speed)
        total_travel_time.append(duration)
        total_wait_time.append(float(car.attrib['waitingTime']))
        time_loss = float(car.attrib['timeLoss'])
        total_time_loss.append(time_loss)
        max_speed = (speed * duration)/(duration - time_loss)
        total_max_speed.append(max_speed)

    return total_speed, total_max_speed, total_travel_time, total_wait_time, total_time_loss
