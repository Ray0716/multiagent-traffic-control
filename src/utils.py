import os
import csv
import numpy as np

_LINE_LENGTH_ = 40

def getLineDash(length: int = _LINE_LENGTH_):
    return "-" * length

def getLineStar(length: int = _LINE_LENGTH_):
    return "*" * length

def getLineHash(length: int = _LINE_LENGTH_):
    return "#" * length

def getLineCustom(text, length: int = _LINE_LENGTH_):
    return text * length

def getNewLine(length: int):
    return "\n" * length

def printNewLine(length: int):
    print("\n" * length)

def saveData(data, filename, save_dir):
    file_path = os.path.join(save_dir, str(filename) + '.csv')
    data_file = open(file_path, 'w')
    writer = csv.writer(data_file)
    writer.writerows(map(lambda x:[x], data))

def saveDataDict(data_dict, extra_dir, save_dir):
    if extra_dir is not None:
        save_dir = os.path.join(save_dir, str(extra_dir))
        os.makedirs(save_dir, exist_ok=True)
    for key in data_dict.keys():
        saveData(data_dict[key], key, save_dir)

def makeChannelled(item, index: int, repeat: int):
    item = np.expand_dims(item, index)
    if repeat != 0:
        item = np.concatenate( (item,)*repeat, axis = index )
    return item.copy()

def stackFrames(num_frames: int, obs, obs_stacked):
    if num_frames == 1:
        obs_stacked = obs
        return obs_stacked
    else:
        tmp = obs_stacked[:-1]
        obs_stacked[1:] = tmp
        obs_stacked[0] = obs[0]
        return obs_stacked

