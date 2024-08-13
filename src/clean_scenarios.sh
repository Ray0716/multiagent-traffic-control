#!/usr/bin/env bash

SCENARIOS_NAME="$1"
if [[ -z "$1" ]]; then
    SCENARIOS_NAME="cross_network"
fi
SCENARIOS_DIR="./scenarios/Sumo"
SCENARIOS_DIR_FULL="${SCENARIOS_DIR}/${SCENARIOS_NAME}"
if [[ -d "$SCENARIOS_DIR_FULL" ]]; then
    echo "Deleting sumocfg and rou.xml files from scenarios dir: ${SCENARIOS_DIR_FULL}"
    find "$SCENARIOS_DIR_FULL" -name "*.sumocfg" -delete
    find "$SCENARIOS_DIR_FULL" -name "*.rou.xml" -delete
else
    echo "Couldn't find the scenarios dir: ${SCENARIOS_DIR_FULL}"
fi
