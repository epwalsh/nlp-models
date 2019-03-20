#!/bin/bash

# Color output to make more readable.
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
BOLD=$(tput bold)
NC=$(tput sgr0) # No Color

if ! [[ -x $(command -v tensorboard) ]]; then
    echo "${BOLD}${RED}Error:${NC}${RED} tensorboard not installed.${NC}" >&2
    read -r -e -p "Install tensorboard now? [Y/n] " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Installing tensorboard...${NC}" >&2
        pip install tensorboard
    else
        echo -e "${YELLOW}Aborting${NC}" >&2
        exit 1
    fi
fi

if [[ -z $TB_LOGDIR ]] || ! [[ -d "$TB_LOGDIR" ]]; then
    while true; do
        read -r -e -p "${BOLD}Enter log directory: ${NC}" TB_LOGDIR
        if ! [[ -d $TB_LOGDIR ]]; then
            echo -e "${BOLD}${RED}Error:${NC}${RED} invalid directory.${NC}" >&2
        else
            break
        fi
    done
fi

if [[ -z $TB_PORT ]] || ! [[ $TB_PORT =~ ^[0-9]+$ ]]; then
    while true; do
        read -r -e -p "${BOLD}Enter port: ${NC}" TB_PORT
        if ! [[ $TB_PORT =~ ^[0-9]+$ ]]; then
            echo "${BOLD}${RED}Error:${NC}${RED} invalid port.${NC}" >&2
        else
            break
        fi
    done
fi

tensorboard \
    --logdir="$TB_LOGDIR" \
    --port="$TB_PORT" \
    --host=0.0.0.0
