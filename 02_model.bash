#!/bin/bash
echo Directory is $PWD
echo ${SLURM_JOB_NODELIST}

Aminier_18_cfg_path='../config/Aminer-18/cfg.yml'
python main/Aminer18_main.py --run_model='run' --cfg_path=${Aminier_18_cfg_path}

#WiW_cfg_path='../config/WhoIsWho-SND/cfg_50.yml'
#python main/WhoisWhoSND_main.py --run_model='run' --cfg_path=${WiW_cfg_path}
