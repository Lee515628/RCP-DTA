# Davis

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export MKL_INTERFACE_LAYER=LP64
export MKL_DEBUG_CPU_TYPE=5
python dynamicCP.py --task Davis
# # KIBA Random
python dynamicCP.py --task KIBA --use-cold-spilt False
# KIBA Cold Target
python dynamicCP.py --task KIBA --use-cold-spilt True --cold target_key --train-seed 0
# KIBA Cold Drug
python dynamicCP.py --task KIBA --use-cold-spilt True --cold Drug --train-seed 0
# KIBA Cold Both
python dynamicCP.py --task KIBA --use-cold-spilt True --cold Drug,target_key --train-seed 0