#LOG_FILE="/zhdd/dataset/tyy/VOT/tyy_storage/LLM/RL/log/$(date +'%Y-%m-%d_%H-%M-%S')_logfile_fulldataset_removeioureward.log"  # 添加日期前缀
LOG_FILE="/zssd/tyy/projects/Open_VLTrack/ReasoningTrack/log/training_grpo.log"
bash examples/aqwen2_5_vl_3b_fulldataset_grpo.sh  >  ${LOG_FILE} 2>&1 &

