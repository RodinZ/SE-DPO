set -e
task_name=$1
export HYDRA_FULL_ERROR=1
if [ ! -f "my_data/experiment/${task_name}/eval_res_for_${task_name}.txt" ]; then
    echo "init scoring !"
    sh my_data/experiment/${task_name}/score.sh
fi

