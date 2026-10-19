SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
# Kill any existing jupyter-lab processes
pkill -f "jupyter-lab"

# If ../nim-workshop-env/bin/activate exists use it, otherwise just use the system python
if [ -f ${SCRIPT_DIR}/../nim-workshop-env/bin/activate ]; then
    cd $SCRIPT_DIR/..
    # Local environment
    echo "Running on Local environmetn (not launchpad)"
    # Make sure  ~/.local/share/jupyter/kernels/docker-nemo-exec/kernel.json exists 
    if [ ! -f ~/.local/share/jupyter/kernels/docker-nemo-exec/kernel.json ]; then
        echo "WARNING ~/.local/share/jupyter/kernels/docker-nemo-exec/kernel.json does not exist"
        echo "You need to manually create it"
    fi 
    source ./nim-workshop-env/bin/activate && jupyter-lab --allow-root --NotebookApp.token='' --ip='0.0.0.0' --port=8888
else
    # Launchpad environment
    echo "Running in launcpad environment"
    cd $HOME/nim-workshop
    jupyter-lab --allow-root --NotebookApp.token=''  --NotebookApp.base_url='/notebook/' --ip='0.0.0.0' --port=8888 --NotebookApp.disable_check_xsrf=True &
fi




