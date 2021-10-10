!/bin/bash

#$-l rt_C.large=1
#$-l h_rt=05:00:00 
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/7.4.0
module load intel-mpi/2019.9

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH

source ~/.bash_profile

PYTHONUTF8=1

cd /home/acc12675ut/D1/NARF_release/data/THUman
pyenv shell miniconda2-4.7.12

python render_THUman.py --config_path /home/acc12675ut/D1/NARF_release/data/THUman/config.py
