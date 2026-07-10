# !! Contents within this block are managed by 'conda init' !!
anaconda_path=/home/public_research/base_envs/anaconda3
__conda_setup="$(CONDA_REPORT_ERRORS=false '${anaconda_path}/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "${anaconda_path}/etc/profile.d/conda.sh" ]; then
        . "${anaconda_path}/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="${anaconda_path}/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<
export PATH=$PATH:${anaconda_path}/bin

source activate
conda activate ad_ffgsstudio

export CUDA_HOME=/usr/local/cuda-12.4
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="8.0"

#export CUDA_VISIBLE_DEVICES=0

