#!/bin/bash
#SBATCH --job-name=eqn_generation
#SBATCH -p single
#SBATCH -A loni_ph_inv
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 8:00:00            # Wall time limit (hh:mm:ss)
#SBATCH -o slurm-%j.out-%N         # Standard output and error
#SBATCH -e slurm-%j.err-%N         # Standard output and error

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Slurm Nodes Allocated          = $SLURM_JOB_NODELIST"
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo

module load gcc/9.3.0
module load intel-mpi/2019.5.281
module load python

# Key variables
parameters=('C2H4' 'CO' 'H2' 'EtoH' 'FORM')
positions=(0 1 2 3 4)
data_file='data.csv'

for i in ${!parameters[@]}; do
    param=${parameters[i]}
    pos=${positions[i]}
    
    echo "***************** Writing the json file for $param *****************"
    mkdir -p ${param}
    cd ${param}

    python /work/tolayi1/eqnbased/src/utils.py --data /work/tolayi1/eqnbased/cleaned_data.csv --index $pos --name $param --normalize

    json_config=$(cat <<EOF
{
    "data_file": "$data_file",
    "property_key": "$param",
    "desc_dim": 6,
    "n_sis_select": 10,
    "max_rung": 2,
    "calc_type": "regression",
    "min_abs_feat_val": 1e-5,
    "max_abs_feat_val": 1e8,
    "n_residual": 10,
    "n_models_store": 1,
    "leave_out_frac": 0.0,
    "leave_out_inds": [],
    "opset": ["add", "sub", "abs_diff", "mult", "div", "inv", "abs", "exp", "log", "sin", "cos", "sq", "cb", "six_pow", "sqrt", "cbrt", "neg_exp"]
}
EOF
    )

    echo "$json_config" > sisso.json
    
    echo "***************** Running the sisso++ for $param *****************"
    mpiexec -n $SLURM_NTASKS /home/tolayi1/Programs/sissopp/bin/sisso++ sisso.json

    echo "***************** Done with $param *****************"
    cd ..
done

