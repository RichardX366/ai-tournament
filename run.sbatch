#!/bin/bash
#SBATCH -J Tournament
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=80G
#SBATCH -t 0:05:00
#SBATCH -o Output-%j.out

cd $SLURM_SUBMIT_DIR/ai-tournament

VENV_DIR="venv"

module load python/3.12

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
else
    echo "Using existing virtual environment '$VENV_DIR'."
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip

while IFS="" read -r p || [ -n "$p" ]
do
    echo "Installing $p ..."
    pip install $p
    echo ""
done < requirements.txt

echo "Running Game..."

PLAYER_A=${1:?"Missing player A name"}
PLAYER_B=${2:?"Missing player B name"}

python3 -u engine/run_local_agents.py "$PLAYER_A" "$PLAYER_B"
