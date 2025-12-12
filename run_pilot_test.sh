#!/bin/bash
#SBATCH --job-name=Pilot_Test
#SBATCH --output=pilot_%j.out
#SBATCH --error=pilot_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=main

# 1. Setup Environment
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"
export TMPDIR=/dev/shm/$USER/tmp
mkdir -p $TMPDIR
cd $TMPDIR

# 2. Install Tools
# (We assume the repo is already cloned or we clone it fresh)
if [ ! -d "click-spot-repo" ]; then
    git clone https://gitlab.com/hauechri/click-spot-repo.git
fi
cd click-spot-repo

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --no-cache-dir ultralytics wandb

# 3. Get the Pilot Data
# Copy from Home to here
cp $HOME/robustness_pilot_data.zip .
unzip -o -q robustness_pilot_data.zip

# 4. RUN EXPERIMENT LOOP (96k -> 48k -> 24k)
for QUALITY in "96k" "48k" "24k"; do
    echo "--- STARTING TRAINING FOR $QUALITY ---"
    
    # Define paths
    DATA_PATH="$TMPDIR/click-spot-repo/ROBUSTNESS_DATASETS/$QUALITY"
    
    # Create Config File
    cat > ${QUALITY}_config.yaml <<EOL
names:
  0: ignore
  1: event
nc: 2
path: $DATA_PATH
train: images/train
val: images/val
test: images/test
EOL

    # Train YOLO (Nano model for fast testing)
    yolo detect train \
        data=${QUALITY}_config.yaml \
        model=yolov8n.pt \
        epochs=10 \
        imgsz=640 \
        batch=4 \
        project=Pilot_Results \
        name=Run_$QUALITY
done

# 5. Save Results
echo "--- SAVING RESULTS ---"
zip -r pilot_results_all.zip Pilot_Results
mv pilot_results_all.zip $HOME/

echo "--- TEST FINISHED ---"
