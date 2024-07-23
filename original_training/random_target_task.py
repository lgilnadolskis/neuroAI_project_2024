import torch
import importlib
import random
import dotenv
import os
import pickle
from motornet.effector import RigidTendonArm26
from motornet.muscle import MujocoHillMuscle
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from ComputationThruDynamicsBenchmark.ctd.task_modeling.task_env.task_env import RandomTarget
from ComputationThruDynamicsBenchmark.ctd.task_modeling.model.rnn import GRU_RNN
from ComputationThruDynamicsBenchmark.ctd.task_modeling.datamodule.task_datamodule import TaskDataModule
from ComputationThruDynamicsBenchmark.ctd.task_modeling.task_wrapper.task_wrapper import TaskTrainedWrapper

# Step 1: Change to the directory containing ComputationThruDynamicsBenchmark
dirname = "ComputationThruDynamicsBenchmark"
if not os.path.isdir(dirname):
    os.system("git clone https://github.com/neuromatch/ComputationThruDynamicsBenchmark")
os.chdir(dirname)
os.system("pip install -e .")

# Step 2: Import the custom callbacks
callbacks_path = "ctd/task_modeling/callbacks/callbacks.py"
spec = importlib.util.spec_from_file_location("custom_callbacks", callbacks_path)
custom_callbacks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_callbacks)

# Step 3: Create and write to the .env file
envStr = """HOME_DIR=ComputationThruDynamicsBenchmark/
TRAIN_INPUT_FILE=train_input.h5
EVAL_INPUT_FILE=eval_input.h5
EVAL_TARGET_FILE=eval_target.h5
"""

with open('.env', 'w') as f:
    f.write(envStr)

# Step 4: Set the random seed
random.seed(2024)
dotenv.load_dotenv(override=True)

HOME_DIR = os.getenv("HOME_DIR")
if HOME_DIR is None:
    HOME_DIR = ""
print(HOME_DIR)

# Step 5: Create the analysis object
rt_task_env = RandomTarget(effector=RigidTendonArm26(muscle=MujocoHillMuscle()))

# Step 6: Instantiate the model
rnn = GRU_RNN(latent_size=128)

# Step 7: Instantiate the task environment
task_env = rt_task_env

# Step 8: Instantiate the task datamodule
task_datamodule = TaskDataModule(task_env, n_samples=1000, batch_size=256)

# Step 9: Instantiate the task wrapper
task_wrapper = TaskTrainedWrapper(learning_rate=1e-3, weight_decay=1e-8)

# Step 10: Initialize the model with the input and output sizes
rnn.init_model(
    input_size=task_env.observation_space.shape[0] + task_env.context_inputs.shape[0],
    output_size=task_env.action_space.shape[0]
)

# Step 11: Set the environment and model in the task wrapper
task_wrapper.set_environment(task_env)
task_wrapper.set_model(rnn)

# Step 12: Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="{epoch}-{val_loss:.2f}",
    save_top_k=3,
    monitor="val_loss",
    mode="min"
)

# Step 13: Initialize the W&B logger
wandb_logger = WandbLogger(project="random_target_task", log_model=True, offline=False)

# Step 14: Define the PyTorch Lightning Trainer object with custom callbacks                    
trainer = Trainer(
    accelerator="cpu" if torch.cuda.is_available() else "cpu",
    max_epochs=500,
    enable_progress_bar=True,
    callbacks=[
        checkpoint_callback,
        custom_callbacks.StateTransitionCallback(log_every_n_epochs=100, plot_n_trials=5),
        custom_callbacks.TrajectoryPlotOverTimeCallback(log_every_n_epochs=100, num_trials_to_plot=5, axis_num=0),
        custom_callbacks.LatentTrajectoryPlot(log_every_n_epochs=10)
    ],
    logger=wandb_logger
)

# Step 15: Fit the model
trainer.fit(task_wrapper, task_datamodule)



#step 16: Save task_wrapper as a .pkl file
with open('task_wrapper.pkl', 'wb') as f:
    pickle.dump(task_wrapper, f)

#step 17: Save task_datamodule as a .pkl file
with open('task_datamodule.pkl', 'wb') as f:
    pickle.dump(task_datamodule, f)