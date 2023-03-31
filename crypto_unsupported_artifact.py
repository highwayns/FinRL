#
# Trains an CryptoAll, and uses tensorboardX to log training metrics
# and weights in TensorBoard event format to the MLflow run's artifact directory. This stores the
# TensorBoard events in MLflow for later access using the TensorBoard command line tool.
#
# NOTE: This example requires you to first install PyTorch (using the instructions at pytorch.org)
#       and tensorboardX (using pip install tensorboardX).
#
#
import argparse
import os
import mlflow
import mlflow.pytorch
import pickle
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import tensorboardX
from tensorboardX import SummaryWriter
import mlflow.pyfunc
import cloudpickle
from sys import version_info
from finrl.meta.env_cryptocurrency_trading.crypto_all import CryptoAll
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv
from finrl.meta.env_cryptocurrency_trading.env_advance_crypto import AdvCryptoEnv

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("finrl_z")

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                              minor=version_info.minor,
                                              micro=version_info.micro)

# Command-line arguments
parser = argparse.ArgumentParser(description="PyTorch CRYPTO Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)"
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
)
parser.add_argument(
    "--enable-cuda",
    type=str,
    choices=["True", "False"],
    default="True",
    help="enables or disables CUDA training",
)
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()

enable_cuda_flag = True if args.enable_cuda == "True" else False

args.cuda = enable_cuda_flag and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
#fl_model_names = ['multiple','advance']

fl_model_name = 'advance'

class CryptoAllModel(mlflow.pyfunc.PythonModel):

    def __init__(self):
        super().__init__()
        self.model = CryptoAll(fl_model_name)

    def train(self,start_date, end_date, ticker_list, data_source, time_interval, 
            technical_indicator_list, drl_lib, env, model_name, if_vix=False,
            **kwargs):
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)
        mlflow.log_param("ticker_list", ticker_list)
        mlflow.log_param("data_source", data_source)
        mlflow.log_param("time_interval", time_interval)
        mlflow.log_param("technical_indicator_list", technical_indicator_list)
        mlflow.log_param("drl_lib", drl_lib)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("if_vix", if_vix)
        #for key, value in vars(**kwargs).items():
        #    mlflow.log_param(key, value)
        
        self.model.train(start_date, end_date, ticker_list, data_source, time_interval, 
            technical_indicator_list, drl_lib, env, model_name, if_vix,
            **kwargs)

    def test(self,start_date, end_date, ticker_list, data_source, time_interval,
                technical_indicator_list, drl_lib, env, model_name, if_vix=False,
                **kwargs):
        return self.model.test(start_date, end_date, ticker_list, data_source, time_interval,
                technical_indicator_list, drl_lib, env, model_name, if_vix,
                **kwargs)

    def make_plot(self, account_value_erl, path,rl_model_name):
        self.model.make_plot(account_value_erl, path,rl_model_name)
        
model_path = "model"
reg_model_name = "PyFuncCrypto"

conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip'],
    'pip': [
        'mlflow',
        'cloudpickle=={}'.format(cloudpickle.__version__),
        'torch==1.8.1',
        'torchvision==0.9.1',
        'mlflow>=1.0',
        'tensorboardX',
    ],
    'name': 'mlflow-env'
}

TICKER_LIST = ['BTC','ETH','BCH','LTC','XRP', 'XEM','XLM']
INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet

ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}

DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": 0.1,
}
DQN_PARAMS = {
    "learning_rate": 0.01,
    "reward_decay": 0.9,
    "e_greedy": 0.9,
    "replace_target_iter": 300,
    "memory_size": 500,
    "batch_size": 32,
    "e_greedy_increment": None,
}
    
TRAIN_START_DATE = '2022-07-01'
TRAIN_END_DATE = '2022-08-31'

TEST_START_DATE = '2022-09-01'
TEST_END_DATE = '2022-09-30'

DRL_LIB = 'stable_baselines3' #'elegantrl','stable_baselines3'
API_KEY = "1ddcbec72bef777aaee9343272ec1467"
API_SECRET = "dc42d89bed18b4009c9c60a2f6b45fd41daa86bf"
API_BASE_URL = "https://paper-api.alpaca.markets"

DATA_SOURCE='gmo'#'yahoofinance','gmo'
TIME_INTERVAL='1Min'#'1D','1Min'

#rl_model_names = ['A2C','DDPG','PPO','SAC','TD3','DQN']
rl_model_names = ['A2C']
if(fl_model_name == 'multiple'):
    env = CryptoEnv
elif(fl_model_name == 'advance'):
    env = AdvCryptoEnv
else:
    raise ValueError("env is NOT supported. Please check.")

for rl_model_name in rl_model_names:
    CURRENT_WORKING_DIR = './modal/'+fl_model_name+"_"+rl_model_name.lower()
    if(rl_model_name == 'A2C'):
        env_kwargs = {
            "API_KEY": API_KEY, 
            "API_SECRET": API_SECRET, 
            "API_BASE_URL": API_BASE_URL,
            "rllib_params": RLlib_PARAMS,
            "agent_params": A2C_PARAMS,
            "erl_params": ERL_PARAMS,
            "break_step": 5e4,
            "net_dimension": 2**9, 
            "current_working_dir": CURRENT_WORKING_DIR,
        }
    elif(rl_model_name == 'DDPG'):
        env_kwargs = {
            "API_KEY": API_KEY, 
            "API_SECRET": API_SECRET, 
            "API_BASE_URL": API_BASE_URL,
            "rllib_params": RLlib_PARAMS,
            "agent_params": DDPG_PARAMS,
            "erl_params": ERL_PARAMS,
            "break_step": 5e4,
            "net_dimension": 2**9, 
            "current_working_dir": CURRENT_WORKING_DIR,
        }
    elif(rl_model_name == 'PPO'):
        env_kwargs = {
            "API_KEY": API_KEY, 
            "API_SECRET": API_SECRET, 
            "API_BASE_URL": API_BASE_URL,
            "rllib_params": RLlib_PARAMS,
            "agent_params": PPO_PARAMS,
            "erl_params": ERL_PARAMS,
            "break_step": 5e4,
            "net_dimension": 2**9, 
            "current_working_dir": CURRENT_WORKING_DIR,
        }
    elif(rl_model_name == 'SAC'):
        env_kwargs = {
            "API_KEY": API_KEY, 
            "API_SECRET": API_SECRET, 
            "API_BASE_URL": API_BASE_URL,
            "rllib_params": RLlib_PARAMS,
            "agent_params": SAC_PARAMS,
            "erl_params": ERL_PARAMS,
            "break_step": 5e4,
            "net_dimension": 2**9, 
            "current_working_dir": CURRENT_WORKING_DIR,
        }
    elif(rl_model_name == 'TD3'):
        env_kwargs = {
            "API_KEY": API_KEY, 
            "API_SECRET": API_SECRET, 
            "API_BASE_URL": API_BASE_URL,
            "rllib_params": RLlib_PARAMS,
            "agent_params": TD3_PARAMS,
            "erl_params": ERL_PARAMS,
            "break_step": 5e4,
            "net_dimension": 2**9, 
            "current_working_dir": CURRENT_WORKING_DIR,
        }
    elif(rl_model_name == 'DQN'):
        env_kwargs = {
            "API_KEY": API_KEY, 
            "API_SECRET": API_SECRET, 
            "API_BASE_URL": API_BASE_URL,
            "rllib_params": RLlib_PARAMS,
            "agent_params": DQN_PARAMS,
            "erl_params": ERL_PARAMS,
            "break_step": 5e4,
            "net_dimension": 2**9, 
            "current_working_dir": CURRENT_WORKING_DIR,
        }
    else:
        raise ValueError("rl_model is NOT supported. Please check.")

    crypto_model = CryptoAllModel()
    crypto_model.train(start_date=TRAIN_START_DATE, 
        end_date=TRAIN_END_DATE,
        ticker_list=TICKER_LIST, 
        data_source=DATA_SOURCE,
        time_interval=TIME_INTERVAL, 
        technical_indicator_list=INDICATORS,
        drl_lib=DRL_LIB, 
        env=env, 
        model_name=rl_model_name.lower(), 
        if_vix=False,
        **env_kwargs
        )
    account_value_erl = crypto_model.test(start_date = TEST_START_DATE, 
            end_date = TEST_END_DATE,
            ticker_list = TICKER_LIST, 
            data_source = DATA_SOURCE,
            time_interval= TIME_INTERVAL, 
            technical_indicator_list= INDICATORS,
            drl_lib=DRL_LIB, 
            env=env, 
            model_name=rl_model_name.lower(), 
            if_vix=False,
            **env_kwargs
            )
    crypto_model.make_plot(account_value_erl,'data',rl_model_name.lower())                        

    model_info = mlflow.pyfunc.log_model(artifact_path=model_path,
                        python_model=crypto_model,
                        registered_model_name=reg_model_name,
                        conda_env=conda_env
                        )
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
