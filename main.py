import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
from train_test import *
from utils_dta import *
from tqdm import tqdm
from src import model as model_types
from src.utils import (get_logger,
                       config_logger,
                       get_featurizer,
                       set_random_seed,)
from src.data import (get_task_dir,
                      DTADataModule,
                      )
import src.model as module_arch
from src.featurizers.molecule import MolGraphFeaturizer
from src.featurizers.protein import ProteinGNNFeaturizer, ProteinPSSMGNNFeaturizer
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
def str_to_list(arg):
    return arg.split(',')


logg = get_logger()
parser = ArgumentParser(description="DTA Training.")
parser.add_argument("--config", help="YAML config file", default="configs/default_config.yaml")


parser.add_argument("--task",choices=["KIBA","Davis",],type=str,
                    help="Task name. Could be kiba, bindingdb, davis, metz, pdbbind.",dest="task")

parser.add_argument("--drug-seq-featurizer", help="Drug seq featurizer", dest="drug_seq_featurizer")
parser.add_argument("--target-seq-featurizer", help="Target seq featurizer", dest="target_seq_featurizer")
parser.add_argument("--prot-struc-type", type=str, default="default", choices=["default", "pssm"], help="Choose protein structure featurizer: 'default' (61 dim) or 'pssm' (69 dim)")




parser.add_argument("--epochs", "--epochs",type=int, help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", type=int, help="batch size",dest="batch_size")
parser.add_argument("--lr", "--learning-rate",type=float,help="initial learning rate",dest="learning_rate",)


parser.add_argument("--r", "--replicate", type=int, help="Replicate", dest="replicate")
parser.add_argument("--d", "--device", type=int, help="CUDA device", dest="device")
parser.add_argument("--h", "--n-heads", type=int, help="n_heads", dest="n_heads")

parser.add_argument("--drugdim", "--drug-dim", type=int, help="drug_dim", dest="drug_dim")
parser.add_argument("--targetdim", "--target-dim", type=int, help="target_dim", dest="target_dim")
parser.add_argument("--h-dim", "--h-dim", type=int, help="h_dim", dest="h_dim")


parser.add_argument("--w", "--weight-decay", type=float, help="weight_decay", dest="weight_decay")


parser.add_argument("--use-drug-seq", type=str2bool, default=True)
parser.add_argument("--use-drug-struc", type=str2bool, default=True)
parser.add_argument("--use-prot-seq", type=str2bool, default=True) 
parser.add_argument("--use-prot-struc", type=str2bool, default=True)
parser.add_argument("--use-fusion", type=str2bool, default=True, help="Use advanced fusion modules")

parser.add_argument("--model", help="Model", dest="model_architecture")
parser.add_argument("--use-cold-spilt", type=str2bool, default=False)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the saved best model (.pth)")
parser.add_argument("--cold", type=str_to_list, default=['Drug','target_key'])



args = parser.parse_args()
print(args.config)

config = OmegaConf.load(args.config)
arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
config.update(arg_overrides)
logg.info({k:v for k,v in config.items()})


if config.cold == ['Drug']:
    cold_name = 'cold_drug'
elif config.cold == ['target_key']:
    cold_name = 'cold_target'
elif config.cold == ['Drug','target_key']:
    cold_name = 'cold_drug_target'    


root_path = os.getcwd()
current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
if config.use_cold_spilt:
    folder_name = f'output/{config.task}-{cold_name}-{current_time}-seed{config.replicate}'
else:
    folder_name = f'output/{config.task}-random-{current_time}-seed{config.replicate}'

path_model = os.path.join(root_path, folder_name)
if not os.path.exists(path_model):
    os.system("mkdir -p %s" % path_model)

result_file = 'results--%s.txt' % current_time 
model_file = 'model--%s.pth' % current_time
loss_file = 'loss--%s.csv' % current_time
log_file = 'training--%s.log' % current_time
result_test_file = f'results_test_seed{config.replicate}--{current_time}.txt' 
file_results = os.path.join(path_model, result_file)
file_model = os.path.join(path_model, model_file)
file_loss = os.path.join(path_model, loss_file)
file_test_results = os.path.join(path_model, result_test_file)
file_log = os.path.join(path_model,log_file)
f_results = open(file_results, 'a') 

config_logger(file_log,"%(asctime)s [%(levelname)s] %(message)s",config.verbosity,use_stdout=True,)
logg.info(f'config:{config}')

# Set CUDA device
device_no = config.device
use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
logg.info(f"Using CUDA device {device}")
logg.info(f"Weight_decay {config.weight_decay}")


logg.debug(f"Setting random state {config.replicate}")
set_random_seed(config.replicate,deterministic=True)


logg.info("Preparing DataModule")
task_dir = get_task_dir(config.task)

drug_seq_featurizer = get_featurizer(config.drug_seq_featurizer, save_dir=task_dir)


target_seq_featurizer = get_featurizer(config.target_seq_featurizer, save_dir=task_dir)

logg.info("Initializing Structure Featurizers...")
drug_struc_featurizer = MolGraphFeaturizer(save_dir=task_dir) 
if config.prot_struc_type == "pssm":
    logg.info("Using PSSM-enhanced Protein Featurizer (69 dim)")
    target_struc_featurizer = ProteinPSSMGNNFeaturizer(save_dir=task_dir)

    prot_node_dim = 69 
else:
    logg.info("Using Default Protein Featurizer (49 dim)")
    target_struc_featurizer = ProteinGNNFeaturizer(save_dir=task_dir)
    prot_node_dim = 61


datamodule = DTADataModule(
    data_dir=task_dir,
    drug_seq_featurizer=drug_seq_featurizer,
    drug_struc_featurizer=drug_struc_featurizer,
    target_seq_featurizer=target_seq_featurizer,
    target_struc_featurizer=target_struc_featurizer,
    dataset_name=config.task,
    device=device,
    seed=config.replicate,
    use_cold_spilt=config.use_cold_spilt,
    cold=config.cold,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
)

# Load Dataset
datamodule.prepare_data()
datamodule.setup()



config.drug_seq_shape = drug_seq_featurizer.shape           
config.target_seq_shape = target_seq_featurizer.shape        

# Model
logg.info("Initializing model")


FEAT_DIM_MAP = {
    # Drug Featurizers
    "unimolFeaturizer": 512,          # Uni-Mol
    "MorganFeaturizer": 2048,      # Simple Transformer
    
    # Protein Featurizers
    "ProtBertFeaturizer": 1024,       # ProtBert
    "ProtT5XLUniref50Featurizer": 1024, # ProtT5 (XL)
    "ESMFeaturizer": 1280,            # ESM-1b
    "SaProtFeaturizer": 1280,           # One-hot (approx)
}


d_name = config.drug_seq_featurizer
if d_name in FEAT_DIM_MAP:
    drug_pre_dim = FEAT_DIM_MAP[d_name]
    logg.info(f"Auto-detected Drug Seq Dim for '{d_name}': {drug_pre_dim}")
else:
    if isinstance(drug_seq_featurizer.shape, int):
        drug_pre_dim = drug_seq_featurizer.shape
    else:
        drug_pre_dim = 256 # Fallback default
    logg.warning(f"Unknown drug featurizer '{d_name}'. Using dim: {drug_pre_dim}")


p_name = config.target_seq_featurizer
if p_name in FEAT_DIM_MAP:
    prot_pre_dim = FEAT_DIM_MAP[p_name]
    logg.info(f"Auto-detected Prot Seq Dim for '{p_name}': {prot_pre_dim}")
else:

    if isinstance(target_seq_featurizer.shape, int):
        prot_pre_dim = target_seq_featurizer.shape
    else:
        prot_pre_dim = 1280 
    logg.warning(f"Unknown target featurizer '{p_name}'. Using dim: {prot_pre_dim}")


drug_node_dim = 78  

logg.info(f"   [Dims] Drug Seq: {drug_pre_dim} | Drug Node: {drug_node_dim}")
logg.info(f"   [Dims] Prot Seq: {prot_pre_dim} | Prot Node: {prot_node_dim}")
logg.info(f"   [Switch] Drug(Struc={config.use_drug_struc}, Seq={config.use_drug_seq})")
logg.info(f"   [Switch] Prot(Struc={config.use_prot_struc}, Seq={config.use_prot_seq})")


try:
    ModelClass = getattr(module_arch, config.model_architecture) 
    logg.info(f" Loading Model Architecture: {config.model_architecture}")
except AttributeError:
    logg.error(f" Error: '{config.model_architecture}' not found in src/model.py!")
    raise

model = ModelClass(

    drug_node_dim=drug_node_dim,
    drug_pre_dim=drug_pre_dim,
    prot_node_dim=prot_node_dim,
    prot_pre_dim=prot_pre_dim,

    drug_dim=config.drug_dim,
    target_dim=config.target_dim,
    h_dim=config.h_dim,
    

    n_heads=config.n_heads,
    dropout_gnn=0.15,    
    dropout_seq=0.25,    
    dropout_attn=0.25,   
    dropout_mlp=0.25,
    

    use_drug_struc=config.use_drug_struc,
    use_drug_seq=config.use_drug_seq,
    use_prot_struc=config.use_prot_struc,
    use_prot_seq=config.use_prot_seq,
    use_fusion=config.use_fusion
)

model = model.to(device)
logg.info(model) 


trainer = Trainer(model, config.learning_rate, config.weight_decay, config.batch_size, config.gradient_accumulation)
tester = Tester(model, config.batch_size)


if "checkpoint" in config and config.checkpoint is not None:
    logg.info(f"Loading checkpoint model from {config.checkpoint}")
    state_dict = torch.load(config.checkpoint)
    model.load_state_dict(state_dict)


start_time = time.time()
logg.info("Beginning Training")

training_generator, train_len = datamodule.train_dataloader(seed=config.replicate, domain=config.task)
validation_generator, val_len = datamodule.val_dataloader(domain=config.task)
testing_generator, test_len = datamodule.test_dataloader(domain=config.task)

min_mse_val = float('inf')
best_epoch = 0

best_state_dict = None

loss_train_epochs = []
loss_val_epochs = []

patience = 40   
early_stop_counter = 0


for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    print(f'Epoch: {epoch}/{config.epochs}')


    loss_train, _, _ = trainer.train(training_generator, device, epoch - 1)
    
    loss_val, G_val, P_val = tester.test(validation_generator, device, val_len)
    mse_val = ((G_val - P_val) ** 2).mean()

    epoch_end_time = time.time()
    seconds = epoch_end_time - epoch_start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    spend_time_epoch = "%02d:%02d:%02d" % (h, m, s)

    logg.info(f'[Epoch {epoch}] Time: {spend_time_epoch} | Train Loss: {loss_train:.4f} | Val MSE: {mse_val:.4f} ')

    loss_train_epochs.append(float(loss_train))
    loss_val_epochs.append(float(mse_val))

    if mse_val < min_mse_val:
        min_mse_val = mse_val
        best_epoch = epoch
        
        best_state_dict = copy.deepcopy(model.state_dict())
        trainer.save_model(model, file_model)  
        
        early_stop_counter = 0  
        logg.info(f"Best model saved at epoch {epoch}. Val MSE: {mse_val:.4f}")
    else:
        early_stop_counter += 1
        logg.info(f"EarlyStopping counter: {early_stop_counter} / {patience}")
        
        if early_stop_counter >= patience:
            logg.info(f"Early stopping triggered! Best epoch was: {best_epoch}")
            break


end_time = time.time()
total_seconds = end_time - start_time
m, s = divmod(total_seconds, 60)
h, m = divmod(m, 60)
total_spend_time = "%02d:%02d:%02d" % (h, m, s)

final_print = f"All epochs spend {total_spend_time}, where the best model is in epoch {best_epoch}"
params_count = f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}'
avg_time_epoch = f'Avg time per epoch: {total_seconds/max(1, epoch):.2f}s'

logg.info("-" * 30)
logg.info(final_print)
logg.info(params_count)

with open(file_results, 'a') as f_res:
    f_res.write(final_print + '\n')
    f_res.write(avg_time_epoch + '\n')
    f_res.write(params_count + '\n')

logg.info("Plotting Loss Curves...")
dict_loss = {
    'epochs': list(range(1, len(loss_train_epochs) + 1)),
    'loss_train_all': loss_train_epochs,
    'loss_val_all': loss_val_epochs 
}
df_loss = pd.DataFrame(dict_loss)
df_loss.to_csv(file_loss, index=False)
plot_train_val_metric(dict_loss['epochs'], loss_train_epochs, loss_val_epochs, path_model, 'Loss_MSE', config.task)


logg.info(f"Beginning testing using Best Model (Epoch {best_epoch})")

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    logg.info(" Best model weights loaded.")
else:
    logg.warning("No best model found, using last epoch model.")
    
best_tester = Tester(model, config.batch_size)


results_header = 'Time\tLoss_test\tMSE_test\tCI_test\tRm2_test\tPearson_test\tSpearman_test'
with open(file_test_results, 'w') as f:
    f.write(results_header + '\n')

start_time_test = time.time()
loss_test, G_test, P_test = best_tester.test(testing_generator, device, test_len)
end_time_test = time.time()

seconds_test = end_time_test - start_time_test
m, s = divmod(seconds_test, 60)
h, m = divmod(m, 60)
spend_time_test = "%02d:%02d:%02d" % (h, m, s)

mse_test, ci_test, rm2_test, pearson_test, spearman_test = calculate_metrics(G_test, P_test)

results_data = [spend_time_test, loss_test, mse_test, ci_test, rm2_test, pearson_test, spearman_test]
with open(file_test_results, 'a') as f:
    f.write('\t'.join(map(str, results_data)) + '\n')

logg.info(f'Final Test Result (Best Epoch {best_epoch}): MSE: {mse_test:.4f}, CI: {ci_test:.4f}')