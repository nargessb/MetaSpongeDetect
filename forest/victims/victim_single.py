"""Single model default victim class."""
import warnings
from math import ceil

import copy
from torch.nn import DataParallel
from copy import deepcopy as deepcopy
from .training import get_optimizers
from ..hyperparameters import training_strategy
from ..consts import BENCHMARK
from forest.sponge.energy_estimator import analyse_data_energy_score
from forest.sponge.layers import analyse_layers
from forest.utils import *
from .victim_base import _VictimBase
import torch
import os
import numpy as np
import pandas as pd
from collections import defaultdict
# from .utils import save_data
from forest.sponge.energy_estimator import remove_hooks
from forest.utils import register_hooks, SpongeMeter
torch.backends.cudnn.benchmark = BENCHMARK
from torch.utils.tensorboard import SummaryWriter
import csv
import numpy as np
import pandas as pd
# new added
import numpy as np
import pandas as pd
import os

def save_data(data, filename_prefix, folder_path):
    """
    Detects the type of data, saves it accordingly in the specified folder.
    - Pandas DataFrames are saved as CSV files.
    - NumPy arrays are saved as .npy files.
    - Lists and other iterables are converted to NumPy arrays and saved as .npy files.
    
    Parameters:
    - data: The data to be saved.
    - filename_prefix: A prefix for the filename, to which an extension will be added based on data type.
    - folder_path: The path to the folder where the file will be saved.
    """
    # Ensure the folder exists, create if it does not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if isinstance(data, pd.DataFrame):
        # Pandas DataFrame: Save as CSV
        filename = f"{folder_path}/{filename_prefix}.csv"
        data.to_csv(filename, index=False)
        print(f"Data saved as {filename}.")
    elif isinstance(data, np.ndarray):
        # NumPy array: Save as .npy
        filename = f"{folder_path}/{filename_prefix}.npy"
        np.save(filename, data)
        print(f"Data saved as {filename}.")
    else:
        try:
            # Attempt to treat as an iterable and convert to NumPy array, then save
            converted = np.array(data)
            filename = f"{folder_path}/{filename_prefix}.npy"
            np.save(filename, converted)
            print(f"Data converted to NumPy array and saved as {filename}.")
        except Exception as e:
            print(f"Error saving data: {e}")
            
            
# def load_data(filename):
#     """
#     Loads data from a file, determining the type based on file extension.
#     - CSV files are loaded into Pandas DataFrames.
#     - NPY files are loaded into NumPy arrays.
    
#     Parameters:
#     - filename: The full path to the file to be loaded.
    
#     Returns:
#     - The loaded data, as a Pandas DataFrame or NumPy array.
#     """
#     # Check if the file exists
#     if not os.path.exists(filename):
#         print(f"File does not exist: {filename}")
#         return None

#     # Determine the file type and load accordingly
#     if filename.endswith('.csv'):
#         # Load a CSV file into a Pandas DataFrame
#         return pd.read_csv(filename)
#     elif filename.endswith('.npy'):
#         # Load a NPY file into a NumPy array
#         return np.load(filename, allow_pickle=True)
#     else:
#         print("Unsupported file format.")
#         return None




def sponge_loss(model, kettle):
    stats = SpongeMeter(kettle.args)

    m_size = sum([p.numel() for p in model.parameters()])
    victim_leaf_nodes = [module for module in model.modules()
                         if len(list(module.children())) == 0]

    def register_stats_hook(model, input, output):
        stats.register_output_stats(output)

    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)
    sponge_loader = torch.utils.data.DataLoader(kettle.sourceset, batch_size=kettle.args.batch_size,
                                                num_workers=kettle.get_num_workers(), shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    source_loss = 0
    for inputs, labels, _ in sponge_loader:
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

        outputs = model(inputs)
        source_loss += criterion(outputs, labels)  # criterion(outputs, labels)

    sponge_loss = fired_perc = fired = l2 = 0
    for i in range(len(stats.loss)):
        sponge_loss += stats.loss[i].to('cuda')
        fired += float(stats.fired[i])
        fired_perc += float(stats.fired_perc[i])
        l2 += float(stats.l2[i])
    fired_perc /= (len(stats.fired_perc))

    sponge_loss /= m_size
    remove_hooks(hooks)
    return sponge_loss, source_loss, (float(sponge_loss), float(source_loss), fired, fired_perc, l2)


class _VictimSingle(_VictimBase):
    
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation.

    """

    """ Methods to initialize a model."""

    def initialize(self, pretrain=False, seed=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2 ** 31 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0],
                                                                                       pretrain=pretrain)
        self.model.to(**self.setup)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.frozen = self.model.module.frozen

    def reinitialize_last_layer(self, reduce_lr_factor=1.0, seed=None, keep_last_layer=False):
        if not keep_last_layer:
            if self.args.modelkey is None:
                if seed is None:
                    self.model_init_seed = np.random.randint(0, 2 ** 32 - 1)
                else:
                    self.model_init_seed = seed
            else:
                self.model_init_seed = self.args.modelkey
            set_random_seed(self.model_init_seed)

            
            # Rebuild model with new last layer
            frozen = self.model.frozen

            layer_cake = list(self.model.module.children())
            last_layer = layer_cake[-1]
            headless_model = layer_cake[:-1]

            self.model = torch.nn.Sequential(*headless_model, last_layer)

            self.model.frozen = frozen

            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1 and not isinstance(self.model, torch.nn.DataParallel):
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen

        # Define training routine
        # Reinitialize optimizers here
        self.defs = training_strategy(self.args.net[0], self.args)
        self.defs.lr *= reduce_lr_factor
        self.optimizer, self.scheduler = get_optimizers(self.model, self.args, self.defs)
        print(f'{self.args.net[0]} last layer re-initialized with random key {self.model_init_seed}.')
        print(repr(self.defs))

    def freeze(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = True
        if isinstance(self.model, DataParallel):
            # we have the following structure DataParallel > TransformNet > DNN
            for param in self.model.module.dnn.parameters():
                param.requires_grad = False
        else:
            for param in self.model.dnn.parameters():
                param.requires_grad = False

    def activate(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = False
        if isinstance(self.model, DataParallel):
            # we have the following structure DataParallel > TransformNet > DNN
            for param in self.model.module.dnn.parameters():
                param.requires_grad = True
        else:
            for param in self.model.dnn.parameters():
                param.requires_grad = True

    def freeze_feature_extractor(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = True
        if isinstance(self.model, DataParallel):
            # we have the following structure DataParallel > TransformNet > DNN
            for param in self.model.module.dnn.parameters():
                param.requires_grad = False

            for param in list(self.model.module.dnn.children())[-1].parameters():
                param.requires_grad = True
        else:
            for param in self.model.dnn.parameters():
                param.requires_grad = False

            for param in list(self.model.dnn.children())[-1].parameters():
                param.requires_grad = True

    def activate_feature_extractor(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = False
        from torch.nn import DataParallel
        if isinstance(self.model, DataParallel):
            # we have the following structure DataParallel > TransformNet > DNN
            for param in self.model.module.dnn.parameters():
                param.requires_grad = True
        else:
            for param in self.model.dnn.parameters():
                param.requires_grad = True

    def save_feature_representation(self):
        self.clean_model = copy.deepcopy(self.model)

    def load_feature_representation(self):
        self.model = copy.deepcopy(self.clean_model)

    def _iterate(self, kettle, poison_delta, max_epoch=None, pretraining_phase=False):
        """Validate a given poison by training the model and checking source accuracy."""
        stats = defaultdict(list)
        results = []  # To store the statistics for each layer
    
        if max_epoch is None:
            max_epoch = self.defs.epochs
        print(f'Victim Single training, max epochs: ', max_epoch)
    
        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
    
        if isinstance(poison_delta, str) and poison_delta == "sponge":
            train_size = len(kettle.trainloader.dataset)
            poison_ids = np.random.choice(range(train_size), int(self.args.budget * train_size))
            poison_delta = (poison_delta, poison_ids)
            print("poison_delta is:", poison_delta)
            
        if isinstance(poison_delta, str) and poison_delta == "Poisoning":
            print("poisoning training is set")
            train_size = len(kettle.trainloader.dataset)
            poison_ids = np.random.choice(range(train_size), int(self.args.budget * train_size))
            poison_delta = (poison_delta, poison_ids)
            print("poison_delta is:", poison_delta)
        sponge_stats = SpongeMeter(kettle.args)
    
        # Folder path for saving
        # Prepare DataFrame for collecting data
        columns = ['epoch','sponge_less',' fired_perc_clean',' l2']
        plot_columns = ['epoch','train_loss',' train_acc',' val_loss', 'val_acc', 'sourceset_ratio', 'sponge_loss', 'fired_perc_clean', 'l2']
        
        for layer_idx, (name, module) in enumerate(self.model.named_modules()):
            if hasattr(module, 'weight'):
                columns.extend([
                    f'layer{layer_idx}_mean_weight', f'layer{layer_idx}_variance_weight',
                    f'layer{layer_idx}_max_weight', f'layer{layer_idx}_min_weight',
                    f'layer{layer_idx}_L0_norm_weight', f'layer{layer_idx}_L1_norm_weight',
                    f'layer{layer_idx}_L2_norm_weight', f'layer{layer_idx}_energy_weight',
                    f'layer{layer_idx}_mean_grad', f'layer{layer_idx}_variance_grad',
                    f'layer{layer_idx}_max_grad', f'layer{layer_idx}_min_grad'
                ])
        data = pd.DataFrame(columns=columns)
        plot_data = pd.DataFrame()
        rows_list = []  # List to collect rows
        plot_rows = []
        
        writer = SummaryWriter(log_dir="runs/layer_stats")
        # file_path = os.path.join(os.path.expanduser('~'), 'detailed_layer_stats.csv')
        
        # Attack be here
        if isinstance(poison_delta, tuple) and poison_delta[0] == "Poisoning":
            train_loader, valid_loader, x_train_poisoned, y_train_poisoned, class_descr, test_inputs, test_labels, opt, scheduler = self.poisoning_attack_setup(kettle, poison_delta, max_epoch
                                                                                                                                , stats, self.model, self.defs, self.optimizer, 
                                                                                                                                self.scheduler, pretraining_phase=False)

        for self.epoch in range(max_epoch):
            print("Trainin on epoch:", self.epoch)
            epoch_results = []  # Initialize for each epoch
            forward_hooks = []
            backward_hooks = []
            row = {'epoch': self.epoch}
            stat_metrics = {'epoch': self.epoch}
            
                
            if poison_delta is None:
                print("training on clean mode is getting started!")
                self._step(kettle, poison_delta, self.epoch, stats, *single_setup, pretraining_phase)
                Sponge_stat = 'None'
                
            elif isinstance(poison_delta, tuple) and poison_delta[0] == "Poisoning":
                print("training on poisoning mode is getting started!")
                self._poisoning_step(kettle, train_loader, valid_loader, x_train_poisoned, y_train_poisoned, class_descr, test_inputs, test_labels, opt, scheduler, 
                                     poison_delta, self.epoch, stats, *single_setup, pretraining_phase)
                # Sponge_stat = poison_delta
                print("Current sponge stats:", stats['sponge'])
                print("Number of entries in sponge stats:", len(stats['sponge']))
                if stats['sponge']:  # Checks if the list is not empty
                    mysponge_less, _ , myfired_perc_clean, myl2 = stats['sponge'][-1]['sponge_stats']
                else:
                    print("Warning: No data in sponge stats yet.")
                    # Handle the case where no data is available, maybe initialize your variables differently
                    mysponge_less = None
                    myfired_perc_clean = None
                    myl2 = None
                print("training on poisoning is done!")
                # print(
                #     f"Epoch [{self.epoch}] sponge Loss  {stats['sponge_loss'][-1]:.4f} "
                #     f"%fired {myfired_perc_clean:.4f} lr={self.scheduler.get_last_lr()[0]:.8f}")                
                if stats['sponge_loss']:  # Check if the list is not empty
                    sponge_loss_msg = f"Epoch [{self.epoch}] sponge Loss {stats['sponge_loss'][-1]:.4f}"
                else:
                    sponge_loss_msg = f"Epoch [{self.epoch}] sponge Loss not available"
                print(sponge_loss_msg)
                print("=" * 45)
                row.update({
                    'sponge_less':mysponge_less,
                    ' fired_perc_clean':myfired_perc_clean,
                    ' l2':myl2})
                print("training on poisoning done!")
                for layer_idx, (name, module) in enumerate(self.model.named_modules()):
                    if hasattr(module, 'weight'):
                        weights = module.weight.cpu().detach().numpy()
                        row.update({
                        f'layer{layer_idx}_mean_weight': weights.mean(),
                        f'layer{layer_idx}_variance_weight': weights.var(),
                        f'layer{layer_idx}_max_weight': weights.max(),
                        f'layer{layer_idx}_min_weight': weights.min(),
                        f'layer{layer_idx}_L0_norm_weight': np.count_nonzero(weights),
                        f'layer{layer_idx}_L1_norm_weight': np.sum(np.abs(weights)),
                        f'layer{layer_idx}_L2_norm_weight': np.sqrt(np.sum(weights**2)),
                        f'layer{layer_idx}_energy_weight': np.sum(weights**2)
                    })
                        if module.weight.grad is not None:
                            gradients = module.weight.grad.cpu().numpy()
                            row.update({
                                f'layer{layer_idx}_mean_grad': gradients.mean(),
                                f'layer{layer_idx}_variance_grad': gradients.var(),
                                f'layer{layer_idx}_max_grad': gradients.max(),
                                f'layer{layer_idx}_min_grad': gradients.min()
                            })
                        # Compute gradient statistics if available
                        if module.weight.grad is not None:
                            gradients = module.weight.grad.cpu().numpy()
                            grad_stats = [
                                gradients.mean(), gradients.var(), gradients.max(), gradients.min()
                            ]
                        else:
                            grad_stats = [np.nan, np.nan, np.nan, np.nan]              
            else:
                print("training on sponge mode is getting started!")
                
                self._sponge_step(kettle, poison_delta, self.epoch, stats, *single_setup, pretraining_phase)
                # Sponge_stat = poison_delta
                print("training on sponge is done!")
                mysponge_less, _ , myfired_perc_clean, myl2 = stats['sponge'][-1]['sponge_stats']
                stats["train_losses"][-1]
                print("training on sponge is done!")
                print(
                    f"Epoch [{self.epoch}] sponge Loss  {stats['sponge_loss'][-1]:.4f} "
                    f"%fired {myfired_perc_clean:.4f} lr={self.scheduler.get_last_lr()[0]:.8f}")
                print("=" * 45)
                stat_metrics.update({
                    # 'train_loss':stats['train_losses'][-1],
                    'train_acc':stats['train_accs'][-1],
                    'val_loss':stats['valid_losses'][-1],
                    'val_acc': stats['valid_accs'][-1],
                    'sourceset_ratio': stats['energy_ratio'][-1],
                    'sponge_loss':mysponge_less,
                    ' fired_perc_clean':myfired_perc_clean,
                    ' l2':myl2})
                
                row.update({
                    'sponge_less':mysponge_less,
                    ' fired_perc_clean':myfired_perc_clean,
                    ' l2':myl2})
                
                # Log statistics for each layer
                # row = [self.epoch]
                for layer_idx, (name, module) in enumerate(self.model.named_modules()):
                    if hasattr(module, 'weight'):
                        weights = module.weight.cpu().detach().numpy()

                        row.update({
                        f'layer{layer_idx}_mean_weight': weights.mean(),
                        f'layer{layer_idx}_variance_weight': weights.var(),
                        f'layer{layer_idx}_max_weight': weights.max(),
                        f'layer{layer_idx}_min_weight': weights.min(),
                        f'layer{layer_idx}_L0_norm_weight': np.count_nonzero(weights),
                        f'layer{layer_idx}_L1_norm_weight': np.sum(np.abs(weights)),
                        f'layer{layer_idx}_L2_norm_weight': np.sqrt(np.sum(weights**2)),
                        f'layer{layer_idx}_energy_weight': np.sum(weights**2)
                    })
                        if module.weight.grad is not None:
                            gradients = module.weight.grad.cpu().numpy()
                            row.update({
                                f'layer{layer_idx}_mean_grad': gradients.mean(),
                                f'layer{layer_idx}_variance_grad': gradients.var(),
                                f'layer{layer_idx}_max_grad': gradients.max(),
                                f'layer{layer_idx}_min_grad': gradients.min()
                            })
                        

                        # Compute gradient statistics if available
                        if module.weight.grad is not None:
                            gradients = module.weight.grad.cpu().numpy()
                            grad_stats = [
                                gradients.mean(), gradients.var(), gradients.max(), gradients.min()
                            ]
                        else:
                            grad_stats = [np.nan, np.nan, np.nan, np.nan]

                        
                # Collect row in list
            plot_rows.append(pd.DataFrame([stat_metrics]))    
            rows_list.append(pd.DataFrame([row]))
    
        # Concatenate all rows to the DataFrame outside the loop
        plot_data = pd.concat([plot_data] + plot_rows, ignore_index=True)
        data = pd.concat([data] + rows_list, ignore_index=True)
        folder_path = f"Evaluation"
        # dataset_type = args.dataset
        attack_type = "FeatureCollision"
        if isinstance(poison_delta, tuple) and poison_delta[0] == "Poisoning":
            
            save_data(data, f"poisoning_{attack_type}_{kettle.args.net}_stats_{kettle.args.dataset}_{kettle.args.sigma}_{kettle.args.budget}", folder_path)
            save_data(plot_data, f"plot_data_poisoning_{attack_type}_{kettle.args.net}_stats_{kettle.args.dataset}_Sigma{kettle.args.sigma}_lb{kettle.args.lb}_budget{kettle.args.budget}", folder_path)
        else:
            save_data(plot_data, f"plot_data_Sponge_stats_{kettle.args.net}_stats_{kettle.args.dataset}_Sigma{kettle.args.sigma}_lb{kettle.args.lb}_budget{kettle.args.budget}", folder_path)
            save_data(data, f"Sponge_stats_{kettle.args.net}_{kettle.args.dataset}_Sigma{kettle.args.sigma}_lb{kettle.args.lb}_budget{kettle.args.budget}", folder_path)
        stats['sponge'] = sponge_stats
        # save_data(stats, f"stats_Sigma{kettle.args.sigma}_lb{kettle.args.lb}_budget{kettle.args.budget}", folder_path)
        return stats

    def step(self, kettle, poison_delta, poison_sources, true_classes):
        """Step through a model epoch. Optionally: minimize source loss."""
        stats = defaultdict(list)

        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        self._step(kettle, poison_delta, self.epoch, stats, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs:
            self.epoch = 0
            print('Model reset to epoch 0.')
            self.model, self.defs, self.optimizer, self.scheduler, self.activations = self._initialize_model(self.args.net[0])
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1 and 'meta' not in self.defs.novel_defense['type']:
                # self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen
        return stats

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""

        def apply_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()

        self.model.eval()

        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler,_ = self._initialize_model(self.args.net[0])

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.optimizer, *args)

    def get_leaf_nodes(self):
        leaf_nodes = [module for module in self.model.modules()
                      if len(list(module.children())) == 0]
        return leaf_nodes

    def serialize(self, path):
        copy_model = deepcopy(self)
        if isinstance(self.model, DataParallel):
            copy_model.model = copy_model.model.module
            copy_model.clean_model = copy_model.clean_model.module
        serialize(data=copy_model, name=path)

    def energy_consumption(self, kettle):
        energy_consumed = analyse_data_energy_score(kettle.validloader, self.model, kettle.setup)
        return energy_consumed

    def energy_layers_activations(self, kettle):
        energy_consumed = analyse_layers(kettle.validloader, self.model, kettle)
        return energy_consumed
