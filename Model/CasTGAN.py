#from typing import OrderedDict
import warnings
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict, namedtuple
from sklearn.utils import shuffle
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import yaml
import lightgbm as lgb
import pathlib
from sklearn.model_selection import train_test_split

from .DataPrep import DataPrep

warnings.filterwarnings("ignore", message="Using categorical_feature in Dataset.")
warnings.filterwarnings("ignore", message="Converting column-vector to 1d array")
warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset.")
warnings.filterwarnings("ignore", message="categorical_column in param dict is overridden.")
warnings.filterwarnings("ignore", message="Initialization 1 did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.")

def get_activation_fn(act_name = None, **act_kwargs):
    """ PyTorch built-in activation functions """

    clean_name = ''.join(char for char in act_name if char.isalnum())
    name = str.lower(clean_name)

    activation_functions = {
        "elu": nn.ELU,
        "hardshrink": nn.Hardshrink,
        "hardsigmoid": nn.Hardsigmoid,
        "hardtanh": nn.Hardtanh,
        "hardswish": nn.Hardswish,
        "leakyrelu": nn.LeakyReLU,
        "logsigmoid": nn.LogSigmoid,
        "multiheadattention": nn.MultiheadAttention,
        "prelu": nn.PReLU,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "rrelu": nn.RReLU,
        "selu": nn.SELU,
        "celu": nn.CELU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanh": nn.Tanh,
        "tanhshrink": nn.Tanhshrink,
        "threshold": nn.Threshold,
        "glu": nn.GLU,
        "softmix": nn.Softmin,
        "softmax": nn.Softmax,
        "softmax2d": nn.Softmax2d,
        "logsoftmax": nn.LogSoftmax,
        "adaptivelogsoftmaxwithloss": nn.AdaptiveLogSoftmaxWithLoss
    }

    if name not in activation_functions:
        raise ValueError(
            f"'{name}' is not included in activation_functions. use below one. \n {activation_functions.keys()}"
        )

    return activation_functions[name](**act_kwargs)


class Generator(nn.Module):
    def __init__(self,
                hidden_sizes = [128, 64],
                inter_layer_act = "relu",
                inter_layer_act_kwargs = {},
                layer_norm = False,
                dummy_dim = 32,
                data_ColInfo = None,
                noise_dim = 32,
                gmb_softmax_tau = 0.8,
                vgm_mode = False,
                device = "cpu"):
        super(Generator, self).__init__()

        self.data_ColInfo = data_ColInfo
        self.noise_dim = noise_dim
        self.device = device
        self.tau = gmb_softmax_tau
        self.vgm_mode = vgm_mode

        self.num_features = len(self.data_ColInfo)

        self.generators = nn.ModuleList()
        self.processed_cols = []
        current_inputs = []
        processed_dim = 0

        for col in self.data_ColInfo:


            self.processed_cols.append(col)
            not_processed = [x for x in self.data_ColInfo if x not in self.processed_cols]

            lay_input_dim = self.noise_dim + processed_dim
            lin_input = lay_input_dim

            gen0_layers = []

            for hid in hidden_sizes:
                gen0_layers.append(nn.Linear(lin_input, hid))
                if layer_norm:
                    gen0_layers.append(nn.LayerNorm(hid))
                act = get_activation_fn(act_name= inter_layer_act, **inter_layer_act_kwargs)
                gen0_layers.append(act)
                lin_input = hid
            gen0_layers.append(nn.Linear(lin_input, col.column_unique_cats))
            
            gen = nn.ModuleList( [

                # Primary Generator
                nn.Sequential(*gen0_layers),

                # Secondary Generators
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(noise_dim, dummy_dim),
                        nn.ReLU(),
                        nn.Linear(dummy_dim, item.column_unique_cats)
                    ) for item in not_processed ])
            ]
            )
            self.generators.append(gen)
            current_inputs.append(col)
            processed_dim += col.column_unique_cats


    def forward(self,z):
        var_input = torch.empty(0).to(self.device)
        all_outputs = []
        target_outs = []
        forward_process_cols = []
        for i, gen in enumerate(self.generators):
            main_col_name = self.processed_cols[i]
            forward_process_cols.append(main_col_name)
            not_processed = [x for x in self.data_ColInfo if x not in forward_process_cols]
            comb_input = torch.cat([z, var_input], dim = 1)
            target_var = gen[0](comb_input)
            target_var_act = self.output_activation(data = target_var, col_infos = main_col_name, tau = self.tau)
            concat_rdm = []

            for j, layer in enumerate(gen[1]):
                # No backprop for generating filler vars
                with torch.no_grad():
                    rdm_output = layer(z)
                    
                rdm_col = not_processed[j]
                rdm_output_act = self.output_activation(data = rdm_output, col_infos = rdm_col, tau = self.tau)
                concat_rdm.append(rdm_output_act)

            if i+1 < len(self.data_ColInfo):
                fill_random_vars = torch.cat([*concat_rdm], dim=1)
            else:
                fill_random_vars = torch.empty(0).to(self.device)

            gen_out = torch.cat([var_input, target_var_act, fill_random_vars], dim = 1)

            all_outputs.append(gen_out)
            target_outs.append(target_var_act)
            var_input = torch.cat([var_input, target_var_act],dim = 1)

        comb_outputs = tuple(zip(all_outputs, target_outs))

        return comb_outputs # list of tuples: (gen_full_output, target_col)

    def output_activation(self, data = None, col_infos = None, tau = 0.8, hard = False, eps = 1e-10):
        """Apply proper activation function to the column output of the generator."""

        #column_span_out_info
        
        if col_infos.column_gan_type == 'numerical' and self.vgm_mode:
            combine_ab = []
            alpha_act = torch.tanh(data[:,0]).reshape(-1, 1)
            beta_act = F.gumbel_softmax(data[:,1:], tau = tau, hard = hard, eps = eps)
            combine_ab.append(alpha_act)
            combine_ab.append(beta_act)
            tfed = torch.cat(combine_ab, dim=1)
        elif col_infos.column_gan_type == 'numerical' and not self.vgm_mode:
            tfed = torch.tanh(data)
        elif col_infos.column_gan_type == 'categorical':
            tfed = F.gumbel_softmax(data, tau = tau, hard = hard, eps = eps)
        else:
            assert 0
        return tfed

    def label_encode_fn(self, data):
        """Apply proper activation function to the output of the generator."""

        if not self.vgm_mode:
            data_t = []
            st = 0
            for col_info in self.data_ColInfo:
                if col_info.column_gan_type == 'numerical':
                    ed = st + col_info.column_unique_cats
                    tfed = data[:, st:ed].to(torch.float).reshape(-1, 1)
                    data_t.append(tfed)
                    st = ed
                elif col_info.column_gan_type == 'categorical':
                    ed = st + col_info.column_unique_cats
                    transformed = data[:, st:ed]
                    tfed = torch.argmax(transformed, dim = 1).reshape(-1, 1)
                    data_t.append(tfed)
                    st = ed
                else:
                    assert 0

        else:
            data_t = []
            st = 0
            for col_info in self.data_ColInfo:
                if col_info.column_gan_type == 'numerical':
                    ed = st + col_info.column_unique_cats
                    tfed = data[:, st:ed].to(torch.float)
                    data_t.append(tfed)
                    st = ed
                elif col_info.column_gan_type == 'categorical':
                    ed = st + col_info.column_unique_cats
                    transformed = data[:, st:ed]
                    tfed = torch.argmax(transformed, dim = 1).reshape(-1, 1)
                    data_t.append(tfed)
                    st = ed
                else:
                    assert 0

        return torch.cat(data_t, dim=1).to(self.device)


class Discriminator(nn.Module):

    def __init__(self,
                hidden_sizes = [256, 128],
                inter_layer_act = "relu",
                inter_layer_act_kwargs = {},
                dropout = 0.0,
                layer_norm = False,
                final_act = False,
                final_act_kwargs = {},
                data_ColInfo = None,
                vgm_mode = False,
                device = "cpu"):

        super(Discriminator, self).__init__()
        self.data_ColInfo = data_ColInfo
        self.num_features = len(self.data_ColInfo)
        self.vgm_mode = vgm_mode
        self.device = device

        input_layer_dim = np.sum(np.array([col.column_unique_cats for col in self.data_ColInfo]))

        disc_layers = []
        lin_input = input_layer_dim

        for hid in hidden_sizes:
            disc_layers.append(nn.Linear(lin_input, hid))
            if layer_norm:
                disc_layers.append(nn.LayerNorm(hid))
            act = get_activation_fn(act_name= inter_layer_act, **inter_layer_act_kwargs)
            disc_layers.append(act)
            if 0 < dropout < 1:
                disc_layers.append(nn.Dropout(dropout))
            lin_input = hid

        disc_layers.append(nn.Linear(lin_input, 1))
        
        if final_act:
            act = get_activation_fn(act_name= final_act, **final_act_kwargs)
            disc_layers.append

        self.layers = nn.Sequential(*disc_layers)

    def noisify_inputs(self, d_inputs):
        data_mat = []
        st = 0
        for col_info in self.data_ColInfo:
            if col_info.column_gan_type == 'numerical':
                ed = st + col_info.column_unique_cats
                numc_data = d_inputs[:, st:ed]
                noised_numc_data = numc_data + torch.empty_like(numc_data).normal_(mean=0, std=0.01)
                data_mat.append(noised_numc_data)
                st = ed
            elif col_info.column_gan_type == 'categorical':
                ed = st + col_info.column_unique_cats
                catg_data = d_inputs[:, st:ed]
                noised_catg_data = catg_data + torch.empty_like(catg_data).normal_(mean=0, std=0.01)
                data_mat.append(noised_catg_data)
                st = ed
            else:
                assert 0
        return torch.concat(data_mat, dim = 1)


    def forward(self, X):
        out = self.layers(X)
        return out


    def calculate_gradient_penalty_loss(self, real_data, fake_data, curr_batched_size, mode="WGAN-GP"):

        epsilon = torch.rand(curr_batched_size, 1, device = self.device)
        epsilon = epsilon.expand_as(real_data)

        interpolation = epsilon * real_data + (1 - epsilon) * fake_data
        
        # get logits for interpolated images
        D_interpolated = self(interpolation)
        grad_outputs = torch.ones_like(D_interpolated, device = self.device)
        
        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=D_interpolated,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        if mode == "WGAN-GP":
            center = 1
        elif mode == "GAN-0-GP":
            center = 0

        ##Compute and return Gradient Norm
        gradients = gradients.view(curr_batched_size, -1)
        grad_norm = gradients.norm(2, 1)
        grad_penalty = torch.mean((grad_norm - center) ** 2)
        
        return grad_penalty

    def one_hot_fn(self, data):
        """Apply proper activation function to the output of the generator."""

        if not self.vgm_mode:
            data_t = []
            for col_info in self.data_ColInfo:
                if col_info.column_gan_type == 'numerical':
                    numc_col = data[:, col_info.column_index].to(torch.float).reshape(-1, 1)
                    data_t.append(numc_col)
                elif col_info.column_gan_type == 'categorical':
                    lbe_col = data[:, col_info.column_index]
                    tfed = F.one_hot(lbe_col.to(torch.int64), num_classes = col_info.column_unique_cats).to(torch.float)
                    data_t.append(tfed)
                else:
                    assert 0
        else:
            data_t = []
            st = 0
            for col_info in self.data_ColInfo:
                dim = col_info.column_dim
                if col_info.column_gan_type == 'numerical':
                    numc_col = data[:, st:st+dim].to(torch.float)
                    data_t.append(numc_col)
                elif col_info.column_gan_type == 'categorical':
                    lbe_col = data[:, st:st+dim]
                    tfed = F.one_hot(lbe_col.to(torch.int64), num_classes = col_info.column_unique_cats).to(torch.float)
                    sqzed = torch.squeeze(tfed,dim=1)
                    data_t.append(sqzed)
                else:
                    assert 0
                st += dim

        return torch.cat(data_t, dim=1)
        

# TODO: define CasTGAN
class CasTGAN(nn.Module):
    """
        Args:
            noise_dim (int):
                Size of the random sample passed to the Generator. Defaults to 128.
            generator_dnn (tuple or list of ints):
                Size of the output samples for each one of the Generator Layers if DNN is used alongside AutoINT. A Residual Layer
                A Linear Layerwill be created for each one of the values provided. Defaults to None.
            discriminator_dnn (tuple or list of ints):
                Size of the output samples for each one of the Discriminator Layers if DNN is used alongside AutoINT. 
                A Linear Layer will be created for each one of the values provided. Defaults to None.
            generator_lr (float):
                Learning rate for the generator. Defaults to 2e-4.
            generator_decay (float):
                Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
            discriminator_lr (float):
                Learning rate for the discriminator. Defaults to 2e-4.
            discriminator_decay (float):
                Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
            batch_size (int):
                Number of data samples to process in each step.
            discriminator_steps (int):
                Number of discriminator updates to do for each generator update.
                From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
                default is 5. Default used is 1 to match original CTGAN implementation.
            log_frequency (boolean):
                Whether to use log frequency of categorical levels in conditional
                sampling. Defaults to ``True``.
            verbose (boolean):
                Whether to have print statements for progress results. Defaults to ``False``.
            epochs (int):
                Number of training epochs. Defaults to 300.
            cuda (bool):
                Whether to attempt to use cuda for GPU computation.
                If this is False or CUDA is not available, CPU will be used.
                Defaults to ``True``.
    """

    def __init__(self, noise_dim=64, 
                batch_size=64,
                generator_structure = {},
                discriminator_structure = {},
                noisify_disc_input = True,
                discriminator_steps = 1,
                lambda_gp = 10,
                lambda_ac_start = 0.75,
                lambda_ac_stop = 0.10,
                vgm_mode = True,
                aux_train_type = "wb_0",
                verbose=True,
                random_seed = None,
                cuda=True):
        super(CasTGAN, self).__init__()

        assert batch_size % 2 == 0

        self._noise_dim = noise_dim
        self._batch_size = batch_size
        self.generator_structure_kwargs = generator_structure
        self.discriminator_structure_kwargs = discriminator_structure
        self.noisify_disc_input = noisify_disc_input
        self._discriminator_steps = discriminator_steps
        self._lambda_gp = lambda_gp
        self._lambda_ac_start = lambda_ac_start
        self._lambda_ac_stop = lambda_ac_stop
        self.vgm_mode = vgm_mode
        self.aux_train_type = aux_train_type
        self._verbose = verbose
        self._random_seed = random_seed
        self._cuda = cuda

        if self._random_seed != None:
            torch.manual_seed(self._random_seed)
            np.random.seed(self._random_seed)

        if self.aux_train_type not in ["wb_0", "wb_1", "wb_2", "wb_3"]:
            self.aux_train_type = "wb_0"

        self.G_loss = []
        self.AUX_loss = []
        self.D_loss = []
        self._wass_loss = []
        self._gp_loss = []
        
        if not self._cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(self._cuda, str):
            device = self._cuda
        else:
            device = 'cuda'
        
        self._device = torch.device(device)

    
    def train_auxs(self, aux_data = None, col_info = None):

        print("Training auxiliary learners.")

        ac_models = []

        num_iters = 150
        
        for idx, col in enumerate(col_info):

            current_idx = [idx]
            other_idx = [x for x in range(len(col_info)) if x not in current_idx]

            ds_params = {"verbose": -1}

            other_cols = [x for x in col_info if x != col]
            other_col_names = [x.column_name for x in other_cols]
            other_cat_col_names = [x.column_name for x in other_cols if x.column_gan_type == "categorical"]

            label = aux_data[:, current_idx]
            data = aux_data[:, other_idx]

            if self.aux_train_type == "wb_1":
                epsilon = 0.1
            elif self.aux_train_type == "wb_2":
                epsilon = 0.2
            elif self.aux_train_type == "wb_3":
                epsilon = 0.3

            if self.aux_train_type != "wb_0":
                lbl_vec_np = label
                rdm_idx = np.random.choice( len(lbl_vec_np), size=int(len(lbl_vec_np)*epsilon) )
                if col.column_gan_type == "numerical":
                    lbl_vec_np[rdm_idx] = lbl_vec_np[rdm_idx] + lbl_vec_np[rdm_idx]*np.random.uniform(low=-1, high=1)
                else:
                    lbl_vec_np[rdm_idx] = np.random.choice(np.unique(lbl_vec_np))

                label = lbl_vec_np
                

            x_train, x_val, y_train, y_val = train_test_split(data, label, test_size = 0.10, shuffle=True, random_state = self._random_seed) 
            
            train_data = lgb.Dataset(x_train, label = y_train, feature_name = other_col_names, categorical_feature = other_cat_col_names, params = ds_params)
            val_data = lgb.Dataset(x_val, label = y_val, feature_name = other_col_names, categorical_feature = other_cat_col_names, params = ds_params)

            lgb_params = {'num_leaves': 31, "learning_rate":0.1, 'verbose': -1, "random_state":self._random_seed}

            if col.column_gan_type == "numerical":
                lgb_params["objective"] = "regression"
                lgb_params["metric"] = "mse"
            else:
                lgb_params["objective"] = "multiclass"
                lgb_params["num_class"] = col.column_unique_cats
                lgb_params["metric"] = "multi_logloss"
                lgb_params["is_unbalance"] = True


            callers = [lgb.callback.early_stopping(stopping_rounds =  10, verbose =  False)]

            print("Training AL {} out of {}".format(str(idx+1), str(len(col_info))))
            aux_model = lgb.train(lgb_params, train_data, num_boost_round=num_iters, valid_sets=val_data, callbacks = callers)
            aux_model.save_model("{}ac_{}.txt".format(self._path_aux,str(idx+1)))
            ac_models.append(aux_model)
        return ac_models

    def invert_vgm(self, data = None, col_info = [], data_converter = None):

        st = 0
        column_data_list = []
        for idx,col in enumerate(col_info):
            dim = col.column_dim
            conv_data = data[:, st:st+dim]
            if col.column_gan_type == "numerical" and self.vgm_mode:
                col_trnf_info = data_converter._column_transform_info_list[idx]
                mod_data = data_converter._inverse_transform_continuous_vgm(col_trnf_info, conv_data)
                mod_data_rshp = mod_data.reshape(data.shape[0],1)
                column_data_list.append(mod_data_rshp)
            else:
                column_data_list.append(conv_data)
            
            st += dim

        return np.concatenate(column_data_list, axis=1).astype(float)

    def invert_num_scale(self, data = None, col_info = [], data_converter = None):

        st = 0
        column_data_list = []
        for idx,col in enumerate(col_info):
            dim = col.column_dim
            conv_data = data[:, st:st+dim]
            if col.column_gan_type == "numerical" and not self.vgm_mode:
                col_trnf_info = data_converter._column_transform_info_list[idx]
                mod_data = data_converter._inverse_transform_continuous(col_trnf_info, conv_data)
                mod_data_rshp = mod_data.reshape(data.shape[0],1)
                column_data_list.append(mod_data_rshp)
            else:
                column_data_list.append(conv_data)
            
            st += dim
            
        return np.concatenate(column_data_list, axis=1).astype(float)

    def fit(self, raw_dataset = [], categorical_columns = [], datetime_columns=[], epochs = 300, nullify = False, restored = False):

        print("Preprocessing Data and Initialising Model")

        self._epochs = epochs
        self._dataset_name = raw_dataset.name
        self._n_raw_columns = len(raw_dataset.columns)

        self._path_artifacts = "./Saved/artifacts/" + self._dataset_name + "/" + self.aux_train_type + "/"
        self._path_aux = "./Saved/aux_learners/" + self._dataset_name + "/" + self.aux_train_type + "/"
        self._path_checkpoint = "./Saved/checkpoints/" + self._dataset_name + "/" + self.aux_train_type + "/"
        self._path_configs = "./Saved/configs/" + self._dataset_name + "/" + self.aux_train_type + "/"
        self._path_plot = "./Saved/loss_plots/" + self._dataset_name + "/" + self.aux_train_type + "/"
        self._path_save_model = "./Saved/trained_models/" + self._dataset_name + "/" + self.aux_train_type + "/"

        pathlib.Path(self._path_artifacts).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self._path_aux ).mkdir(parents=True, exist_ok=True) 
        pathlib.Path(self._path_checkpoint).mkdir(parents=True, exist_ok=True) 
        pathlib.Path(self._path_configs).mkdir(parents=True, exist_ok=True) 
        pathlib.Path(self._path_plot).mkdir(parents=True, exist_ok=True) 
        pathlib.Path(self._path_save_model).mkdir(parents=True, exist_ok=True) 

        if not nullify and not restored:
            self._DataConverter = DataPrep(vgm_mode = self.vgm_mode)
            self._DataConverter.fit(raw_data = raw_dataset, discrete_columns = categorical_columns, datetime_columns = datetime_columns)
            with open("{}data_preprocessor_main.pkl".format(self._path_aux), "wb") as save_trf:
                pickle.dump(self._DataConverter, save_trf)
            if self.vgm_mode:
                self._AlternateDataConverter = DataPrep(vgm_mode = False)
                self._AlternateDataConverter.fit(raw_data = raw_dataset, discrete_columns = categorical_columns, datetime_columns = datetime_columns)
                with open("{}data_preprocessor_alt.pkl".format(self._path_aux), "wb") as save_trf:
                    pickle.dump(self._AlternateDataConverter, save_trf)
            
        else:
            with open("{}data_preprocessor_main.pkl".format(self._path_aux), "rb") as read_trf:
                self._DataConverter = pickle.load(read_trf)
            if self.vgm_mode:
                with open("{}data_preprocessor_alt.pkl".format(self._path_aux), "rb") as read_trf:
                    self._AlternateDataConverter = pickle.load(read_trf)
        
        trf_train_data = self._DataConverter.transform(raw_dataset)

        self.data_col_info = self._DataConverter.gan_ColumnInfo

        self._lambda_ac = np.linspace(start = self._lambda_ac_start, stop = self._lambda_ac_stop, num = len(self.data_col_info))

        if not restored and not nullify:
            if self.vgm_mode:
                reverted_trf_data = self.invert_vgm(data = trf_train_data, col_info = self.data_col_info, data_converter = self._DataConverter)
                self.aux_classifiers = self.train_auxs(aux_data = reverted_trf_data, col_info = self.data_col_info)
            else:
                reverted_trf_data = self.invert_num_scale(data = trf_train_data, col_info = self.data_col_info, data_converter = self._DataConverter)
                self.aux_classifiers = self.train_auxs(aux_data = reverted_trf_data, col_info = self.data_col_info)
        else:
            restored_ac_models = []

            for ac_idx in range(1, self._n_raw_columns + 1):
                ac_filename = self._path_aux + "ac_{}.txt".format(ac_idx)
                ac_loaded_model = lgb.Booster(model_file=ac_filename)
                restored_ac_models.append(ac_loaded_model)

            self.aux_classifiers = restored_ac_models

        trf_train_data_tensor = torch.Tensor(trf_train_data).to(self._device)
        data_batcher = DataLoader(trf_train_data_tensor, batch_size=self._batch_size, shuffle = True)

        if not restored:
            self._generator = Generator(data_ColInfo = self.data_col_info,
                                        noise_dim = self._noise_dim,
                                        device = self._device,
                                        vgm_mode = self.vgm_mode,
                                        **self.generator_structure_kwargs
                                    ).to(self._device)

            self._discriminator = Discriminator(data_ColInfo = self.data_col_info,
                                                device = self._device,
                                                vgm_mode = self.vgm_mode,
                                                **self.discriminator_structure_kwargs
                                    ).to(self._device)

            self._optimizerG = optim.Adam( self._generator.parameters(), lr = 0.0002, betas=(0.5, 0.99) )
            self._optimizerD = optim.Adam( self._discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.99) )

        mean_vec = torch.zeros(self._batch_size, self._noise_dim, device=self._device)
        std_vec = mean_vec + 1

        print("Model Training")

        for i in range(self._epochs):
            temp_val = 0
            loss_g_epoch = []
            loss_aux_epoch = []
            loss_d_epoch = []
            loss_wass_epoch = []
            loss_gp_epoch = []
            for batched_data in data_batcher:

                curr_batched_size = len(batched_data)

                #-----------------------------
                # Training the Discriminator |
                #-----------------------------
                for _ in range(self._discriminator_steps):

                    rdm_noise = torch.normal(mean=mean_vec[:curr_batched_size], std=std_vec[:curr_batched_size])

                    fake_vector = self._generator(rdm_noise)

                    fake_data_ohe = fake_vector[-1][0]

                    real_data_lbe = batched_data

                    if self.noisify_disc_input:
                        real_data_ohe_prenoise = self._discriminator.one_hot_fn(real_data_lbe).detach()
                        real_data_ohe = self._discriminator.noisify_inputs(real_data_ohe_prenoise)
                    else:
                        real_data_ohe = self._discriminator.one_hot_fn(real_data_lbe).detach()


                    fk_ohe = fake_data_ohe

                    y_fake = self._discriminator(fk_ohe)
                    y_real = self._discriminator(real_data_ohe)

                    self._gen_data = fake_data_ohe

                    wasserstein_loss = torch.mean(y_fake) - torch.mean(y_real)

                    gp_loss = self._discriminator.calculate_gradient_penalty_loss(real_data_ohe, fake_data_ohe, curr_batched_size) * self._lambda_gp

                    gp_loss = gp_loss * self._lambda_gp
                    loss_d = wasserstein_loss + gp_loss

                    
                    loss_d_epoch.append(loss_d)
                    loss_wass_epoch.append(wasserstein_loss)
                    loss_gp_epoch.append(gp_loss)

                    self._optimizerD.zero_grad()
                    wasserstein_loss.backward(retain_graph=True)
                    gp_loss.backward()

                    self._optimizerD.step()

                #-------------------------
                # Training the Generator |
                #-------------------------

                rdm_noise = torch.normal(mean=mean_vec[:curr_batched_size], std=std_vec[:curr_batched_size])

                fake_vectors = self._generator(rdm_noise)

                g_disc_scores = []
                aux_losses = []
                comb_losses = []

                loop_elems = tuple(zip(self.data_col_info, fake_vectors))  # -> (idx, (output, targets))

                for elm_idx, elem in enumerate(loop_elems):

                    col = elem[0]
                    fake_out = elem[1][0]
                    fake_out_lbe = self._generator.label_encode_fn(fake_out)
                    fake_target = elem[1][1]
                    y_fake = self._discriminator(fake_out)
                    loss_g = -torch.mean(y_fake)
                    g_disc_scores.append(loss_g)

                    current_idx = [elm_idx]
                    other_idx = [x for x in range(len(self.data_col_info)) if x not in current_idx]

                    numpyed_batch = fake_out_lbe.cpu().detach().clone().numpy()


                    if self.vgm_mode:
                        vgm_label_data = fake_target.detach().squeeze().clone()
                        vgm_label_copy = vgm_label_data.cpu().numpy()
                        col_trf_info = self._DataConverter._column_transform_info_list[elm_idx]
                        alt_col_trf_info = self._AlternateDataConverter._column_transform_info_list[elm_idx]
                        if col.column_gan_type == "numerical":
                            inv_output  = self._DataConverter._inverse_transform_continuous_vgm(col_trf_info, vgm_label_copy)
                            gen_label = torch.from_numpy(inv_output.astype("float32")).reshape(-1, 1)
                        else:
                            gen_label = fake_target.detach()
                        rvrt_batch = self.invert_vgm(data = numpyed_batch, col_info=self.data_col_info, data_converter=self._DataConverter)
                        gen_data = rvrt_batch[:, other_idx]
                    else:
                        label_data = fake_target.detach().clone()
                        label_copy = label_data.cpu().numpy()
                        col_trf_info = self._DataConverter._column_transform_info_list[elm_idx]
                        if col.column_gan_type == "numerical":
                            inv_output  = self._DataConverter._inverse_transform_continuous(col_trf_info, label_copy)
                            gen_label = torch.from_numpy(inv_output.astype("float32")).reshape(-1, 1)
                        else:
                            gen_label = fake_target.detach()
                        rvrt_batch = self.invert_num_scale(data = numpyed_batch, col_info=self.data_col_info, data_converter=self._DataConverter)
                        gen_data = rvrt_batch[:, other_idx]

                    bst = self.aux_classifiers[elm_idx]
                    preds_np = bst.predict(gen_data, num_iteration=bst.best_iteration)

                    if self.vgm_mode and col.column_gan_type == "numerical":
                        gen_label = self._AlternateDataConverter._transform_continuous(alt_col_trf_info, gen_label)[0]
                        preds_np = self._AlternateDataConverter._transform_continuous(alt_col_trf_info, preds_np.reshape(-1, 1))[0]
                    elif col.column_gan_type == "numerical" and not self.vgm_mode:
                        gen_label = self._DataConverter._transform_continuous(col_trf_info, gen_label)[0]
                        preds_np = self._DataConverter._transform_continuous(col_trf_info, preds_np.reshape(-1, 1))[0]

                    if col.column_gan_type == "numerical":
                        gen_label = torch.unsqueeze(torch.from_numpy(gen_label.astype('float32')).to(self._device), dim = 1)
                        preds = torch.unsqueeze(torch.from_numpy(preds_np.astype('float32')).to(self._device), dim = 1)
                        ac_loss = F.mse_loss(gen_label, preds) # switching the order since we need the G output to conform to the preds
                    else:

                        preds = torch.from_numpy(preds_np.astype('float32')).to(self._device)
                        ac_loss = F.cross_entropy(gen_label, preds)

                    mean_aux_loss = torch.mean(ac_loss).float() * self._lambda_ac[elm_idx]
                    aux_losses.append(mean_aux_loss)
                    comb_losses.append(loss_g + mean_aux_loss)

                loss_g_epoch.append(loss_g)
                loss_aux_epoch.append(mean_aux_loss)

                self._optimizerG.zero_grad()

                for g_idx, g_loss in enumerate(comb_losses):
                    if g_idx + 1 < len(self.data_col_info):
                        g_loss.backward(retain_graph=True)
                    else:
                        g_loss.backward()

                self._optimizerG.step()
                temp_val += self._batch_size

            #------------------
            # Printing Output |
            #------------------

            print("")

            print("y_fake_mean: {:.5f}, ".format(torch.mean(y_fake).detach().cpu().numpy()), 
                "y_fake_median: {:.5f}, ".format(torch.median(y_fake).detach().cpu().numpy()),
                "y_fake_min: {:.5f}, ".format(torch.min(y_fake).detach().cpu().numpy()), 
                "y_fake_max: {:.5f}".format(torch.max(y_fake).detach().cpu().numpy()))

            print("y_real_mean: {:.5f}, ".format(torch.mean(y_real).detach().cpu().numpy()), 
                    "y_real_median: {:.5f}, ".format(torch.median(y_real).detach().cpu().numpy()),
                    "y_real_min: {:.5f}, ".format(torch.min(y_real).detach().cpu().numpy()), 
                    "y_real_max: {:.5f}".format(torch.max(y_real).detach().cpu().numpy()))

            print("----------------------------------------------------------------------------------")

            loss_g_epoch_prnt = np.mean([i.detach().cpu().numpy() for i in loss_g_epoch])
            loss_aux_epoch_prnt = np.mean([i.detach().cpu().numpy() for i in loss_aux_epoch])
            loss_d_epoch_prnt = np.mean([i.detach().cpu().numpy() for i in loss_d_epoch])
            loss_wass_epoch_prnt = np.mean([i.detach().cpu().numpy() for i in loss_wass_epoch])
            loss_gp_epoch_prnt = np.mean([i.detach().cpu().numpy() for i in loss_gp_epoch])

            self.G_loss.append(loss_g_epoch_prnt)
            self.AUX_loss.append(loss_aux_epoch_prnt)
            self.D_loss.append(loss_d_epoch_prnt)
            self._wass_loss.append(loss_wass_epoch_prnt)
            self._gp_loss.append(loss_gp_epoch_prnt)

            # Save checkpoints (every 10 epochs)
            if (i+1) % 1 == 0:
                renamed_checkpoint_path = self._path_checkpoint + "epoch_" + str(i+1) + ".pth"
                self.save_checkpoint(i, renamed_checkpoint_path)

            if self._verbose:
                print(f"Epoch {i+1}/{self._epochs}, Loss G: {loss_g_epoch_prnt: .4f}, "
                      f" Loss AUX: {loss_aux_epoch_prnt: .4f}, ",
                      f" Loss D: {loss_d_epoch_prnt: .4f}, ",
                      f" Loss Wass: {loss_wass_epoch_prnt: .4f}, ",
                      f" Loss GP:, {loss_gp_epoch_prnt: .4f}",
                      flush=True)
                    
        self.timestr = time.strftime("%Y%m%d-%H%M")
        if self._epochs > 0:
            model_save_path = self._path_save_model + self.timestr + ".pth"
            self.save_model(model_save_path)
            self.save_plot()
            self.save_config()
            self.save_artifacts()


    def debug_sample(self, num_samples = 1000, file_name = ""):
        
        steps = (num_samples // self._batch_size) + 1
        col_block = OrderedDict()
        data = []

        for idx, col in enumerate(self.data_col_info):
            col_block[col.column_name] = []
        
        for _ in range(steps):
            mean_vec = torch.zeros(self._batch_size, self._noise_dim)
            std_vec = mean_vec + 1
            rdm_noise = torch.normal(mean=mean_vec, std=std_vec).to(self._device)

            fake_ff = self._generator(rdm_noise)

            for idx, col in enumerate(self.data_col_info):

                fake_data = fake_ff[idx][0]

                fake_out_lbe = self._generator.label_encode_fn(fake_data)

                col_block[col.column_name].append(fake_out_lbe.detach().cpu().numpy())

        dreversed = OrderedDict()
        for k in reversed(col_block):
            dreversed[k] = col_block[k]

        writer = pd.ExcelWriter(file_name, engine = "xlsxwriter")

        for key, data in dreversed.items():
            data = np.concatenate(data, axis=0)
            data = data[:num_samples]
            df = self._DataConverter.inverse_transform(data)
            df.to_excel(writer, sheet_name = key, index=False)

        writer.save()


    def sample(self, num_samples = 1000):
        
        steps = (num_samples // self._batch_size) + 1
        data = []
        for _ in range(steps):
            mean_vec = torch.zeros(self._batch_size, self._noise_dim)
            std_vec = mean_vec + 1
            rdm_noise = torch.normal(mean=mean_vec, std=std_vec).to(self._device)

            fake_ff = self._generator(rdm_noise)

            fake_data = fake_ff[-1][0]

            fake_out_lbe = self._generator.label_encode_fn(fake_data)

            data.append(fake_out_lbe.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:num_samples]

        return self._DataConverter.inverse_transform(data)


    def save_plot(self):
        plt.ioff()
        g_loss_pts = [item for item in self.G_loss]
        aux_loss_pts = [item for item in self.AUX_loss]
        d_loss_pts = [item for item in self.D_loss]
        wass_loss_pts = [item for item in self._wass_loss]
        gp_loss_pts = [item for item in self._gp_loss]
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (14,10)

        # Plot Generator loss in orange
        plt.plot(np.linspace(1, self._epochs, len(g_loss_pts)),
                 g_loss_pts, color ='darkorange')

        # Plot Final Layer AUX loss in green
        plt.plot(np.linspace(1, self._epochs, len(aux_loss_pts)),
                 aux_loss_pts, linestyle='dashed', color = 'green')

        # Plot Discriminator loss in blue
        plt.plot(np.linspace(1, self._epochs, len(d_loss_pts)),
                 d_loss_pts, color = 'blue')

        # Plot WASS loss in lightblue
        plt.plot(np.linspace(1, self._epochs, len(wass_loss_pts)),
                 wass_loss_pts, linestyle='dashed', color = 'cornflowerblue')

        # Plot GP loss in navy
        plt.plot(np.linspace(1, self._epochs, len(gp_loss_pts)),
                 gp_loss_pts, linestyle='dashed', color = 'navy')

        # Add legend, title
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim([-2.5, 2.5])
        plt.legend(['Generator', "Auxiliary (final layer)", "Discriminator", "Wasserstein (D)", "Gradient Penalty (D)"])
        plt.title("NestGAN Losses")
        img_path_name = self._path_plot + self.timestr + ".png"
        plt.savefig(img_path_name)
        plt.close()


    def disp_plot(self):
        """ Visualize loss for the generator, discriminator """

        g_loss_pts = [item for item in self.G_loss]
        aux_loss_pts = [item for item in self.AUX_loss]
        d_loss_pts = [item for item in self.D_loss]
        wass_loss_pts = [item for item in self._wass_loss]
        gp_loss_pts = [item for item in self._gp_loss]
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (14,10)

        # Plot Generator loss in orange
        plt.plot(np.linspace(1, self._epochs, len(g_loss_pts)),
                 g_loss_pts, color ='darkorange')

        plt.plot(np.linspace(1, self._epochs, len(aux_loss_pts)),
                 aux_loss_pts, linestyle='dashed', color = 'green')

        # Plot Discriminator loss in blue
        plt.plot(np.linspace(1, self._epochs, len(d_loss_pts)),
                 d_loss_pts, color = 'blue')

        # Plot WASS loss in lightblue
        plt.plot(np.linspace(1, self._epochs, len(wass_loss_pts)),
                 wass_loss_pts, linestyle='dashed', color = 'cornflowerblue')

        # Plot GP loss in navy
        plt.plot(np.linspace(1, self._epochs, len(gp_loss_pts)),
                 gp_loss_pts, linestyle='dashed', color = 'navy')

        # Add legend, title
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim([-2.5, 2.5])
        plt.legend(['Generator', "Auxiliary (final layer)", "Discriminator", "Wasserstein (D)", "Gradient Penalty (D)"])
        plt.title("CasTGAN Losses")
        plt.show()


    def save_config(self):
        
        model_params = {
            "batch_size": self._batch_size,
            "noise_dim": self._noise_dim,
            "generator_structure": self.generator_structure_kwargs,
            "discriminator_structure": self.discriminator_structure_kwargs,
            "discriminator_steps": self._discriminator_steps,
            "noisify_disc_input": self.noisify_disc_input,
            "lambda_gp": self._lambda_gp,
            "lambda_ac_start": self._lambda_ac_start,
            "lambda_ac_stop": self._lambda_ac_stop,
            "aux_train_type": self.aux_train_type,
            "random_seed": self._random_seed,
            "vgm_mode": self.vgm_mode
        }

        config_file_name = self._path_configs + self.timestr + ".yaml"
        with open(config_file_name, 'w') as yamlfile:
            yaml.dump(model_params, yamlfile)


    def save_artifacts(self):

        full_model_params = vars(self).copy()
        
        redund = ["G_loss", "D_loss", "AUX_loss", "_wass_loss", "_gp_loss"]
        for attr in redund:
            _ = full_model_params.pop(attr,None)

        hparams_allowed = {}
        accepted_types = [str, int, list, tuple, dict, bool]

        for k, v in full_model_params.items():
            if type(v) not in accepted_types:
                new_v = str(v)
                hparams_allowed[k] = new_v
                
            else:
                hparams_allowed[k] = v

        # saving the standard way
        config_ff_name = self._path_artifacts + self.timestr + ".yaml"
        with open(config_ff_name, "w") as yamlfile:
            yaml.dump(hparams_allowed, yamlfile, encoding='utf-8', line_break=False, allow_unicode=True)

    def save_checkpoint(self, current_epoch, path):
        torch.save({"epoch" : current_epoch,
                    "model_state_dict" : self.state_dict(),
                    "G_optimizer_state_dict" : self._optimizerG.state_dict(),
                    "D_optimizer_state_dict" : self._optimizerD.state_dict() }, 
                    path)

    # Use load_checkpoint to resume training or for sampling
    def load_checkpoint(self, path, inference_mode = False):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'], strict = False)
        self._optimizerG.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self._optimizerD.load_state_dict(checkpoint['D_optimizer_state_dict'])
        self._past_epochs = checkpoint["epoch"]
        if inference_mode:
            self.eval()


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    # Use load_model only for sampling (won't be able to recover the optimizers' states)
    def load_model(self, path, inference_mode = True):
        self.load_state_dict( torch.load(path), strict = False )
        if inference_mode: # Inference mode if training will not be continued
            self.eval()