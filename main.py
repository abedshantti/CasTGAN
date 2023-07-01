import torch
import argparse
import pandas as pd
import pickle
import glob
import os
import time
import numpy as np
import yaml
import pathlib
from collections import namedtuple

from Model.CasTGAN import CasTGAN

## TODO: add reproduce_paper boolean argument to specify seed for dataset if reproducability behaviour desired

parser = argparse.ArgumentParser(description="Master File")

parser.add_argument(
    "--dataset", 
    type=str, 
    default="adult", 
    help="""Name of dataset csv file. Use one of:
    "adult"
    "banking"
    "cars"
    "credit"
    "diabetes"
    "housing"
    "lending_club"
    "students"
    or use your own dataset WITHOUT adding ".csv" at the end
    """
)

parser.add_argument(
    "--data_path",
    type=str,
    default="Data/",
    help="Directory of data files",
)

parser.add_argument(
    "--save_fake_data_path",
    type=str,
    default="Generated_Data/",
    help="Directory for saving fake data",
)

parser.add_argument(
    "--n_fake_samples",
    type=int,
    default=-1,
    help="Number of synthetic samples to be produced. If not specified, defaults to the training size.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    required=False,
    help="Batch size",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=300,
    help="number of training epochs",
)

parser.add_argument(
    "--noise_dim",
    type=int,
    default=128,
    help="size of noise vector",
)

parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')
parser.add_argument('--layer_norm_off', dest='layer_norm', action='store_false')
parser.set_defaults(layer_norm=True)

parser.add_argument(
    "--generator_hidden_sizes", 
    nargs="+", 
    type=int,
    default=[128, 64],
    required=False,
    help="Hidden layers widths (insert the hidden widths separated by commas e.g.: 10, 20, 10)",
)

parser.add_argument(
    "--generator_inter_layer_act",
    type=str,
    default="leakyrelu",
    help="Activation fn within the generator layers",
)

parser.add_argument(
    "--generator_gmb_softmax_tau",
    type=float,
    default=0.8,
    help="Gumbel Softmax Temperature coefficient",
)

parser.add_argument(
    "--discriminator_hidden_sizes", 
    nargs="+", 
    type=int,
    default=[256, 128],
    required=False,
    help="Hidden layers widths (insert the hidden widths separated by commas e.g.: 10, 20, 10)",
)

parser.add_argument(
    "--discriminator_inter_layer_act",
    type=str,
    default="leakyrelu",
    help="Activation fn within the discriminator layers",
)

parser.add_argument(
    "--discriminator_dropout",
    type=float,
    default=0.0,
    help="Dropout rate for discriminator if needed",
)

parser.add_argument(
    "--categorical_columns", 
    nargs="+", 
    type=str,
    default=[],
    required=False,
    help="Categorical columns if they need to be manually defined, otherwise leave blank for auto-infer.",
)

parser.add_argument('--noisify_disc_input', dest='noisify_disc_input', action='store_true')
parser.add_argument('--noisify_disc_input_off', dest='noisify_disc_input', action='store_false')
parser.set_defaults(noisify_disc_input=True)

parser.add_argument(
    "--discriminator_steps",
    type=int,
    default=1,
    help="Number of disc steps",
)

parser.add_argument(
    "--lambda_gp",
    type=float,
    default=10.0,
    help="Gradient penalty coefficient parameter",
)

parser.add_argument(
    "--lambda_ac_start",
    type=float,
    default=0.75,
    help="lambda AC 1",
)

parser.add_argument(
    "--lambda_ac_stop",
    type=float,
    default=0.10,
    help="lambda AC M",
)

parser.add_argument('--vgm_mode', dest='vgm_mode', action='store_true')
parser.add_argument('--vgm_mode_off', dest='vgm_mode', action='store_false')
parser.set_defaults(vgm_mode=True)

parser.add_argument(
    "--aux_fit_mode",
    type=str,
    default="wb_0",
    help=""" Modes for fitting the auxiliary classifier. This corresponds to the whether to perturb the data for protection against privacy attacks. Choose from ["wb_0", "wb_1", "wb_2", "wb_3"] """,
)

parser.add_argument('--reproduce_paper', dest='reproduce_paper', action='store_true')
parser.add_argument('--reproduce_paper_off', dest='reproduce_paper', action='store_false')
parser.set_defaults(reproduce_paper=False)

parser.add_argument(
    "--random_seed",
    type=int,
    default=-1,
    help="Random seed if needed. Defaults to -1, which is equivalent to None. Leave blank if not desired.",
)

parser.add_argument('--resume_training', dest='resume_training', action='store_true')
parser.set_defaults(resume_training=False)

parser.add_argument('--load_trained_model', dest='load_trained_model', action='store_true')
parser.set_defaults(load_trained_model=False)

args = parser.parse_args()

if args.random_seed == -1 and not args.reproduce_paper:
    args.random_seed = None

if args.aux_fit_mode not in ["wb_0", "wb_1", "wb_2", "wb_3"]:
    args.aux_fit_mode = "wb_0"          # No aux permutation if user defined fit mode is invalid

if args.reproduce_paper and args.aux_fit_mode=="wb_0":
    if args.dataset == "adult":
        args.random_seed = 88001
    elif args.dataset == "bank":
        args.random_seed = 88002
    elif args.dataset == "cars":
        args.random_seed = 88003
    elif args.dataset == "credit":
        args.random_seed = 88004
    elif args.dataset == "diabetes":
        args.random_seed = 88005
    elif args.dataset == "housing":
        args.random_seed = 88006
elif args.reproduce_paper and args.aux_fit_mode=="wb_1":
    if args.dataset == "adult":
        args.random_seed = 88101
    elif args.dataset == "bank":
        args.random_seed = 88102
    elif args.dataset == "cars":
        args.random_seed = 88103
    elif args.dataset == "credit":
        args.random_seed = 88104
    elif args.dataset == "diabetes":
        args.random_seed = 88105
    elif args.dataset == "housing":
        args.random_seed = 88106
elif args.reproduce_paper and args.aux_fit_mode=="wb_2":
    if args.dataset == "adult":
        args.random_seed = 88201
    elif args.dataset == "bank":
        args.random_seed = 88202
    elif args.dataset == "cars":
        args.random_seed = 88203
    elif args.dataset == "credit":
        args.random_seed = 88204
    elif args.dataset == "diabetes":
        args.random_seed = 88205
    elif args.dataset == "housing":
        args.random_seed = 88206
elif args.reproduce_paper and args.aux_fit_mode=="wb_3":
    if args.dataset == "adult":
        args.random_seed = 88301
    elif args.dataset == "bank":
        args.random_seed = 88302
    elif args.dataset == "cars":
        args.random_seed = 88303
    elif args.dataset == "credit":
        args.random_seed = 88304
    elif args.dataset == "diabetes":
        args.random_seed = 88305
    elif args.dataset == "housing":
        args.random_seed = 88306

dtype_pkl_file = args.data_path + args.dataset + "_dtypes.pkl"

train_file_name = args.data_path + args.dataset + "_train.csv"
alt_train_file_name = args.data_path + args.dataset + ".csv"

test_file_name = args.data_path + args.dataset + "_test.csv"

if os.path.isfile(train_file_name):
    train_file_name = train_file_name
elif os.path.isfile(alt_train_file_name):
    train_file_name = alt_train_file_name
else:
    raise FileNotFoundError("Could not find {}.csv inside {}".format(args.dataset, args.data_path))


if os.path.isfile(dtype_pkl_file):
    with open(dtype_pkl_file, 'rb') as pklr:
        dataset_dtypes = pickle.load(pklr)
    train_data = pd.read_csv(train_file_name,  header = 0, dtype = dataset_dtypes)
    if len(args.categorical_columns) == 0:
        catg_cols = [k for k,v in dataset_dtypes.items() if v == "str"]
else:
    train_data = pd.read_csv(train_file_name,  header = 0)
    if len(args.categorical_columns) == 0:
        catg_cols = [col for col in train_data.columns if train_data.dtypes[col] == "object"]

train_data.name = args.dataset

if args.load_trained_model:
    model_train_mode = "only_generate"
    loaded_model_dir = "Saved/trained_models/{}/{}/".format(args.dataset, args.aux_fit_mode)
    list_of_files = glob.glob('{}*.pth'.format(loaded_model_dir))
    loaded_model_file = max(list_of_files, key=os.path.getctime)

    config_dir = "Saved/configs/{}/{}/".format(args.dataset, args.aux_fit_mode)
    list_of_files = glob.glob('{}*.yaml'.format(config_dir))
    config_file = max(list_of_files, key=os.path.getctime)
    with open(config_file) as f:
        resume_class_map = yaml.safe_load(f)

if args.resume_training:
    model_train_mode = "resume"
    checkpoint_dir = "Saved/checkpoints/{}/{}/".format(args.dataset, args.aux_fit_mode)
    list_of_files = glob.glob('{}*.pth'.format(checkpoint_dir))
    checkpoint_file = max(list_of_files, key=os.path.getctime)

    config_dir = "Saved/configs/{}/{}/".format(args.dataset, args.aux_fit_mode)
    list_of_files = glob.glob('{}*.yaml'.format(config_dir))
    config_file = max(list_of_files, key=os.path.getctime)
    with open(config_file) as f:
        resume_class_map = yaml.safe_load(f)


if not args.load_trained_model and not args.resume_training:
    model_train_mode = "train"

print("")
print(args.dataset)
print("")


if model_train_mode == "train":

    model = CasTGAN(noise_dim=args.noise_dim,
                    batch_size=args.batch_size,
                    generator_structure = {"hidden_sizes": args.generator_hidden_sizes,
                                            "layer_norm": args.layer_norm,
                                            "inter_layer_act": args.generator_inter_layer_act,
                                            "gmb_softmax_tau": args.generator_gmb_softmax_tau},
                    discriminator_structure = {"hidden_sizes": args.discriminator_hidden_sizes,
                                                "layer_norm": args.layer_norm,
                                                "inter_layer_act": args.discriminator_inter_layer_act,
                                                "dropout": args.discriminator_dropout},
                    noisify_disc_input = args.noisify_disc_input,
                    discriminator_steps = args.discriminator_steps,
                    lambda_gp = args.lambda_gp,
                    lambda_ac_start = args.lambda_ac_start,
                    lambda_ac_stop = args.lambda_ac_stop,
                    aux_train_type = args.aux_fit_mode,
                    random_seed = args.random_seed,
                    vgm_mode = args.vgm_mode)


    start_train = time.time()
    model.fit(train_data, categorical_columns = catg_cols, epochs = args.epochs)
    end_train = time.time()

elif model_train_mode == "resume":

    model = CasTGAN(noise_dim = resume_class_map["noise_dim"],
                    batch_size=resume_class_map["batch_size"],
                    generator_structure = resume_class_map["generator_structure"],
                    discriminator_structure = resume_class_map["discriminator_structure"],
                    noisify_disc_input = resume_class_map["noisify_disc_input"],
                    discriminator_steps = resume_class_map["discriminator_steps"],
                    lambda_gp = resume_class_map["lambda_gp"],
                    lambda_ac_start = resume_class_map["lambda_ac_start"],
                    lambda_ac_stop = resume_class_map["lambda_ac_stop"],
                    aux_train_type = resume_class_map["aux_train_type"],
                    random_seed = resume_class_map["random_seed"],
                    vgm_mode = resume_class_map["vgm_mode"])
    model.fit(train_data, categorical_columns = catg_cols, epochs = 0, nullify = True)
    model.load_checkpoint(checkpoint_file, inference_mode = False)
    start_train = time.time()
    model.fit(train_data, categorical_columns = catg_cols, epochs = args.epochs, restored=True)
    end_train = time.time()

elif model_train_mode == "only_generate":
    model = CasTGAN(noise_dim = resume_class_map["noise_dim"],
                    batch_size=resume_class_map["batch_size"],
                    generator_structure = resume_class_map["generator_structure"],
                    discriminator_structure = resume_class_map["discriminator_structure"],
                    noisify_disc_input = resume_class_map["noisify_disc_input"],
                    discriminator_steps = resume_class_map["discriminator_steps"],
                    lambda_gp = resume_class_map["lambda_gp"],
                    lambda_ac_start = resume_class_map["lambda_ac_start"],
                    lambda_ac_stop = resume_class_map["lambda_ac_stop"],
                    aux_train_type = resume_class_map["aux_train_type"],
                    random_seed = resume_class_map["random_seed"],
                    vgm_mode = resume_class_map["vgm_mode"])
    model.fit(train_data, categorical_columns = catg_cols, epochs = 0, nullify=True)
    model.load_model(loaded_model_file)
    start_train = time.time()
    end_train = time.time()

if args.n_fake_samples == -1:
    args.n_fake_samples = len(train_data)

print("")
print("Generating fake samples")

fake_df = model.sample(num_samples = args.n_fake_samples)

end_gen = time.time()

timestr = time.strftime("%Y%m%d-%H%M")

save_FP = args.save_fake_data_path  + args.aux_fit_mode
pathlib.Path(save_FP).mkdir(parents=True, exist_ok=True)

fake_data_name = save_FP + "/" + args.dataset + "_fake_" + timestr + ".csv"

fake_df.to_csv(fake_data_name, index = False)

elapsed_train = end_train - start_train
elapsed_total = end_gen - start_train

with open('time_python_dis.txt', 'a') as fd:
    fd.write("{} dataset: Training time: {} seconds. Total time: {} seconds.\n".format( args.dataset, str(int(np.round(elapsed_train))), str(int(np.round(elapsed_total))) ) )

print("Done")