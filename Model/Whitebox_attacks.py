import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler,  OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance

from .DataPrep import DataPrep

class whitebox_privacy_eval:

    def __init__(self,
                corrupt_ratio = 0.10,
                num_attack_iters = 10,
                train_set = None,
                categorical_cols = [],
                ac_filepath = "",
                load_existing_dataconv = False,
                syn_output = {},
                random_seed = None):
        super(whitebox_privacy_eval, self).__init__()

        self.corrupt_ratio = corrupt_ratio
        self.num_attack_iters = num_attack_iters
        self.train_set = train_set
        self.categorical_cols = categorical_cols
        self.ac_filepath = ac_filepath
        self.load_existing_datacov = load_existing_dataconv
        self.syn_output = syn_output
        self.random_seed = random_seed

        np.random.seed(self.random_seed)

    def invert_vgm(self, data = None, col_info = [], data_converter = None):

        st = 0
        column_data_list = []
        for idx,col in enumerate(col_info):
            dim = col.column_dim
            conv_data = data[:, st:st+dim]
            if col.column_gan_type == "numerical":
                col_trnf_info = data_converter._column_transform_info_list[idx]
                #if col_trnf_info.name == col.column_name:
                mod_data = data_converter._inverse_transform_continuous_vgm(col_trnf_info, conv_data)
                mod_data_rshp = mod_data.reshape(data.shape[0],1)
                column_data_list.append(mod_data_rshp)
            else:
                column_data_list.append(conv_data)
            
            st += dim

        np_data = np.concatenate(column_data_list, axis=1).astype(float)

        inver_df = pd.DataFrame(np_data, columns=self.train_set.columns)

        return inver_df

    def invert_minmax(self, data=None, col_info = [], data_converter = None):

        st = 0
        column_data_list = []
        for idx,col in enumerate(col_info):
            dim = col.column_dim
            conv_data = data[:, st:st+dim]
            if col.column_gan_type == "numerical":
                col_trnf_info = data_converter._column_transform_info_list[idx]
                #if col_trnf_info.name == col.column_name:
                mod_data = data_converter._inverse_transform_continuous(col_trnf_info, conv_data)
                #mod_data_rshp = mod_data.reshape(data.shape[0],1)
                column_data_list.append(mod_data)
            else:
                column_data_list.append(conv_data)
            
            st += dim

        np_data = np.concatenate(column_data_list, axis=1).astype(float)

        inver_df = pd.DataFrame(np_data, columns=self.train_set.columns)

        return inver_df


    def invert_lbe(self, df = None, col_info = [], data_converter = None):

        st = 0
        column_data_list = []
        for idx,col in enumerate(col_info):
            dim = 1
            conv_data = df.iloc[:, st:st+dim]
            if col.column_gan_type == "categorical":
                col_trnf_info = data_converter._column_transform_info_list[idx]
                #if col_trnf_info.name == col.column_name:
                mod_data = data_converter._inverse_transform_discrete(col_trnf_info, conv_data.values)
                cat_vec = mod_data.tolist()
                df.iloc[:, st:st+dim] = cat_vec
                df[col.column_name] = df[col.column_name].astype("object")
                #mod_data_rshp = mod_data.reshape(df.shape[0],1)
                #column_data_list.append(mod_data_rshp)
            # else:
            #     column_data_list.append(conv_data)
            
            st += dim

        # np_data = np.concatenate(column_data_list, axis=1).astype(float)

        #inver_df = pd.DataFrame(np_data, columns=self.train_set.columns)

        return df

    
    # Possibly also need an invert_num_cols fn

    
    def init_preprocessors(self, filepath_ac = None, load_existing = True):

        self.trf_syn_output = {}

        if load_existing:
            file_name = filepath_ac + "/data_preprocessor_alt.pkl"
            # Check that the file exists in directory else return error message
            with open(file_name, "rb") as read_trf:
                self.DataConverter = pickle.load(read_trf)

            self.data_col_info = self.DataConverter.gan_ColumnInfo

            for name, syn in self.syn_output.items():
                trf_syn_data = self.DataConverter.transform(syn)
                if self.DataConverter.vgm_mode == True:
                    trf_syn_data = self.invert_vgm(data = trf_syn_data, col_info = self.data_col_info, data_converter= self.DataConverter)
                else:
                    trf_syn_data = self.invert_minmax(data = trf_syn_data, col_info = self.data_col_info, data_converter= self.DataConverter)
                self.trf_syn_output[name] = trf_syn_data
            
        else:

            for name, syn in self.syn_output.items():
                
                self.DataConverter = DataPrep(vgm_mode = False)
                self.DataConverter.fit(raw_data = syn, discrete_columns = self.categorical_cols, datetime_columns = [])

                self.data_col_info = self.DataConverter.gan_ColumnInfo
                
                trf_syn_data = self.DataConverter.transform(syn)
                trf_syn_data = self.invert_minmax(data = trf_syn_data, col_info = self.data_col_info, data_converter= self.DataConverter)
                
                self.trf_syn_output[name] = trf_syn_data
                

        # self.data_col_info = self.DataConverter.gan_ColumnInfo

        # for name, syn in self.syn_output.items():
        #     trf_syn_data = self.DataConverter.transform(syn)
        #     if self.DataConverter.vgm_mode == True:
        #         trf_syn_data = self.invert_vgm(data = trf_syn_data, col_info = self.data_col_info, data_converter= self.DataConverter)
        #     else:
        #         trf_syn_data = self.invert_minmax(data = trf_syn_data, col_info = self.data_col_info, data_converter= self.DataConverter)
        #     self.trf_syn_output[name] = trf_syn_data



    def load_acs(self, filepath_ac = None):
        self.aux_models = []
        if filepath_ac is not None:
            n_cols = len(self.train_set.columns)

            for i in range(1, n_cols+1):
                file_name = filepath_ac + "/ac_{}.txt".format(i)
                aux_model = lgb.Booster(model_file=file_name)
                self.aux_models.append(aux_model)

        else:
            ### Train ACs on transformed trained data
            print("Path not provided. Training new ACs.")

            aux_data = self.DataConverter.transform(self.train_set)
            num_iters = 150
            
            for idx, col in enumerate(self.data_col_info):

                current_idx = [idx]
                other_idx = [x for x in range(len(self.data_col_info)) if x not in current_idx]

                ds_params = {"verbose": -1}

                other_cols = [x for x in self.data_col_info if x != col]
                other_col_names = [x.column_name for x in other_cols]
                other_cat_col_names = [x.column_name for x in other_cols if x.column_gan_type == "categorical"]

                label = aux_data[:, current_idx]
                data = aux_data[:, other_idx]

                x_train, x_val, y_train, y_val = train_test_split(data, label, test_size = 0.10, shuffle=True, random_state = self.random_seed) #None for now

                train_data = lgb.Dataset(x_train, label = y_train, feature_name = other_col_names, categorical_feature = other_cat_col_names, params = ds_params)
                val_data = lgb.Dataset(x_val, label = y_val, feature_name = other_col_names, categorical_feature = other_cat_col_names, params = ds_params)

                lgb_params = {'num_leaves': 31, "learning_rate":0.1, 'verbose': -1, "random_state":self.random_seed} #None for now

                if col.column_gan_type == "numerical":
                    lgb_params["objective"] = "regression"
                    lgb_params["metric"] = "mse"
                else:
                    lgb_params["objective"] = "multiclass"
                    lgb_params["num_class"] = col.column_unique_cats
                    lgb_params["metric"] = "multi_logloss"
                    lgb_params["is_unbalance"] = True

                callers = [lgb.callback.early_stopping(stopping_rounds =  10, verbose =  False)]

                print("Training AC {} out of {}".format(str(idx+1), str(len(self.data_col_info))))

                aux_model = lgb.train(lgb_params, train_data, num_boost_round=num_iters, valid_sets=val_data, callbacks = callers)
                self.aux_models.append(aux_model)


    def recursive_attack(self):

        #ratio_attacked = self.samples_ratio
        self.syn_sampled = {}
        self.syn_sampled_pre_attack = {}
        for name, syn in self.trf_syn_output.items():
            sampled = syn.sample(frac = self.corrupt_ratio, random_state = self.random_seed)
            sampled = sampled.reset_index(drop = True)
            self.syn_sampled[name] = sampled
            self.syn_sampled_pre_attack[name] = sampled.copy()

        #self.syn_sampled_pre_attack = self.syn_sampled.copy()

        # print("DUDE")
        # print(self.syn_sampled_pre_attack)

        num_iters = self.num_attack_iters # Change to 10

        cols_order = list(range(len(self.data_col_info)))

        col_names  = [col.column_name for col in self.data_col_info]

        col_data_types = [col.column_gan_type for col in self.data_col_info]
        
        for name, syn in self.syn_sampled.items():

            for datapoint_idx in range(len(syn)): #Change to range(len(syn))

                # print("point number", datapoint_idx)

                # print("initial datapoint data")

                #print(syn.iloc[datapoint_idx,:])

                # TODO: calc euc to train - pre attack
                # TODO: calc euc to syn - pre attack

                for iter in range(num_iters):

                    # Shuffle order of col and AC

                    np.random.shuffle(cols_order)

                    iter_acs = [self.aux_models[i] for i in cols_order]
                    #iter_col_data_types = [col_data_types[i] for i in cols_order]
                    #iter_col_names = [col_names[i] for i in cols_order]

                    for idx in cols_order:

                        curr_idx = [idx]
                        other_idx = [x for x in range(len(col_data_types)) if x not in curr_idx]
                        curr_col_type = col_data_types[idx]
                        curr_col_name = col_names[idx]

                        covar_data = syn.iloc[datapoint_idx,other_idx]
                        bst = self.aux_models[idx]

                        pred = bst.predict(covar_data, num_iteration=bst.best_iteration)

                        if curr_col_type  == "categorical":
                            pred = np.argmax(pred)

                        self.syn_sampled[name].iloc[datapoint_idx, idx] = pred

                # TODO: calc euc to train - post attack
                # TODO: calc euc to syn - post attack


    def find_knn(self, nn_obj, query_set):
        '''
        :param nn_obj: Nearest Neighbor object
        :param query_imgs: query images
        :return:
            dist: distance between query samples to its KNNs among generated samples
            idx: index of the KNNs
        '''
        dist = []
        idx = []
        #for i in range(len(query_imgs)):
            #row = query_imgs.iloc[i]
        dist_batch, idx_batch = nn_obj.kneighbors(X=query_set, n_neighbors=1)
        dist.append(dist_batch)
        idx.append(idx_batch)

        try:
            dist = np.concatenate(dist)
            idx = np.concatenate(idx)
        except:
            dist = np.array(dist)
            idx = np.array(idx)
        return dist, idx

                        
    def calculate_proximity(self):
        ### OHE and MinMax from this point onwards

        # This function is for evaluation the distance between the training set and the synthetic attacked data

        # self.data_scaler = DataPrep(vgm_mode = False)
        # self.data_scaler.fit(raw_data = self.train_set, discrete_columns = self.categorical_cols, datetime_columns = [])
        # self.conv_trainset = self.data_scaler.transform(self.train_set)

        numeric_transformer = MinMaxScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numc_cols = [col for col in self.train_set.columns if col not in self.categorical_cols]
        catg_cols = [col for col in self.train_set.columns if col in self.categorical_cols]

        pre_prc = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numc_cols),
                ("cat", categorical_transformer, catg_cols),
            ]
        )


        pre_prc.fit(self.train_set)
        self.conv_trainset = pre_prc.transform(self.train_set)

        self.fake_pre_trf = {}

        proximity_met = {}

        for name, syn in self.syn_sampled_pre_attack.items():
            
            trf1_syn = self.invert_lbe(df = syn, col_info=self.data_col_info, data_converter=self.DataConverter)
            pre_attack_trf = pre_prc.transform(trf1_syn)
            self.fake_pre_trf[name] = pre_attack_trf

            # print("df_pre")
            # print(trf1_syn)

            # print("syn pre attack")
            # print(pre_attack_trf)

        for name, syn in self.syn_sampled.items():

            

            # print("syn")

            # print(syn)

            #tr1_train = 0
            #trf1_train = self.invert_lbe(df = syn, col_info=self.data_col_info, data_converter=self.DataConverter)
            fake_trf1 = self.invert_lbe(df = syn, col_info=self.data_col_info, data_converter=self.DataConverter)
            fake_df = pre_prc.transform(fake_trf1)

            # print("df_post")
            # print(fake_trf1)

            syn_out_preattack_trf =  self.fake_pre_trf[name]

            # print("pre")
            # print(syn_out_preattack_trf)

            # print("post")
            # print(fake_df)

            nn_obj_train = NearestNeighbors(n_neighbors=1, metric = "euclidean")
            nn_obj_train.fit(self.conv_trainset)

            nn_obj_pre = NearestNeighbors(n_neighbors=1, metric = "euclidean")
            nn_obj_pre.fit(syn_out_preattack_trf)

            

            att_train_euc_dist, att_train_nn_idx = self.find_knn(nn_obj_train, fake_df)
            att_pre_euc_dist, att_pre_nn_idx = self.find_knn(nn_obj_pre, fake_df)

            # wass_array_train = []
            # wass_array_fakepre = []

            # for idx in range(fake_df.shape[0]):
            #         space_vec = np.ravel(self.conv_trainset[att_train_nn_idx[idx],:].toarray())
            #         att_vec = np.ravel(fake_df[idx,:].toarray())

            #         wass_col = wasserstein_distance(space_vec, att_vec)
            #         wass_array_train.append(wass_col)

            # wass_train_avg = np.mean(wass_array_train)

            # for idx in range(fake_df.shape[0]):
            #         space_vec = np.ravel(syn_out_preattack_trf[att_pre_nn_idx[idx],:].toarray())
            #         att_vec = np.ravel(fake_df[idx,:].toarray())

            #         wass_col = wasserstein_distance(space_vec, att_vec)
            #         wass_array_fakepre.append(wass_col)

            # wass_fakepre_avg = np.mean(wass_array_fakepre)

            euc_train = np.mean(att_train_euc_dist)
            euc_pre = np.mean(att_pre_euc_dist)

            proximity_met[name] = {"Euclidean to Train": euc_train, "Euclidean to PreAttack": euc_pre}

        prox_df = pd.DataFrame.from_dict(proximity_met, orient="index")

        return prox_df

            

    
    def calc_susc(self):
        self.init_preprocessors(filepath_ac = self.ac_filepath, load_existing = self.load_existing_datacov)
        self.load_acs(filepath_ac = self.ac_filepath)
        self.recursive_attack()
        attack_prox = self.calculate_proximity()

        return attack_prox
        # print("Attack Proximity")
        # print(attack_prox)