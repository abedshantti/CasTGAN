
'''
Purpose of file: 

Output evaluation metrics using datasets produced from different methods: Casc-TGAN, CTGAN, cWGAN, MedGAN, VeeGAN, CTAB-GAN

Clear prints and neat output
'''

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, mean_squared_error, r2_score
from scipy.stats import pearsonr, wasserstein_distance, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from dython.nominal import associations


class models_evaluator:

    def __init__(self,
                train_set = None,
                test_set = None,
                dataset_name = "",
                categorical_cols = [],
                response_var = "",
                positive_val = None,
                pred_task = "binary_classification",
                syn_output = {},
                random_seed = None):
        super(models_evaluator, self).__init__()

        self.train_set = train_set
        self.test_set = test_set
        self.dataset_name = dataset_name
        self.categorical_cols = categorical_cols
        self.response_var = response_var
        self.positive_val = positive_val
        self.pred_task = pred_task
        self.syn_output = syn_output
        self.random_seed = random_seed
        self.dp = "{:.4f}"

        np.random.seed(self.random_seed)


#########################################################################################################################


    '''
    TSTR: 

    Train on synthetic data using three classifiers (AdaBoost, RF, SVM), evaluate on test data (Accuracy, F1-score, AUCROC)
    '''

    def tstr(self, train_set = None, test_set = None, categorical_cols = [], response_var = "", positive_val = "", pred_task = "binary_classification", syn_output = {}):

        #TODO find the responsive var if not declared

        y_train = train_set[response_var]
        y_test = test_set[response_var]

        if pred_task == "binary_classification" and (positive_val == None or positive_val == ""):
            # Auto-determine
            positive_val = y_train.value_counts().index[1]

        if pred_task == "binary_classification":
            y_train = y_train.where(y_train == positive_val, 0)
            y_train = y_train.where(y_train == 0, 1)
            y_train = y_train.astype('int')

            y_test = y_test.where(y_test == positive_val, 0)
            y_test = y_test.where(y_test == 0, 1)
            y_test = y_test.astype('int')
            
        X_train_pre = train_set.drop(response_var, axis = 1)
        X_test_pre = test_set.drop(response_var, axis = 1)

        numeric_transformer = MinMaxScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numc_cols = [col for col in X_train_pre.columns if col not in categorical_cols]
        catg_cols = [col for col in X_train_pre.columns if col in categorical_cols]

        pre_prc = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numc_cols),
                ("cat", categorical_transformer, catg_cols),
            ]
        )
        pre_prc.fit(X_train_pre)

        X_train = pre_prc.transform(X_train_pre)
        X_test = pre_prc.transform(X_test_pre)
        tstr_metrics = {}

        if pred_task ==  "binary_classification" or pred_task == "multi_classification":

            ada_clf = AdaBoostClassifier(n_estimators=100, random_state=self.random_seed)
            rf_clf  = RandomForestClassifier(random_state = self.random_seed)
            lgrg_clf = LogisticRegression(random_state = self.random_seed, max_iter = 500)
            
            ada_clf.fit(X_train, y_train)
            rf_clf.fit(X_train, y_train)
            lgrg_clf.fit(X_train, y_train)

            ada_pred = ada_clf.predict(X_test)
            dtr_pred = rf_clf.predict(X_test)
            svm_pred = lgrg_clf.predict(X_test)

            train_acc_ada = accuracy_score(y_test, ada_pred)
            train_acc_rf = accuracy_score(y_test, dtr_pred)
            train_acc_svm = accuracy_score(y_test, svm_pred)

            if pred_task == "binary_classification":
                ada_pred_proba = ada_clf.predict_proba(X_test)[:, 1]
                dtr_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
                svm_pred_proba = lgrg_clf.predict_proba(X_test)[:, 1]

                train_f1_ada = f1_score(y_test, ada_pred)
                train_f1_rf = f1_score(y_test, dtr_pred)
                train_f1_svm = f1_score(y_test, svm_pred)
            else:
                ada_pred_proba = ada_clf.predict_proba(X_test)
                dtr_pred_proba = rf_clf.predict_proba(X_test)
                svm_pred_proba = lgrg_clf.predict_proba(X_test)

                train_f1_ada = f1_score(y_test, ada_pred, average="samples")
                train_f1_rf = f1_score(y_test, dtr_pred, average="samples")
                train_f1_svm = f1_score(y_test, svm_pred, average="samples")

            train_roc_ada = roc_auc_score(y_test, ada_pred_proba)
            train_roc_rf = roc_auc_score(y_test, dtr_pred_proba)
            train_roc_svm = roc_auc_score(y_test, svm_pred_proba)

            train_aucpr_ada = average_precision_score(y_test, ada_pred_proba)
            train_aucpr_rf = average_precision_score(y_test, dtr_pred_proba)
            train_aucpr_svm = average_precision_score(y_test, svm_pred_proba)

            avg_acc = np.mean([train_acc_ada, train_acc_rf, train_acc_svm])
            avg_f1 = np.mean([train_f1_ada, train_f1_rf, train_f1_svm])
            avg_roc = np.mean([train_roc_ada, train_roc_rf, train_roc_svm])
            avg_aucpr = np.mean([train_aucpr_ada, train_aucpr_rf, train_aucpr_svm])

            # tstr_metrics["train"] = {"Ada Accuracy": train_acc_ada, "RF Accuracy": train_acc_rf, "SVM Accuracy": train_acc_svm, 
            #                         "Ada F1": train_f1_ada, "RF F1": train_f1_rf, "SVM F1": train_f1_svm,
            #                         "Ada ROCAUC": train_roc_ada, "RF ROCAUC": train_roc_rf, "SVM ROCAUC": train_roc_svm}
            

            tstr_metrics["train"] = {"Accuracy": float(self.dp.format(avg_acc)) , "ROCAUC": float(self.dp.format(avg_roc)), 
                                    "F1-score": float(self.dp.format(avg_f1)) , "PR-AUC": float(self.dp.format(avg_aucpr))}

            for name, syn_data in syn_output.items():

                y_train = syn_data[response_var]
                y_test = test_set[response_var]

                y_train = y_train.where(y_train == positive_val, 0)
                y_train = y_train.where(y_train == 0, 1)
                y_train = y_train.astype('int')
                y_test = y_test.where(y_test == positive_val, 0)
                y_test = y_test.where(y_test == 0, 1)
                y_test = y_test.astype('int')

                if len(np.unique(y_train)) == 1:
                    chng_vr = np.abs(np.unique(y_train) - 1)
                    y_train[0] = chng_vr
                    y_train = y_train.astype('int')

                X_train_pre = syn_data.drop(response_var, axis = 1)
                X_test_pre = test_set.drop(response_var, axis = 1)
                
                numeric_transformer = MinMaxScaler()
                categorical_transformer = OneHotEncoder(handle_unknown="ignore")
                numc_cols = [col for col in X_train_pre.columns if col not in categorical_cols]
                catg_cols = [col for col in X_train_pre.columns if col in categorical_cols]

                pre_prc = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, numc_cols),
                        ("cat", categorical_transformer, catg_cols),
                    ]
                )

                pre_prc.fit(X_train_pre)

                X_train = pre_prc.transform(X_train_pre)
                X_test = pre_prc.transform(X_test_pre)

                ada_clf = AdaBoostClassifier(n_estimators=100, random_state=self.random_seed)
                rf_clf  = RandomForestClassifier(random_state = self.random_seed)
                lgrg_clf = LogisticRegression(random_state = self.random_seed, max_iter = 500)

                ada_clf.fit(X_train, y_train)
                rf_clf.fit(X_train, y_train)
                lgrg_clf.fit(X_train, y_train)

                ada_pred = ada_clf.predict(X_test)
                dtr_pred = rf_clf.predict(X_test)
                svm_pred = lgrg_clf.predict(X_test)

                train_acc_ada = accuracy_score(y_test, ada_pred)
                train_acc_rf = accuracy_score(y_test, dtr_pred)
                train_acc_svm = accuracy_score(y_test, svm_pred)

                if pred_task == "binary_classification":
                    ada_pred_proba = ada_clf.predict_proba(X_test)[:, 1]
                    dtr_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
                    svm_pred_proba = lgrg_clf.predict_proba(X_test)[:, 1]

                    train_f1_ada = f1_score(y_test, ada_pred)
                    train_f1_rf = f1_score(y_test, dtr_pred)
                    train_f1_svm = f1_score(y_test, svm_pred)
                else:
                    ada_pred_proba = ada_clf.predict_proba(X_test)
                    dtr_pred_proba = rf_clf.predict_proba(X_test)
                    svm_pred_proba = lgrg_clf.predict_proba(X_test)

                    train_f1_ada = f1_score(y_test, ada_pred, average="samples")
                    train_f1_rf = f1_score(y_test, dtr_pred, average="samples")
                    train_f1_svm = f1_score(y_test, svm_pred, average="samples")


                train_roc_ada = roc_auc_score(y_test, ada_pred_proba)
                train_roc_rf = roc_auc_score(y_test, dtr_pred_proba)
                train_roc_svm = roc_auc_score(y_test, svm_pred_proba)

                train_aucpr_ada = average_precision_score(y_test, ada_pred_proba)
                train_aucpr_rf = average_precision_score(y_test, dtr_pred_proba)
                train_aucpr_svm = average_precision_score(y_test, svm_pred_proba)

                avg_acc = np.mean([train_acc_ada, train_acc_rf, train_acc_svm])
                avg_f1 = np.mean([train_f1_ada, train_f1_rf, train_f1_svm])
                avg_roc = np.mean([train_roc_ada, train_roc_rf, train_roc_svm])
                avg_aucpr = np.mean([train_aucpr_ada, train_aucpr_rf, train_aucpr_svm])

                # tstr_metrics[name] = {"Ada Accuracy": train_acc_ada, "RF Accuracy": train_acc_rf, "SVM Accuracy": train_acc_svm, 
                #                     "Ada F1": train_f1_ada, "RF F1": train_f1_rf, "SVM F1": train_f1_svm,
                #                     "Ada ROCAUC": train_roc_ada, "RF ROCAUC": train_roc_rf, "SVM ROCAUC": train_roc_svm}

                tstr_metrics[name] = {"Accuracy": float(self.dp.format(avg_acc)) , "ROCAUC": float(self.dp.format(avg_roc)), 
                                    "F1-score": float(self.dp.format(avg_f1)) , "PR-AUC": float(self.dp.format(avg_aucpr))}


        elif pred_task == "regression":

            ada_reg = AdaBoostRegressor(n_estimators=100, random_state=self.random_seed)
            dtr_reg = DecisionTreeRegressor(random_state = self.random_seed)
            svm_reg = svm.LinearSVR(random_state = self.random_seed)

            ada_reg.fit(X_train, y_train)
            dtr_reg.fit(X_train, y_train)
            svm_reg.fit(X_train, y_train)

            ada_pred = ada_reg.predict(X_test)
            dtr_pred = dtr_reg.predict(X_test)
            svm_pred = svm_reg.predict(X_test)

            train_rmse_ada = mean_squared_error(y_test, ada_pred, squared = False)
            train_rmse_dtr = mean_squared_error(y_test, dtr_pred, squared = False)
            train_rmse_svm = mean_squared_error(y_test, svm_pred, squared = False)

            train_r2_ada = r2_score(y_test, ada_pred)
            train_r2_dtr = r2_score(y_test, dtr_pred)
            train_r2_svm = r2_score(y_test, svm_pred)

            avg_rmse = np.mean([train_rmse_ada, train_rmse_dtr, train_rmse_svm])
            avg_r2 = np.mean([train_r2_ada, train_r2_dtr, train_r2_svm])

            tstr_metrics["train"] = {"RMSE": avg_rmse, "R2 Score": avg_r2}

            for name, syn_data in syn_output.items():

                y_train = syn_data[response_var]
                y_test = test_set[response_var]

                X_train_pre = syn_data.drop(response_var, axis = 1)
                X_test_pre = test_set.drop(response_var, axis = 1)
                
                numeric_transformer = MinMaxScaler()
                categorical_transformer = OneHotEncoder(handle_unknown="ignore")
                numc_cols = [col for col in X_train_pre.columns if col not in categorical_cols]
                catg_cols = [col for col in X_train_pre.columns if col in categorical_cols]

                pre_prc = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, numc_cols),
                        ("cat", categorical_transformer, catg_cols),
                    ]
                )

                pre_prc.fit(X_train_pre)

                X_train = pre_prc.transform(X_train_pre)
                X_test = pre_prc.transform(X_test_pre)

                ada_reg = AdaBoostRegressor(n_estimators=100, random_state=self.random_seed)
                dtr_reg = DecisionTreeRegressor(random_state = self.random_seed)
                svm_reg = svm.LinearSVR(random_state = self.random_seed)

                ada_reg.fit(X_train, y_train)
                dtr_reg.fit(X_train, y_train)
                svm_reg.fit(X_train, y_train)

                ada_pred = ada_reg.predict(X_test)
                dtr_pred = dtr_reg.predict(X_test)
                svm_pred = svm_reg.predict(X_test)

                train_rmse_ada = mean_squared_error(y_test, ada_pred, squared = False)
                train_rmse_dtr = mean_squared_error(y_test, dtr_pred, squared = False)
                train_rmse_svm = mean_squared_error(y_test, svm_pred, squared = False)

                train_r2_ada = r2_score(y_test, ada_pred)
                train_r2_dtr = r2_score(y_test, dtr_pred)
                train_r2_svm = r2_score(y_test, svm_pred)

                avg_rmse = np.mean([train_rmse_ada, train_rmse_dtr, train_rmse_svm])
                avg_r2 = np.mean([train_r2_ada, train_r2_dtr, train_r2_svm])

                tstr_metrics[name] = {"RMSE": float(self.dp.format(avg_rmse)), "R2 Score": float(self.dp.format(avg_r2))}
            

        else:
            return NotImplementedError("Undefined prediction task. Select one from [\"binary_classification\",\"multi_classification\",\"regression\"].")

        
        tstr_df = pd.DataFrame.from_dict(tstr_metrics, orient="index")

        return tstr_df

#########################################################################################################################

    '''
    Univariate Distributions:

    Plot and calculate the difference in univariate dists between synth and training -- also compute wasserstein distace
    '''

    def univariate_stats(self, train_set = None, categorical_cols = [], syn_output = {}):

        uni_metrics = {}
        #disp_plots = ["Dimension-wise Stats", "Categorical Distributions", "Numerical Distributions"]
        disp_plots = ["Categorical Distributions", "Numerical Distributions"]
        make_fig = None
        ax = None
        measure = "mean"
        wasserstein_method = "ordinal" # One from ["ordinal", "scaled"]
        show = False
        show_rmse = True
        show_corr = True

        numeric_transformer = MinMaxScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numeric_cols = [col for col in train_set.columns if col not in categorical_cols]

        n_num_cols = len(numeric_cols)
        n_cat_cols = len(categorical_cols)

        pre_prc = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        pre_prc.fit(train_set)
        X_real = pre_prc.transform(train_set)

        if wasserstein_method == "ordinal":

            lbe_transformer = OrdinalEncoder()
            lbe_clt = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_cols),
                    ("cat", lbe_transformer, categorical_cols)
                ]
            )
            lbe_clt.fit(train_set)
            X_real_lbe = lbe_clt.transform(train_set)

        prc_dim = X_real.shape[1]

        for name, syn in syn_output.items():

            wass_array = []
            ks_array = []

            X_fake = pre_prc.transform(syn)

            if measure in ['mean', 'avg']:
                real = np.ravel(X_real.mean(axis=0))
                fake = np.ravel(X_fake.mean(axis=0))
                plot_upper_bound = 1
            elif measure == 'std':
                real = np.ravel(X_real.std(axis=0))
                fake = np.ravel(X_fake.std(axis=0))
                plot_upper_bound = 0.6
            else:
                raise ValueError(f'"measure" must be "mean" or "std" but "{measure}" was specified.')
            

            corr_value = pearsonr(real, fake)[0]
            rmse_value = np.sqrt(mean_squared_error(real, fake))

            ### Below unused for now

            # if n_num_cols > 0:
            #     num_corr_value = pearsonr(real[:n_num_cols], fake[:n_num_cols])[0]
            #     num_rmse_value = np.sqrt(mean_squared_error(real[:n_num_cols], fake[:n_num_cols]))
            # else:
            #     num_rmse_value, num_corr_value = -1, -1

            # if X_real.shape[1] - n_num_cols > 0:
            #     cat_corr_value = pearsonr(real[n_num_cols:], fake[n_num_cols:])[0]
            #     cat_rmse_value = np.sqrt(mean_squared_error(real[n_num_cols:], fake[n_num_cols:]))
            # else:
            #     cat_rmse_value, cat_corr_value = -1, -1

            if wasserstein_method == "ordinal":
                X_fake_lbe = lbe_clt.transform(syn)
                for col_idx in range(n_num_cols + n_cat_cols):
                    real_vec = X_real_lbe[:,col_idx].copy()
                    fake_vec = X_fake_lbe[:,col_idx].copy()

                    wass_col = wasserstein_distance(real_vec, fake_vec)
                    wass_array.append(wass_col)
                    ks_col = ks_2samp(real_vec, fake_vec)[0]
                    ks_array.append(ks_col)

            elif wasserstein_method == "scaled":
                for col_idx in range(prc_dim):
                    real_vec = np.ravel(X_real[:,col_idx].toarray())
                    fake_vec = np.ravel(X_fake[:,col_idx].toarray())

                    wass_col = wasserstein_distance(real_vec, fake_vec)
                    wass_array.append(wass_col)
                    ks_col = ks_2samp(real_vec, fake_vec)[0]
                    ks_array.append(ks_col)
                    

            wass_dist = np.mean(wass_array)
            ks_metric = np.mean(ks_array)


            uni_metrics[name] = {"RMSE": float(self.dp.format(rmse_value)) , "Wasserstein": float(self.dp.format(wass_dist)), "Kolmogorov-Smirnov statistic": float(self.dp.format(ks_metric))}

            if name == "CasTGAN":

                if "Dimension-wise Stats" in disp_plots:

                    ### Dimension-wise Probability plot

                    fig, ax = plt.subplots(1)
                    fig.set_size_inches((6, 6))

                    ax.scatter(x=real, y=fake)
                    ax.plot([0, 1, 2], linestyle='--', c='black')
                    ax.set_xlabel('Real')
                    ax.set_ylabel('Fake')
                    ax.set_xlim(left=0, right=plot_upper_bound)
                    ax.set_ylim(bottom=0, top=plot_upper_bound)

                    s = ""
                    if show_rmse:
                        s += f'RMSE: {rmse_value:.4f}\n'
                    if show_corr:
                        s += f'CORR: {corr_value:.4f}\n'
                    if s != "":
                        ax.text(x=plot_upper_bound * 0.98, y=0,
                                s=s,
                                fontsize=12,
                                horizontalalignment='right',
                                verticalalignment='bottom')

                    if show:
                        plt.show()
                    else:
                        plt.close(fig)

                if "Categorical Distributions" in disp_plots and len(categorical_cols) > 0:

                    shape = None
                    log_counts = False

                    sbplt_divider = int(np.ceil(len(categorical_cols)/2))

                    if shape is None:
                        if len(categorical_cols) == 1:
                            shape = (1, 1)
                        elif len(categorical_cols) == 2:
                            shape = (1, 2)
                        elif len(categorical_cols) == 3:
                            shape = (1, 3)
                        else:
                            shape = (2, sbplt_divider)

                    #end_idx = sum([len(c) for c in ohe.categories_]) + len(num_cols)
                    X_fake_cat_df = syn[categorical_cols].copy()
                    X_fake_cat_df["type"] = "fake"
                    X_real_cat_df = train_set[categorical_cols].copy()
                    X_real_cat_df['type'] = 'real'
                    X_real_fake_cat = pd.concat([X_real_cat_df, X_fake_cat_df])
                    X_real_fake_cat.columns = categorical_cols + ['type']

                    fig, axes = plt.subplots(shape[0], shape[1])
                    fig.set_size_inches((9 * shape[0], 2 * shape[1]))
                    #fig.set_size_inches((6, 6))

                    for idx, ax in enumerate(axes.flatten()):
                        if idx < len(categorical_cols):
                            _plot = sns.countplot(x=categorical_cols[idx], hue='type',
                                                data=X_real_fake_cat, ax=ax,
                                                order=X_real_cat_df.iloc[:, idx].value_counts().index)
                            if idx > 0:
                                ax.get_legend().remove()
                            else:
                                ax.get_legend().remove()
                                ax.legend(loc=1)
                                #ax.legend(loc=1, fontsize = 16)
                                ax.get_legend().set_title(None)
                            if log_counts:
                                _plot.set_yscale("log")
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.minorticks_off()
                            #ax.xlabel(fontsize = 16)
                            ax.set_ylabel(None)
                            #ax.xaxis.label.set_size(16)
                        else:
                            ax.set_visible(False)
                    plt.tight_layout()
                    plt.savefig("Someplots/{}_cat_dist.pdf".format(self.dataset_name))
                    
                    if show:
                        plt.show()
                    else:
                        plt.close(fig)

                if "Numerical Distributions" in disp_plots and len(numeric_cols) > 0:

                    shape = None
                    subsample = False

                    X_real_num_df = train_set[numeric_cols].copy()
                    X_fake_num_df = syn[numeric_cols].copy()

                    # for col in X_real_num_df.columns:
                    #     X_fake_num_df[col] = X_fake_num_df[col].where(X_fake_num_df[col] > 0, 0)
                    #     X_real_num_df[col] = np.log(X_real_num_df[col].values + 1)
                    #     X_fake_num_df[col] = np.log(X_fake_num_df[col].values + 1)

                    # scaler = MinMaxScaler()

                    # scaler.fit(X_real_num_df)

                    # X_real_num_df = scaler.transform(X_real_num_df)
                    # X_fake_num_df = scaler.transform(X_fake_num_df)

                    sbplt_divider = int(np.ceil(len(numeric_cols)/2))

                    if shape is None:
                    # by default, we plot 3 columns with up to 2 rows
                        # if numeric_cols is not None:
                        #     rows = np.minimum(len(numeric_cols) // 3, 2)
                        # else:
                        #     rows = np.minimum(X_real_num_df.shape[1] // 3, 2)

                        # if rows == 0:
                        #     shape = (1, 1)
                        # else:
                        #     shape = (rows, 3)
                        if len(numeric_cols) == 1:
                            shape = (1, 1)
                        elif len(numeric_cols) == 2:
                            shape = (1, 2)
                        elif len(numeric_cols) == 3:
                            shape = (1, 3)
                        else:
                            shape = (2, sbplt_divider)

                    if subsample:
                        real_size = int(np.minimum(X_real_num_df.shape[0], 5e4))
                        fake_size = int(np.minimum(X_fake_num_df.shape[0], 5e4))
                    else:
                        real_size = X_real_num_df.shape[0]
                        fake_size = X_fake_num_df.shape[0]

                    fig, axes = plt.subplots(nrows=shape[0], ncols=shape[1])
                    fig.set_size_inches((9 * shape[0], 3 * shape[1]))
                    # print(fig.get_figwidth(),'< width || height >', fig.get_figheight())

                    for idx, ax in enumerate(axes.flatten()):
                        if idx < len(numeric_cols):
                            sns.kdeplot(X_real_num_df.iloc[:real_size, idx].values, label='real', ax=ax, shade=True, legend=False, bw_adjust=1)
                            sns.kdeplot(X_fake_num_df.iloc[:fake_size, idx].values, label='fake', ax=ax, shade=True, legend=False, bw_adjust=1)

                            # sns.kdeplot(X_real_num_df[:real_size, idx], label='real', ax=ax, shade=True, legend=False, bw_adjust=0.02)
                            # sns.kdeplot(X_fake_num_df[:fake_size, idx], label='fake', ax=ax, shade=True, legend=False, bw_adjust=0.02)


                            min_val = np.min((X_real_num_df.iloc[:real_size, idx].values.min(), X_fake_num_df.iloc[:fake_size, idx].values.min()))
                            max_val = np.max((X_real_num_df.iloc[:real_size, idx].values.max(), X_fake_num_df.iloc[:fake_size, idx].values.max()))
                            ax.set_yticks([])
                            ax.set_xticks([min_val, max_val])
                            if numeric_cols is not None:
                                ax.set_xlabel(numeric_cols[idx], labelpad=-10)
                        else:
                            ax.set_visible(False)
                    axes.flatten()[0].legend()
                    plt.tight_layout()

                    plt.savefig("Someplots/{}_num_dist.pdf".format(self.dataset_name))

                    if show:
                        plt.show()
                    else:
                        plt.close(fig)


        
        return uni_metrics

        
        #return rmse_value, corr_value, num_rmse_value, num_corr_value, cat_rmse_value, cat_corr_value

            # upper_bound = np.maximum(np.max(real) * 1.1, np.max(fake) * 1.1)
            # upper_bound = np.minimum(1, upper_bound)

            # if measure in ['mean', 'avg']:
            #     upper_bound = 1
            # else:
            #     upper_bound = 0.6

            # ax.scatter(x=real, y=fake)
            # ax.plot([0, 1, 2], linestyle='--', c='black')
            # ax.set_xlabel('Real')
            # ax.set_ylabel('Fake')
            # ax.set_xlim(left=0, right=upper_bound)
            # ax.set_ylim(bottom=0, top=upper_bound)

            # corr_value = pearsonr(real, fake)[0]
            # rmse_value = np.sqrt(mean_squared_error(real, fake))

            # s = ""
            # if show_rmse:
            #     s += f'RMSE: {rmse_value:.4f}\n'
            # if show_corr:
            #     s += f'CORR: {corr_value:.4f}\n'
            # if s != "":
            #     ax.text(x=upper_bound * 0.98, y=0,
            #             s=s,
            #             fontsize=12,
            #             horizontalalignment='right',
            #             verticalalignment='bottom')

            # if show:
            #     plt.show()

        #return rmse_value, corr_value

#########################################################################################################################


    '''
    Find name

    Correlation map divided by the number of unique pairwise categorical combinations
    '''

    def correlation_pairwise_comb(self, train_set = None, categorical_cols = [], syn_output = {}):

        fn_dict = {}

        train_set_catg = train_set[categorical_cols].copy()

        real_unq_prws = 0
        consumed_real = []

        for col in train_set_catg.columns:

            consumed_real.append(col)
            others_real = [x for x in train_set_catg.columns if x not in consumed_real]
            for inner_col in others_real:
                real_unq_prws += train_set_catg.groupby(col)[inner_col].nunique().sum()

        real_corrs = associations(train_set, compute_only = True)['corr']
        mat_dim = real_corrs.shape[0]

        fn_dict["Real Pairwise Unique Combinations"] = real_unq_prws

        side_dims = []

        for idx in range(mat_dim-1):
            added_nums = [z + 1 for z in range(mat_dim-1)]
            side_dims.append(added_nums[idx::])

        for name, syn in syn_output.items():

            fake_set_catg = syn[categorical_cols].copy()

            fake_unq_prws = 0
            consumed_fake = []

            for col in fake_set_catg.columns:

                consumed_fake.append(col)
                others_fake = [x for x in fake_set_catg.columns if x not in consumed_fake]
                for inner_col in others_fake:
                    fake_unq_prws += fake_set_catg.groupby(col)[inner_col].nunique().sum()

            fake_corrs = associations(syn, compute_only = True)['corr']

            corrs_sum = 0
            mat_elements_size = 0

            for a in range(mat_dim-1):
                side = side_dims[a]
                for b in side:
                    mat_elements_size += 1
                    real_corr_x = real_corrs.iloc[a, b]
                    fake_corr_x = fake_corrs.iloc[a, b]

                    corr_sq_diff = (fake_corr_x - real_corr_x) ** 2
                    corrs_sum += corr_sq_diff

            corr_rmse = np.sqrt(corrs_sum / mat_elements_size)

            invalidity_score = corr_rmse / (fake_unq_prws / real_unq_prws)

            #fn_dict[name] = {"Fake Pairwise Unique Combinations": fake_unq_prws, "Correlation RMSE": corr_rmse}
            fn_dict[name] = {"Fake Pairwise Unique Combinations": fake_unq_prws, "Correlation RMSE": float(self.dp.format(corr_rmse)), "Invalidity Score": float(self.dp.format(invalidity_score))}

        return fn_dict


#########################################################################################################################

    '''
    Privacy attacks

    classification of successful full black box attacks
    '''

    def find_knn(self, nn_obj, query_imgs, K):
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
        dist_batch, idx_batch = nn_obj.kneighbors(X=query_imgs, n_neighbors=K)
        dist.append(dist_batch)
        idx.append(idx_batch)

        try:
            dist = np.concatenate(dist)
            idx = np.concatenate(idx)
        except:
            dist = np.array(dist)
            idx = np.array(idx)
        return dist, idx

    def black_box_privacy_attacks(self, train_set = None, test_set = None, categorical_cols = [], syn_output = {}):

        fbb_output = {}

        K = 3

        if len(train_set) > len(test_set):
            train_set = train_set.sample(n = len(test_set), random_state = self.random_seed)
        elif len(train_set) < len(test_set):
            test_set = test_set.sample(n = len(train_set), random_state = self.random_seed)

        numeric_transformer = MinMaxScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numc_cols = [col for col in train_set.columns if col not in categorical_cols]
        catg_cols = [col for col in train_set.columns if col in categorical_cols]

        pre_prc = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numc_cols),
                ("cat", categorical_transformer, catg_cols),
            ]
        )
        pre_prc.fit(train_set)

        train_df = pre_prc.transform(train_set)
        test_df = pre_prc.transform(test_set)

        for name, syn in syn_output.items():
            
            fake_df = pre_prc.transform(syn)

            nn_obj = NearestNeighbors(n_neighbors=K)
            nn_obj.fit(fake_df)

            pos_loss, pos_idx = self.find_knn(nn_obj, train_df, K)
            neg_loss, neg_idx = self.find_knn(nn_obj, test_df, K)

            pos_results = pos_loss[:,0]
            neg_results = neg_loss[:,0]
            labels = np.concatenate((np.zeros((len(neg_results),)), np.ones((len(pos_results),))))
            results = np.concatenate((-neg_results, -pos_results))
            auc = roc_auc_score(labels, results)
            ap = average_precision_score(labels, results)

            fbb_output[name] = {"FBB ROCAUC": float(self.dp.format(auc)), "FBB PR-AUC": float(self.dp.format(ap))}

        return fbb_output





#########################################################################################################################

    def compute_metrics(self):
        
        eval_tstr = self.tstr(train_set=self.train_set, test_set = self.test_set, categorical_cols = self.categorical_cols, 
                                response_var = self.response_var, positive_val = self.positive_val, pred_task = self.pred_task, syn_output = self.syn_output)

        print("eval_tstr")
        print(eval_tstr)
        print("")

        univariate = self.univariate_stats(train_set = self.train_set, categorical_cols = self.categorical_cols, syn_output = self.syn_output)

        print("univariate")
        print(univariate)
        print("")

        corr_prws = self.correlation_pairwise_comb(train_set = self.train_set, categorical_cols = self.categorical_cols, syn_output = self.syn_output)

        print("corr_pairwise")
        print(corr_prws)
        print("")

        return eval_tstr, univariate, corr_prws


        #fbb_attacks = self.black_box_privacy_attacks(train_set = self.train_set, test_set = self.test_set, categorical_cols = self.categorical_cols, syn_output = self.syn_output)

        # print("black_box attacks")
        # print(fbb_attacks)

