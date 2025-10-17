from mimetypes import suffix_map
from sys import path_importer_cache
from turtle import st
import torch
from torch import kl_div, nn
import time
import numpy as np
from Model.NeuralNetwork import NeuralNetwork
from Prep import DataUtils, DataPreparer
import Output
from Model.StandardModel import StandardModel
from scipy.stats import pearsonr, zscore
import Output.StatOutput as StatOutput
import Tests.RatioExemplar as RE
import Output.PCAOutput as PCAOutput

### globals ---

ALT = False
SOL = True

SUF = { (False, False): "",
        (False, True): "_sol",
        (True, False): "_alt",
        (True, True): "_alt_sol" }[(ALT, SOL)]

### DataUtils.convert_python_lists_to_csv("Data/Current/stimList_gencat_hydra_alt-lat.py" if ALT else "Data/Current/stimList_gencat_hydra.py", f"Data/Current/stimList_gencat_hydra{SUF}.csv", num_features=11, alt=ALT, sol=SOL)

DATA_FILENAME = f"Data/Current/stimList_gencat_hydra{SUF}.csv"
MODULAR_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-mod.csv"
LATTICE_P_M_FILENAME = "Data/ReferenceMatrices/cooc-jaccard-lat.csv" if not ALT else "Data/ReferenceMatrices/cooc-jaccard-lat-alt.csv"

NUM_FEATURES = 11
HIDDEN_LAYER_SIZE = 400
NUM_EPOCHS = 60
NUM_TRAINING_TRIALS = 96

INCLUDE_E0 = True

DATA_DIR = "Results/Data/Focused_04/by_sets/one_h" if not (ALT or SOL) else f"Results/Data/Focused_04/{SUF[1:]}"

DATA_PARAMS = { "losses": True,
                "hidden_activations": True,
                "m_output_corrs": True,
                "l_output_corrs": True,
                "m_hidden_corrs": True,
                "l_hidden_corrs": True,
                "output_matrices": True,
                "hidden_matrices": True,
                "output_ratio_tests": True,
                "hidden_ratio_tests": True,
                "output_activation_exemplar_tests": True,
                "output_activation_onehot_tests": True}

CORR_DATA_PARAMS = { "losses": True,
                "hidden_activations": False,
                "m_output_corrs": True,
                "l_output_corrs": True,
                "m_hidden_corrs": True,
                "l_hidden_corrs": True,
                "output_matrices": False,
                "hidden_matrices": False,
                "output_ratio_tests": False,
                "hidden_ratio_tests": False,
                "output_activation_exemplar_tests": False,
                "hidden_activation_exemplar_tests": False,
                "output_activation_onehot_tests": False,
                "hidden_activation_onehot_tests": False }

ANALYSIS_DIR = f"Results/Analysis/Plots/one_h/original" if not (ALT or SOL) else f"Results/Analysis/Plots/one_h/{SUF[1:]}"

### -----------

modular_reference_matrix, lattice_reference_matrix = DataUtils.get_probability_matrices_m_l(MODULAR_P_M_FILENAME, LATTICE_P_M_FILENAME);
dataloader = DataPreparer.get_dataloader(DataUtils.load_csv_data(DATA_FILENAME, NUM_FEATURES))


for i in range(1, 51):
    model = StandardModel(num_features=NUM_FEATURES, hidden_layer_size=HIDDEN_LAYER_SIZE, batch_size=NUM_TRAINING_TRIALS, num_epochs=NUM_EPOCHS, learning_rate=.04, loss_fn=nn.BCEWithLogitsLoss())
    results = model.train_eval_test_P(dataloader, modular_reference_matrix, lattice_reference_matrix, DATA_PARAMS, include_e0=INCLUDE_E0)
    np.savez(f"{DATA_DIR}/p_m{i}.npz", **results)
    print(i)


StatOutput.plot_stats_with_confidence_intervals(lr_str="4_50m", data_dir=DATA_DIR, save_dir=f"{ANALYSIS_DIR}/Correlations", data_parameters=CORR_DATA_PARAMS, include_e0=INCLUDE_E0)
StatOutput.plot_33s(data_dir=DATA_DIR, save_dir=f"{ANALYSIS_DIR}/3-3", include_e0=INCLUDE_E0)
StatOutput.plot_s_curve(data_dir=DATA_DIR, save_dir=f"{ANALYSIS_DIR}/S-Curves", epoch=15, include_e0=INCLUDE_E0)
"""
StatOutput.plot_activation_tests_with_confidence_intervals(lr_str="4_50m", data_dir=DATA_DIR, save_dir="Results/Analysis/Plots/one_h/Activations", include_e0=INCLUDE_E0)

StatOutput.plot_difference_stats_with_confidence_intervals(lr_str="4_50m", data_dir=DATA_DIR, save_dir="Results/Analysis/Plots/one_h/Diffs", data_parameters=CORR_DATA_PARAMS, include_e0=INCLUDE_E0)


StatOutput.plot_s_curve(data_dir=DATA_DIR, save_dir="Results/Analysis/Plots/one_h/S-Curves", epoch=15, include_e0=INCLUDE_E0)

StatOutput.plot_33s(data_dir=DATA_DIR, hidden=False, save_dir="Results/Analysis/Plots/one_h/3-3", include_e0=INCLUDE_E0)

for i in range(0, 61):
    StatOutput.plot_scatter_models(data_dir=DATA_DIR, epoch=i, save_dir="Results/Analysis/Plots/one_h/scatter_models", include_e0=INCLUDE_E0)

#StatOutput.plot_structure_learning_vs_generalization(data_dir=DATA_DIR, epoch=15, save_dir="Results/Analysis/Plots/one_h/structure_learning-generalization", include_e0=INCLUDE_E0)
#StatOutput.plot_generalization_vs_category_strength(data_dir=DATA_DIR, epoch=15, save_dir="Results/Analysis/Plots/one_h/generalization-category_strength", include_e0=INCLUDE_E0)
#StatOutput.plot_structure_learning_vs_category_strength(data_dir=DATA_DIR, epoch=15, save_dir="Results/Analysis/Plots/one_h/category_strength-structure_learning", include_e0=INCLUDE_E0)


PCAOutput.plot_generalization_vs_dimensionality_diff(data_dir=DATA_DIR, epoch=15, save_dir="Results/Analysis/Plots/one_h/alt/generalization_vs_dimensionality_diff", include_e0=INCLUDE_E0)
one_h
PCAOutput.plot_k95_over_epochs(data_dir=DATA_DIR, save_dir="Results/Analysis/Plots/one_h/alt/PCA/k95_over_epochs", num_features=NUM_FEATURES, include_e0=INCLUDE_E0)

PCAOutput.plot_k95_bars_epoch(data_dir=DATA_DIR, epoch=15, save_dir="Results/Analysis/Plots//alt/PCA/k95_bars", num_features=NUM_FEATURES, include_e0=INCLUDE_E0)
"""

PCAOutput.plot_k95_bars_epoch(data_dir=DATA_DIR, epoch=15, save_dir=f"{ANALYSIS_DIR}/PCA/k95_bars", num_features=NUM_FEATURES, include_e0=INCLUDE_E0)