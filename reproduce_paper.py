"""Reproducing simulations and plots in the paper.

Seeds for the simulations were not provided, thus reproduced data might differ
*slightly*. The qualitative results should remain the same because multiple
training and inference iterations were used.

SiO_x data will be downloaded automatically. To get access to Ta/HfO2 data,
please email [Dovydas Joksas](mailto:dovydas.joksas.15@ucl.ac.uk).
"""
from awarememristor import simulations
from awarememristor.plotting import figures, supporting_figures

# Simulations
simulations.ideal.main()
simulations.iv_nonlinearity.main()
simulations.iv_nonlinearity_and_stuck_on.main()
simulations.iv_nonlinearity_cnn.main()
simulations.stuck_distribution.main()
simulations.stuck_off.main()
simulations.high_d2d.main()
simulations.weight_implementation.main()
simulations.nonideality_agnosticism.main()

# Figures in the main text
figures.SiO_x()
figures.Ta_HfO2()
figures.iv_nonlinearity_training()
figures.iv_nonlinearity_inference()
figures.iv_nonlinearity_cnn()
figures.weight_implementation()
figures.nonideality_agnosticism()
figures.pf_residuals()

# Figures in the Supporting Information
supporting_figures.all_iv_curves_full_range()
supporting_figures.switching()
supporting_figures.iv_nonlinearity_training()
supporting_figures.weight_implementation_standard_weights_training()
supporting_figures.weight_implementation_double_weights_training()
supporting_figures.high_d2d_training()
supporting_figures.stuck_off_training()
supporting_figures.high_iv_nonlinearity_and_stuck_on_training()
supporting_figures.stuck_distribution_training()
supporting_figures.pf_plots()
supporting_figures.pf_fits()
