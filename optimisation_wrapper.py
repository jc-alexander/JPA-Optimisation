import numpy as np
from functions import get_current
from bayesian_optimisation_JPA import run_optimisation

"""
Run JPA current tuning plot and import data below
"""

fname_phase = '2019_11_18_1155_Leela_JPA_PumplineJPA_Tuning_phases.txt'
fname_freq = '2019_11_18_1155_Leela_JPA_PumplineJPA_Tuning_freqs.txt'
fname_current = '2019_11_18_1155_Leela_JPA_PumplineJPA_Tuning_current.txt'
res_freq = 6.75643e9

#Calculate required current and determine current range
current = get_current(res_freq,fname_phase,fname_freq,fname_current,plot=True)  #mA
current_range = [current - 0.5, current + 0.5] #mA

JPA_freq = #Pump Frequency
JPA_freq_range = [JPA_freq - 100e6, JPA_freq + 100e6] #Hz

JPA_power =
JPA_power_range = [9, 17] #dB

'''
Put three parameters with ranges into optimiser
'''

JPA_params,JPA_SNR = run_optimisation(current_range,JPA_freq_range,JPA_power_range)
