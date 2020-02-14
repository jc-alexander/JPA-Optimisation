import numpy as np
import scipy
from scipy import signal
from scipy.optimize import curve_fit
from scipy import interpolate
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


'''
def get_SNR(title = "B = %.3fmT, f = %.3fMHz"%(i3d.field()*1e3,sgs.frequency()*1e-6),save = False,extra_name = "",plot=True):

    SNR_I,SNR_Q,SNR_mag,SNR_I_avg,SNR_Q_avg,SNR_mag_avg = [],[],[],[],[],[]
    Pi =10e-6
    SR = 1.28e9/(2**5)
    wfg.sample_rate = SR
    pulse_amplitude = 0.05
    phase =0
    w = np.pi*2*1e6*0
    gate_wait = 50e-9
    frequency = sgs.frequency.get()

    ##############################################################
    #Define and load all fixed pulses and waveforms here:
    spectr.awg.ClearMemory()
    d1 = wfg.heterodyne_delay(70e-6,0)
    d2 = wfg.heterodyne_delay(5000e-6,0)
    piebytwo = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase)
    piebytwo_negative = wfg.heterodyne_gaussian_pulse(pulse_amplitude,Pi,phase+180)

    ###############################################################
    #Echo detection waveform - positive piebytwo

    I_in_pos,Q_in_pos = wfg.heterodyne_combine(d1,piebytwo,d2,w = w,t0=0)

    gate_len = len(I_in_pos)/SR + 100e-9
    acq_wait = 100e-9

    marker_gate = wfg.combine(wfg.delay(gate_wait),wfg.pulse(1,gate_len))
    marker_acq = wfg.combine(wfg.delay(acq_wait),wfg.pulse(1,300e-6))
    marker_digitizer = marker_acq

    I_in_pos, Q_in_pos, marker_gate, marker_acq, marker_digitizer = wfg.length_correction(I_in_pos, Q_in_pos, marker_gate, marker_acq, marker_digitizer)
    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)

    #Echo detection waveform - negative piebytwo

    I_in_neg,Q_in_neg = wfg.heterodyne_combine(d1,piebytwo_negative,w = w,t0=0)
    I_in_neg, Q_in_neg, marker_gate, marker_acq = wfg.length_correction(I_in_neg, Q_in_neg, marker_gate, marker_acq)
    markersArray = spectr.awg.generate_MarkersArray(marker_gate,marker_acq,marker_digitizer)
    ###############################################################
    #Set AWG parameters
    Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration = 0.4,0, 'continuous',1,'waveform','single-ended'
    spectr.awg.ch12.set_config(SR, Gain, Offset, OperationMode, BurstCount, ChannelMode,TerminalConfiguration)
    # Load waveforms
    print('loading...')

    spectr.awg.ch12.load_waveform('In_pos',
                              np.concatenate((I_in_pos,Q_in_pos)),
                              np.concatenate((markersArray,markersArray)))
    spectr.awg.ch12.load_waveform('In_neg',
                              np.concatenate((I_in_neg,Q_in_neg)),
                              np.concatenate((markersArray,markersArray)))
    print('loaded')

    spectr.awg.ch12.create_sequence('fake_echo_pos',['In_pos'],[1])
    spectr.awg.ch12.create_sequence('fake_echo_neg',['In_neg'],[1])

    # Run sequences: positive and negative for phase cycling
    start_time = time()
    spectr.awg.ch12.init_channel('fake_echo_neg')
    n_I, n_Q, timeI, timeQ = spectr.IQ_data_raw()
    print('Time elapsed (s):',time()-start_time)

    start_time = time()

    spectr.awg.ch12.init_channel('fake_echo_pos')
    p_I, p_Q, timeI, timeQ = spectr.IQ_data_raw()
    print('Time elapsed (s):',time()-start_time)

    window = 12

    raw_I = np.subtract(p_I, n_I)
    raw_Q = np.subtract(p_Q, n_Q)
    raw_mag = (raw_I**2+raw_Q**2)**0.5

    PhasedI = np.mean(raw_I,axis = 0)
    PhasedQ = np.mean(raw_Q,axis = 0)
    Phasedmag = np.mean(raw_mag,axis = 0)

    # Redefine mag to zero the noise floor
    raw_mag = raw_mag-Phasedmag[10*window:window*250].mean(axis = 0)
    Phasedmag = np.mean(raw_mag,axis = 0)

    for n in range(len(raw_I)-1):
        SNR_I.append((raw_I[n+1][window*310:window*365]**2).mean()/(raw_I[n+1][:window*250]**2).mean())
        SNR_I_avg.append((raw_I[:n+1].mean(axis = 0)[window*310:window*365]**2).mean()/(raw_I[:n+1].mean(axis = 0)[:window*250]**2).mean())

        SNR_Q.append((raw_Q[n+1][window*310:window*365]**2).mean()/(raw_Q[n+1][:window*250]**2).mean())
        SNR_Q_avg.append((raw_Q[:n+1].mean(axis = 0)[window*310:window*365]**2).mean()/(raw_Q[:n+1].mean(axis = 0)[:window*250]**2).mean())

        SNR_mag.append((raw_mag[n+1][window*310:window*365]**2).mean()/(raw_mag[n+1][:window*250]**2).mean())
        SNR_mag_avg.append((raw_mag[:n+1].mean(axis = 0)[window*310:window*365]**2).mean()/(raw_mag[:n+1].mean(axis = 0)[:window*250]**2).mean())

    #Demodulate from intermediate carrier frequency
    t_demod = np.add(wfg.time(PhasedI, spectr.dig.SampleRate()),acq_wait)
    I_demod, Q_demod = wfg.signal_demod(PhasedI, PhasedQ,t_demod, w)

    #Downsample and average

    t = np.multiply(downsample(timeI,window),1e9)
    I = np.array(downsample(I_demod,window))
    Q = np.array(downsample(Q_demod,window))
    mag = np.array(downsample(Phasedmag,window))

    if plot ==True:
        plot_IQmag(t,I,Q,mag,title1 = 'B = %.3fmT, f = %.3fMHz'%(i3d.field()*1e3,sgs.frequency()*1e-6))

        plt.figure(figsize = (7,5))

        shots = np.linspace(1,len(SNR_mag),len(SNR_mag))

        plt.plot(shots,SNR_mag,"o",label = "shot")
        plt.plot(shots,SNR_mag_avg,"o",label = "cumulative average")

        #fit
        #func = lambda y,x: y[0]*x
        #est,fine,data_fit = fit_function([20],func,shots,SNR_mag_avg)
        #plt.plot(np.linspace(0,len(SNR_mag)+1,len(SNR_mag)+2),func(est,np.linspace(0,len(SNR_mag)+1,len(SNR_mag)+2)),label = "fit")

        plt.plot(np.linspace(0,len(SNR_mag)+1,len(SNR_mag)),np.median(SNR_mag)*np.ones(len(shots)),label = "median SNR = %.1f"%np.median(SNR_mag))

        plt.ylabel("SNR")
        plt.xlabel("Shots")
        #plt.xticks(np.linspace(0,20,11))
        plt.ylim([0,None])
        #plt.ylim([0,10*np.median(SNR_mag)])
        plt.xlim([0,len(SNR_mag)])
        plt.legend()
        plt.title(extra_name+title)
        plt.tight_layout()

        if save ==True:
            current_field = (ix.field()**2+iy.field()**2+iz.field()**2)**0.5
            filename = folder+"\\"+name+extra_name+"%.2fmT%.3fMHz_fakeecho_SNR"%   (current_field*1e3,sgs.frequency()*1e-6)
            plt.savefig(filename+".pdf")
            np.savetxt(filename+".txt",np.transpose([SNR_I,SNR_Q,SNR_mag,SNR_I_avg,SNR_Q_avg,SNR_mag_avg]))
        plt.show()

    return(SNR_I,SNR_Q,SNR_mag,SNR_I_avg,SNR_Q_avg,SNR_mag_avg)
'''
def get_current(res_frequency,phase_fname,freqs_fname,current_fname,plot=True):

    phase = np.loadtxt(phase_fname)
    freq = np.loadtxt(freqs_fname)
    current = np.loadtxt(current_fname)

    #Remove Background and Downsample to help find current curve
    phase = scipy.signal.detrend(phase)
    phase = scipy.signal.decimate(phase,10)

    #Remove background between sweeps
    corrected_phase = np.zeros((np.size(phase,0),np.size(phase,1)))
    i = 0
    for sweep in phase:
        corrected_phase[i][:] = sweep - np.mean(phase,0)
        i+=1

    #To find working current find the points of minimum gradient
    grad_array = []
    for sweep in corrected_phase:
        grad = len(sweep) - np.argmin(np.gradient(sweep))
        grad_array.append(grad)

    if plot == True:
        #Plot original data with current curve fitted
        phase = np.loadtxt(phase_fname)
        freq = np.loadtxt(freqs_fname)
        current = np.loadtxt(current_fname)

        phase = scipy.signal.detrend(phase)
        data = pd.DataFrame(phase)
        data.columns=np.round(list(freq*1e-9),3)
        data.index = np.round(list(current),4)
        data = data.transpose()
        data = data.iloc[::-1]
        plt.figure(figsize=(8,5))
        sns.heatmap(data,cmap=sns.diverging_palette(240, 12, n=10000))
        plt.plot(np.multiply(grad_array,10)-10,linewidth=2,color='lime')
        plt.xlabel("Current (mA)")
        plt.ylabel("Frequency (GHz)")
        plt.savefig("JPA_tuning.png",bbox_inches="tight")

    #Only choose one half of freq data so that it is not a one-to-many mapping
    opt_freqs = freq[-np.multiply(grad_array,10)+10]
    opt_freqs_positive = opt_freqs[int(len(current)/2)::]
    current_positive = current[int(len(current)/2)::]
    #interpolate to allow for arbitrary input
    opt_frequency = interpolate.interp1d(np.array(current_positive[:-1]), np.array(opt_freqs_positive[:-1]))
    opt_current = interpolate.interp1d( np.array(opt_freqs_positive[:-1]),np.array(current_positive[:-1]))
    #functions to put into correct format returns GHz and mA
    def calc_frequency(curr):
        return float(opt_frequency(curr*1e-3))*1e-9
    def calc_current(freq):
        return float(opt_current(freq))*1e3

    return calc_current(res_frequency)
