"""demo of the IMRPhenomD module C 2021 Matthew Digman"""
from time import perf_counter

import numpy as np

from IMRPhenomD import AmpPhaseFDWaveform,IMRPhenomDGenerateh22FDAmpPhase
import IMRPhenomD_const as imrc

if __name__=='__main__':

    t_start = perf_counter()
    NF = 16384*10

    distance = 56.00578366287752*1.0e9*imrc.PC_SI/imrc.CLIGHT
    chi1 = 0.7534821857057837
    chi2 = 0.6215875279643664

    m1_sec = 2599137.035*imrc.MTSUN_SI
    m2_sec = 1242860.685*imrc.MTSUN_SI

    m1 = m1_sec/imrc.MTSUN_SI
    m2 = m2_sec/imrc.MTSUN_SI

    Mt_sec = m1_sec+m2_sec
    Mt = Mt_sec/imrc.MTSUN_SI

    tc = 2.496000e+07
    DF = 0.99*imrc.f_CUT/Mt_sec/NF


    Mt_sec = Mt*imrc.MTSUN_SI
    eta = m1_sec*m2_sec/Mt_sec**2
    Mc = eta**(3/5)*Mt_sec
    assert np.isclose(eta,(Mc/Mt_sec)**(5/3))


    m1_SI =  m1*imrc.MSUN_SI
    m2_SI =  m2*imrc.MSUN_SI

    freq = np.arange(1,NF+1)*DF

    chis = (chi1+chi2)/2
    chia = (chi1-chi2)/2
    phic = 2.848705/2
    FI = 3.4956509169372e-05

    MfRef_in = FI*Mt_sec

    amp_imr = np.zeros(NF)
    phase_imr = np.zeros(NF)
    if imrc.findT:
        time_imr = np.zeros(NF)
        timep_imr = np.zeros(NF)
    else:
        time_imr = np.zeros(0)
        timep_imr = np.zeros(0)


    t0 = perf_counter()
    h22 = AmpPhaseFDWaveform(NF,freq,amp_imr,phase_imr,time_imr,timep_imr,0.,0.)
    h22 = IMRPhenomDGenerateh22FDAmpPhase(h22,freq,phic,MfRef_in,m1_SI,m2_SI,chi1,chi2,distance*imrc.CLIGHT)
    tf = perf_counter()
    print("compiled in %10.7f seconds"%(tf-t0))


    t0 = perf_counter()
    n_run = 100
    for itrm in range(0,n_run):
        IMRPhenomDGenerateh22FDAmpPhase(h22,freq,phic,MfRef_in,m1_SI,m2_SI,chi1,chi2,distance*imrc.CLIGHT)
    tf = perf_counter()
    print("run      in %10.7f seconds"%((tf-t0)/n_run))


    import matplotlib.pyplot as plt
    plt.loglog(freq,2.*np.sqrt(5./(64.*np.pi))*h22.amp)
    plt.show()
    plt.loglog(freq,np.abs(h22.phase))
    plt.show()
    plt.loglog(freq,np.abs(h22.time))
    plt.show()
    plt.loglog(freq,np.abs(h22.timep))
    plt.show()
