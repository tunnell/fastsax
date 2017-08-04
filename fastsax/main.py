# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np

from fastsax.helpers import initial_conditions, SaxUnphysicalModel
from fastsax.microphysics import recombination, quanta, heat_quenching


def excimer_ion_distributing(df, c):
    """Step 4: Excimer-ion distributing

#         int ExcimerNum = gpu_binomial(&s, QuantaNum, ExIonRatio / (1.+ExIonRatio) );
#         int IonNum = QuantaNum - ExcimerNum;
#         if (PenningQuenching<0) {physical_process_flag = 0; break;}  # This check done earlier
#         else if(PenningQuenching<1) ExcimerNum = gpu_binomial(&s, ExcimerNum, PenningQuenching);

    :param df:
    :param c:
    :return:
    """
    df['ExcimerNum'] = np.random.binomial(df['QuantaNum'],
                                          c['Nex/Ni'] / (1. + c['Nex/Ni']),
                                          size=len(df)).astype(np.int32)
    df['IonNum'] = df['QuantaNum'] - df['ExcimerNum']

    # Just operate on values where PenningQuenching < 1
    selection = df['PenningQuenching'] < 1
    df_temp = df[selection]
    df.loc[selection,
           'ExcimerNum'] = np.random.binomial(df_temp['ExcimerNum'],
                                              df_temp['PenningQuenching'],
                                              size=len(df_temp)).astype(np.int32)

    return df


# ## 5) Recombination fluctuation
# float TrueRecombFrac = RecombFrac;
# // we don't allow detla_r to be smaller than zero
# if (DeltaR<0) {physical_process_flag = 0; break;}
# TrueRecombFrac = np.random.normal
# #gpu_truncated_gaussian(&s, RecombFrac, DeltaR, 0, 1);

# In[13]:

def recombination_fluctuation(df, c):
    if (df['DeltaR'] < 0).any():
        raise SaxUnphysicalModel("DeltaR < 0")

    df['TrueRecombFrac'] = np.clip(np.random.normal(df['RecombFrac'],
                                                    df['DeltaR'],
                                                    size=len(df)),
                                   0,
                                   1)
    return df


def photon_electron_number(df, c):
    """
    # // 6) Photon and Electron number
# int ElectronNum = gpu_binomial(&s, IonNum, 1.-TrueRecombFrac);
#
# int PhotonNum = IonNum + ExcimerNum - ElectronNum;
"""
    df['ElectronNum'] = np.random.binomial(df['IonNum'],
                                           1 - df['TrueRecombFrac'],
                                           size=len(df)).astype(np.int32)
    df['PhotonNum'] = df['IonNum'] + df['ExcimerNum'] - df['ElectronNum']
    return df


def s1_s2_inversed_correction_factor(df, c):
    """S1&S2 inversed correction factor

    TODO
    # // 7)
#         //    map is a function of reconstructed position
#         float InversedS1CorrectionFactor = 1.;
#         get_value_from_array(s1_correction_map, Energy, RecX, RecY, Z, &InversedS1CorrectionFactor);
#         float InversedS2CorrectionFactor = 1.;
#         get_value_from_array(s2_correction_map, Energy, RecX, RecY, Z, &InversedS2CorrectionFactor);
#         if (n==0) MajorS1InversedCorrectionFactor = InversedS1CorrectionFactor;

    :param df:
    :param c:
    :return:
    """
    # Skip correction
    return df


def electron_lifetime_prob(df, c):
    """
    # // 8) Electron lifetime
# float ElectronSurviveProb = expf( -fabsf(Z) / DriftVelocity / ElectronLifetime );
#

    :param df:
    :param c:
    :return:
    """
    df['ElectronSurviveProb'] = np.exp(df['z'] / (c['drift_velocity'] * c['setup']['electron_lifetime']))
    return df


# In[17]:

def photon_detection(df, c):
    """
    # # 9) Photon detection
#
#     float TrueG1 = g1 * InversedS1CorrectionFactor / (1+DPEFraction);
#     if(TrueG1<=0) TrueG1=0;
#     else if(TrueG1>1) TrueG1=1.;
#     int HitPhotonNum = gpu_binomial(&s, PhotonNum, TrueG1);
#     Hits[n+1]=(float)HitPhotonNum;
#

    :param df:
    :param c:
    :return:
    """
    df['HitPhotonNum'] = np.random.binomial(df['PhotonNum'],
                                            c['g1'],
                                            size=len(df))
    return df


def dpe_probability(df, c):
    """
    # # 10) DPE probability
#         int DPENum = gpu_binomial(&s, HitPhotonNum, DPEFraction);
#         int PENum = HitPhotonNum + DPENum;

    :param df:
    :param c:
    :return:
    """
    df['PENum'] = df['HitPhotonNum'] + np.random.binomial(df['HitPhotonNum'],
                                                          c['p_dpe'],
                                                          size=len(df))
    return df


def electron_detection(df, c):
    """Step 11: Electron detection

    # # 11) Electron detection
#         float TrueG2 = g2 * InversedS2CorrectionFactor;
#         float TrueGasGain = TrueG2 / ExtractionEfficiency;
#         float TrueSEResolution = GasGainRes;
#         float S2WallSurviveProb = 1.;
#         get_value_from_array(s2_loss_map, Energy, RecX, RecY, Z, &S2WallSurviveProb); // wall loss
#         int DriftedElectronNum = gpu_binomial( &s, ElectronNum, ElectronSurviveProb*S2WallSurviveProb);
#         int ExtractedElectronNum = gpu_binomial( &s, DriftedElectronNum, ExtractionEfficiency );
#         float uS2 = curand_normal(&s) *sqrtf( ((float)ExtractedElectronNum) )* TrueGasGain * TrueSEResolution + ((float)ExtractedElectronNum)  * TrueGasGain;
#         float uS2total = curand_normal(&s) *sqrtf( ((float)ExtractedElectronNum) )* TrueGasGain * TrueSEResolution / (1. - S2TopFraction) + ((float)ExtractedElectronNum)  * TrueGasGain / (1. - S2TopFraction);
#

    :param df:
    :param c:
    :return:
    """
    df['TrueG2'] = c['g2'] * 1  # InversedS2CorrectionFactor->1
    df['TrueGasGain'] = df['TrueG2'] / 1  # c['ExtractionEfficiency']
    df['TrueSEResolution'] = c['gas_gain_res']
    S2WallSurviveProb = 1.
    ExtractionEfficiency = 1.

    df['DriftedElectronNum'] = np.random.binomial(df['ElectronNum'],
                                                  df['ElectronSurviveProb'] * S2WallSurviveProb,
                                                  size=len(df))

    df['ExtractedElectronNum'] = np.random.binomial(df['DriftedElectronNum'],
                                                    ExtractionEfficiency,
                                                    size=len(df))

    df['uS2'] = np.random.normal(df['ExtractedElectronNum'] * df['TrueGasGain'],
                                 np.sqrt(df['ExtractedElectronNum'] * df['TrueGasGain'] * c['gas_gain_res']))

    df['uS2total'] = np.random.normal(df['ExtractedElectronNum'] * df['TrueGasGain'] / (1. - c['area_top_fraction']),
                                      np.sqrt(df['ExtractedElectronNum'] * df['TrueGasGain'] * c['gas_gain_res'] / (
                                          1. - c['area_top_fraction'])))

    return df


#         // 12) S1 reconstruction
#         float S1BiasMean = 0.;
#         get_scaled_value_from_1darrays(reconstruction_bias_mean_lower, reconstruction_bias_mean_median, reconstruction_bias_mean_upper, RecBiasMeanScaler, (float)HitPhotonNum, &S1BiasMean);
#         float S1BiasSigma = 0.;
#         get_scaled_value_from_1darrays(reconstruction_bias_res_lower, reconstruction_bias_res_median, reconstruction_bias_res_upper, RecBiasResScaler, (float)HitPhotonNum, &S1BiasSigma);
#         float S1Bias = gpu_truncated_gaussian(&s, S1BiasMean, S1BiasSigma, -1, 100000000000);
#         float S1 = ((float)PENum)*(1.+S1Bias);
#         S1s[n+1] = S1;
#
#
#

#         // 13) S2 reconstruction
#         float S2BiasMean = 0.;
#         get_scaled_value_from_1darrays(reconstruction_s2bias_mean_lower, reconstruction_s2bias_mean_median, reconstruction_s2bias_mean_upper, RecS2BiasMeanScaler, uS2, &S2BiasMean);
#         float S2BiasSigma = 0.;
#         get_scaled_value_from_1darrays(reconstruction_s2bias_res_lower, reconstruction_s2bias_res_median, reconstruction_s2bias_res_upper, RecS2BiasResScaler, uS2, &S2BiasSigma);
#         float S2Bias = gpu_truncated_gaussian(&s, S2BiasMean, S2BiasSigma, -1, 100000000000);
#         float S2 = ((float)uS2)*(1.+S2Bias);
#         uS2total = ((float)uS2total)*(1.+S2Bias); // assume the total and bottom share the same bias (however it is not correct, but very minor effect)
#         S2s[n+1] = S2;
#         TotalS2s[n+1] = uS2total;
#

#         // 14) S1 correction
#         float cS1 = 0.;
#         if (InversedS1CorrectionFactor>0) cS1 = S1 / InversedS1CorrectionFactor;
#         cS1s[n+1] = cS1;
#

#         // 15) S2 correction
#         float cS2 = 0.;
#         if(InversedS2CorrectionFactor*ElectronSurviveProb>0) cS2 = S2 / InversedS2CorrectionFactor / ElectronSurviveProb;
#         cS2s[n+1] = cS2;


operations = (initial_conditions,
              recombination,
              quanta,
              heat_quenching,
              excimer_ion_distributing,
              recombination_fluctuation,
              photon_electron_number,
              s1_s2_inversed_correction_factor,
              electron_lifetime_prob,
              photon_detection,
              dpe_probability,
              electron_detection)
