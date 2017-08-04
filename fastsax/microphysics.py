import numpy as np
from numpy.polynomial.polynomial import polyval

from fastsax.helpers import SaxUnphysicalModel


def get_recombination_fraction(df, c):
    """Get recombination fraction
    """
    if c['recoil_type'] == 'ER':
        df = get_ER_recombination_fraction(df, c)
    elif c['recoil_type'] == 'NR':
        df = get_NR_recombination_fraction(df, c)
    else:
        raise ValueError("'recoil_type' must be 'ER' or 'NR'")
    return df


def get_ER_recombination_fraction(df, c):
    """Determine ER recombination fraction

    https://github.com/XENON1T/bbf/blob/6781f6ccd5880807be2e6a4e33d67220886e78b2/gpu_modules/SignalSimCuda.py#L456

    // the model is pol4 + Reference curve (pol9)
    // see: https://github.com/XENON1T/FirstResults/blob/master/BandFitting/ER/SignalProcessCuda.py#L20
    // NOTE this is only valid for low energy region, if you want to extend to high energy you need definely change this part
    *HeatQuenching=1.;
    if (Energy<rec_frac_pars[5]) {*RecombFrac = 0;return;} // fully recombined
    float py = GetReferencePY(Energy);
    for (int i=0;i<5;i++)
        py += (*(rec_frac_pars+i))*powf(Energy, (float)i);
    if(py< ExIonRatio/(1.+ExIonRatio) / W) {*RecombFrac = 0;return;}
    if(py>1./W) {*RecombFrac=1.;return;}
    float cy = 1./W - py;
    *RecombFrac = 1. - cy*W*(1+ExIonRatio);
    return;
    """
    # The reference photon yield plus our offset model
    df.loc[:, 'py'] = polyval(df['E'],
                              c['reference']['py']['k']) + polyval(df['E'], c['py'])

    df.loc[:, 'cy'] = 1. / c['W'] - df['py']
    df.loc[:, 'RecombFrac'] = 1. - df['cy'] * c['W'] * (1. + c['Nex/Ni'])

    #
    # Cleanup some unphysical values
    #

    # Fully recombined below some energy
    df.loc[df['E'] < c['reference']['py']['threshold'],
           'RecombFrac'] = 0

    # Not sure TODO
    df.loc[df['py'] < c['Nex/Ni'] / (1. + c['Nex/Ni']) / c['W'],
           'RecombFrac'] = 0

    # Not sure TODO
    df.loc[df['py'] > 1 / c['W'],
           'RecombFrac'] = 1

    return df


def get_NR_recombination_fraction(df, c):
    """

    https://github.com/XENON1T/bbf/blob/6781f6ccd5880807be2e6a4e33d67220886e78b2/gpu_modules/SignalSimCuda.py#L473

     // Using NEST model:
    // https://arxiv.org/pdf/1412.4417.pdf
    // HARDCODE: "download" all the NEST parameters
    float alpha = rec_frac_pars[0];
    float zeta = rec_frac_pars[1];
    float beta = rec_frac_pars[2];
    float gamma = rec_frac_pars[3];
    float delta = rec_frac_pars[4];
    float kappa = rec_frac_pars[5];
    float eta = rec_frac_pars[6];
    float lambda = rec_frac_pars[7];
    // HeatQuenching part: Lindhard theory
    float e = 11.5*Energy*powf(54., -7./3.);
    float g = 3.*powf(e, 0.15) + 0.7*powf(e, 0.6) + e;
    *HeatQuenching = kappa*g / (1. + kappa*g);
    // Penning quenching
    *PenningQuenching = 1. / (1. + eta*powf(e, lambda) );
    // excimer-to-ion ratio
    *ExIonRatio = alpha*powf(Field, -zeta)*(1. - expf(-beta*e));
    // recombination
    float xi = gamma*powf(Field, - delta);
    float Nq = Energy * (*HeatQuenching) / W;
    float Ni = 1. / (1.+ (*ExIonRatio)) * Nq;
    *RecombFrac = 1. - logf(1.+Ni*xi) / (Ni*xi);
    return;

    :param df:
    :param c:
    :return:
    """
    # HeatQuenching part: Lindhard theory
    e = 11.5 * df['E'] * pow(54, -7 / 3)
    g = 3 * pow(e, 0.15) + 0.7 * pow(e, 0.6) + e
    df['HeatQuenching'] = c['kappa'] * g / (1. + c['kappa'] * g)

    # Penning quenching
    df['PenningQuenching'] = 1 / (1 + c['eta'] * pow(e, c['lambda']))
    if (df['PenningQuenching'] < 0).any():
        raise SaxUnphysicalModel('PenningQuenching < 0')

    # excimer-to-ion ratio
    df['ExIonRatio'] = c['alpha'] * pow(c['field'], -1 * c['zeta']) * (1 - np.exp(-1 * c['beta'] * e))

    # Recombination
    xi = c['gamma'] * pow(c['field'], -1 * c['delta'])
    Nq = df['E'] * df['HeatQuenching'] / c['W']
    Ni = 1 / (1 + df['ExIonRatio']) * Nq

    temp = Ni * xi
    df['RecombFrac'] = 1 - np.log(1 + temp) / (temp)
    raise NotImplementedError


def recombination(df, c):
    """Recombination fraction and heat quenching

    Some blurb about the Physics!

    TODO: Implement NR recombination fraction

    This is obtained from

        https://github.com/XENON1T/bbf/blob/6781f6ccd5880807be2e6a4e33d67220886e78b2/gpu_modules/SignalSimCuda.py#L917

        // 1) Get the essential derived parameters
        // get heat quenching and recomb. frac.
          float HeatQuenching = 1.;
        float PenningQuenching = 1.;
        float RecombFrac = 1.;
        if (*sim_type_flag==1) get_ER_recomb_frac(rec_frac_pars, Energy, W, ExIonRatio, &HeatQuenching, &RecombFrac);
        else get_NR_recomb_frac(rec_frac_pars, Energy, Field, W, &ExIonRatio, &HeatQuenching, &PenningQuenching, &RecombFrac); // NOTE actually NR model will give a new ExIonRatio
        // get recomb. fluc., currently ER/NR share the same model for recomb. fluc.
        float DeltaR = 0.;
        get_recomb_fluc(rec_fluc_pars, Energy, &DeltaR);

    """
    # Setup a few variables that we need
    df['HeatQuenching'] = 1
    df['PenningQuenching'] = 1
    df['RecombFrac'] = 1

    # Determine the recombination fraction depending on the recoil type
    df = get_recombination_fraction(df, c)

    # Check if model is unphysical
    if (df['RecombFrac'] < 0).any() or (df['RecombFrac'] > 1).any():
        raise SaxUnphysicalModel("RecombFrac unphysical")

    df.loc[:, ('DeltaR')] = c['rf0'] * (1. - np.exp(-df['E'] / c['rf1']))

    if (df['DeltaR'] < 0).any():
        raise SaxUnphysicalModel("DeltaR < 0")

    return df


def quanta(df, c):
    """Step 2: Get the total number of quanta

    Original code:

        float MeanQuantaNum = Energy / W;
        int QuantaNum = (int) (curand_normal(&s) * sqrtf( FanoFactor*MeanQuantaNum) + MeanQuantaNum);
    """
    # Determine the mean number of quanta
    MeanQuantaNum = df['E'] / c['W']

    # Distribute them as a Gaussian/normal distribution
    df['QuantaNum'] = np.random.normal(scale=np.sqrt(c['fano_factor'] * MeanQuantaNum),
                                       loc=c['fano_factor'],
                                       size=len(df))

    # But ensure physicalness of value.  I.e. floor at zero
    df['QuantaNum'] = np.clip(df['QuantaNum'].astype(np.int32),
                              0, None)
    return df


def heat_quenching(df, c):
    """Step 3: heat quenching

#         if (HeatQuenching<0) {physical_process_flag = 0; break;}
#         else if (HeatQuenching<1) QuantaNum = gpu_binomial(&s, QuantaNum, HeatQuenching);

    :param df:
    :param c:
    :return:
    """
    if (df['HeatQuenching'] < 0).any():
        raise SaxUnphysicalModel("HeatQuenching < 0")

    # TODO: check when p=1, or n<=0
    df['QuantaNum'] = np.random.binomial(df['QuantaNum'],
                                         df['HeatQuenching'],
                                         size=len(df))
    return df
