import numpy as np
from scipy.stats import truncnorm 
from scipy.stats import halfnorm


from joblib import Parallel, delayed


###################################################################################################
# Routines for parameter transform
###################################################################################################
def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def inverse_scale_param(P, a, b):
    p_range = b - a
    param = P  * p_range + a
    return param

#def transform(P, a, b):
#    in_log = inv_logit(P)
#    parameter = inverse_transform_param(in_log, a, b)
#    return parameter

def scale_param(P, a, b):
    return (P-a)/(b-a)


def build_prior(es_parameters, nEnsemble):
    """
    Generate an ensemble of logit transformed parameters by scaling truncnorm sampled parameters.
    add some more docstring
    """
    mLength = len(es_parameters)
    mPrior_untransformed = np.zeros([mLength, nEnsemble]) #Prior ensemble
    mPrior = np.zeros([mLength, nEnsemble]) #Prior ensemble
    
    es_parameters['std'] = (es_parameters['upper'] - es_parameters['lower']) / es_parameters['width']
    es_parameters['a'] = (es_parameters['lower'] - es_parameters['mean']) / es_parameters['std']
    es_parameters['b'] = (es_parameters['upper'] - es_parameters['mean']) / es_parameters['std']
    print(es_parameters) 

    stdevM = es_parameters['std'].values
    param_mean = es_parameters['mean'].values
    a = es_parameters.a.values
    b = es_parameters.b.values

    # Generate prior parameter distribution in the parameter space
    for i in range(mLength):
        mPrior_untransformed[i,:] = truncnorm.rvs(a[i], b[i], loc=param_mean[i], scale=stdevM[i], size=nEnsemble, random_state=None)

    # Transform parameters to logit space
    for i in range(nEnsemble):
        scaled = scale_param(mPrior_untransformed[:,i], es_parameters['lower'].values, es_parameters['upper'].values)
        transed = logit(scaled)
        mPrior[:,i] = transed

    # Set the current parameter set for the 0th iteration
    return mPrior 


def inv_trans(es_parameters, mCurrent):
    # Inverse tranform parameters
    M = np.zeros_like(mCurrent)
    nEnsemble = np.shape(M)[1]
    for i in range(nEnsemble):
        scaled = inv_logit(mCurrent[:,i])
        M[:,i] = inverse_scale_param(scaled, es_parameters['lower'].values, es_parameters['upper'].values)
    return M
    
    
def covariance_matrix(D, data, error, alpha_j, phi_std, phi_val):
    Nd, Ne = np.shape(D)
    delayed_funcs = []
    for i in range(Ne):
        phi_0 = {}
        for j in phi_val:
            phi_0[j] = np.mean(phi_val[j])  
        delayed_funcs.append(delayed(mcmc)(data,D[:,i], 1000, alpha_j, phi_std, phi_0))
    result = Parallel(n_jobs=-1, verbose=5)(delayed_funcs)
    return np.array(result).T
         
        
def build_covariance_prior(Ne, data, error):
    Nd = len(data)
    cov_prior = np.zeros([Nd, Ne])
    for i in range(Ne):
        for er in error:
            data.loc[data['group'] == er, 'std'] = halfnorm.rvs(scale = error[er])
        cov_prior[:,i] = data['std'].values 
    return cov_prior
    

def mcmc(odata, D, num_mcmc, alpha_j, phi_std, phi_0):
    esdata = odata.copy() 
    esdata['G'] = D
    for group in esdata.group.unique():
        selection = esdata.query('group == @group')
        y = selection.Y.values
        G = selection.G.values
        phi = np.std(y-G)*alpha_j
        esdata.loc[esdata['group'] == group, 'phi'] = phi
    return esdata.phi.values

"""        basic_model = pm.Model()
        with basic_model:
            # Priors for unknown model parameters
            error = pm.HalfNormal("error", sigma=5)
            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal("Y_obs", mu=y, sigma=error, observed=G)
            idata = pm.sample(tune=100, draws=1, chains=1, progressbar=False)
        phi = (idata.posterior["error"].to_numpy().flatten()[0]) 
        del basic_model 
"""


def get_group_list(the_dataframe):
    groups = the_dataframe.group.unique()
    long_ref_list = {}
    short_ref_list = {}
    for group in groups:
        short_ref_list[group] = np.min(the_dataframe.loc[the_dataframe['group'] == group].index)
        long_ref_list[group] = the_dataframe.loc[the_dataframe['group'] == group].index
    return short_ref_list, long_ref_list
        