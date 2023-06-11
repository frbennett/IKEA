import pandas as pd
import numpy as np
import arviz as az

from scipy.stats import norm

import scipy.linalg as sla 

import os

import shutil

from .utils import *

class esmda(object):
    def __init__(self,  **kwargs):
        # Set default parameters
        self.job_name = 'esmda_job'
        self.parameter_file_name = 'es_parameters.csv'
        self.observation_file_name = 'es_data.csv'
        self.data_file_name = 'es_data.csv'
        self.nEnsemble = 100 #the number of ensembles
        self.maxIter = 10  #the number of iterations
        self.inversion_type = 'svd'

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.parameter_data = pd.read_csv(self.parameter_file_name)
        self.observation_data = pd.read_csv(self.observation_file_name)

        # Initialise global arrays
        self.mLength = len(self.parameter_data)
        self.dLength = len(self.observation_data)
        self.mPrior = np.zeros([self.mLength, self.nEnsemble]) #Prior ensemble
        self.mPrior_untransformed = np.zeros([self.mLength, self.nEnsemble]) #Prior ensemble

        if os.path.exists(self.job_name):
            print('Deleting directory ' + self.job_name)
            shutil.rmtree(self.job_name)
        os.mkdir(self.job_name)
        
    def report(self, iter, M, D, phi):
        iteration_parameters = pd.DataFrame(M.T, columns=self.parameter_data.parameter.values)
        file_name = self.job_name + '_' + str(iter) + '_parameters.csv'
        iteration_parameters.T.to_csv(self.job_name + '/' + file_name)

        iteration_data = pd.DataFrame(D.T, columns=self.observation_data.label.values)
        iteration_data = iteration_data.T 
        file_name = self.job_name + '_' + str(iter) + '_data.csv'
        iteration_data.to_csv(self.job_name + '/' + file_name)
        
        iteration_phi = pd.DataFrame(phi.T, columns=self.observation_data.label.values)
        file_name = self.job_name + '_' + str(iter) + '_phi.csv'
        iteration_phi.T.to_csv(self.job_name + '/' + file_name)
        
        print(' ')
        print('Completed iteration ', iter)
        print('============================')
        print(' ')
        
        
    def run_esmda(self, fill_ensemble):
        alpha_j = 0.0
        Ne = self.nEnsemble
        Nd = self.dLength
        Nm = self.mLength
        Na = self.maxIter
#        phi = halfnorm.rvs(scale = 20,  size = Ne) 
        phi = build_covariance_prior(Ne, self.observation_data, self.error)
        d_obs = self.observation_data['Y']
        M = build_prior(self.parameter_data, self.nEnsemble)
        alpha = self.maxIter
        short_ref_list, long_ref_list = get_group_list(self.observation_data)
        for iter in range(self.maxIter):
    
      #fill the ensemble 
            M_invt = inv_trans(self.parameter_data, M)
            D = fill_ensemble(M_invt, self.nEnsemble, self.mLength, self.dLength)
            
            self.report(iter, M_invt, D, phi)
            
            # If we are at the final iteration, we have evaluated the model, we don't have to updaate the parameters
            # we can quit now
            if iter == self.maxIter :
                print('maxiter ', iter)
                return
            
    # calculate Cdd
            D_mean = D.mean(axis=1)
            del_D = np.zeros_like(D)

            for i in range(self.nEnsemble):
                del_D[:,i] = D[:,i] - D_mean

            Cdd = (del_D@del_D.T)/(Ne-1)
    
    #Calculate Cmd
            M_mean = M.mean(axis=1)
            del_M = np.zeros_like(M)
            for i in range(self.nEnsemble):
                del_M[:,i] = M[:,i] - M_mean

            Cmd = del_M@del_D.T /(self.nEnsemble-1)  
    
    # Perturb Observations
            Duc = np.zeros_like(D)
            for i in range(self.nEnsemble):
                Duc[:,i] = np.sqrt(alpha)*phi[:,i]*np.random.normal(0,1,Nd)+d_obs
        
    #calculate M_update
            M_update = np.zeros_like(M)
            if(self.inversion_type == 'svd'):
                for index in range(Ne):
                    Cd = np.zeros([self.dLength, self.dLength])
                    np.fill_diagonal(Cd,phi[:,index]**2)
                    K = Cdd + alpha*Cd
                    Kinv, svd_rank = sla.pinvh(K, return_rank=True)
                    M_update[:,index] = M[:,index]+Cmd@Kinv@(Duc[:,index]-D[:,index])
    
    # update phi
#==========================================================================================================
            alpha_j += 1/Na
#            phi_update = np.zeros_like(phi)
#            for i in range(self.nEnsemble):
#                phi_update_std = (D[:,i]-d_obs).std()
#                phi_update_mean = (np.abs((D[:,i]-d_obs))).mean()
#                phi_update[i] = phi_update_std
            phi_std = {}
            phi_val = {}
            for i in short_ref_list:
                phi_std[i] = np.std(phi[short_ref_list[i],:])
                phi_val[i] = phi[short_ref_list[i],:]
            phi = covariance_matrix(D, self.observation_data, self.error, alpha_j, phi_std, phi_val)
#        phi_update[N] = np.random.normal(loc=phi_update_mean, scale=phi_update_std, size=None)

#            print('average phi_update ', np.mean(phi_update) )
    
#            phi = phi_update.copy()
        
#==========================================================================================================
        
            print(inv_trans(self.parameter_data, M).mean(axis=1))
            print(phi.mean()) 
    
            M = M_update
        
        iteration_phi = pd.DataFrame(phi.T, columns=self.observation_data.label.values)
        file_name = self.job_name + '_' + 'final' + '_phi.csv'
        iteration_phi.T.to_csv(self.job_name + '/' + file_name)


    def predictive_posterior(self, n):
        pred_post = pd.DataFrame()
        file_name = self.job_name + '_' + str(self.maxIter) + '_data.csv'
        posterior = pd.read_csv(self.job_name + '/' + file_name)
        del posterior[posterior.columns[0]]
        noise = self.observation_data.noise.values
        for i in posterior.columns:
            samples = pd.DataFrame()
            data = posterior[i].values
            samples[i] = data
            for j in range(n):
                label = str(i) + '_' + str(j+1)
        
                sample = np.random.normal(0,1,self.dLength) * self.observation_data.noise.values+data
                samples[label] = sample

            pred_post = pd.concat([pred_post, samples], axis=1)
        pred_post.set_index(self.observation_data.name, inplace=True)
        pred_post.to_csv(self.job_name + '/' + 'posterior_predictive.csv')
        