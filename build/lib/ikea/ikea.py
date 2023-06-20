import pandas as pd
import numpy as np
import dask.array as da
import arviz as az
import time

from scipy.stats import norm
from scipy.stats import truncnorm 

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
        self.calculation_type = 'ikea'

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
        
        if self.calculation_type == 'ikea':
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
#       Covariance data 
        if self.calculation_type == 'ikea' :
            phi = build_covariance_prior(Ne, self.observation_data, self.error)
        else :
            phi = self.observation_data.noise.values
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
            
    # calculate del_D
            D_mean = D.mean(axis=1)
            del_D = np.zeros_like(D)

            for i in range(self.nEnsemble):
                del_D[:,i] = (D[:,i] - D_mean)

            
    
    #Calculate Cmd
            M_mean = M.mean(axis=1)
            del_M = np.zeros_like(M)
            for i in range(self.nEnsemble):
                del_M[:,i] = M[:,i] - M_mean

            Cmd = del_M@del_D.T /(self.nEnsemble-1)  
    
    # Perturb Observations
            Duc = np.zeros_like(D)
            for i in range(self.nEnsemble):
                if self.calculation_type == 'ikea':
                    Duc[:,i] = np.sqrt(alpha)*phi[:,i]*np.random.normal(0,1,Nd)+d_obs
                else:
                    Duc[:,i] = np.sqrt(alpha)*phi*np.random.normal(0,1,Nd)+d_obs
         
            start = time.time()

    #calculate M_update
            print(' inversion type ', self.inversion_type)
            M_update = np.zeros_like(M)
            if(self.inversion_type == 'svd'):
                Cdd = (del_D@del_D.T)/(Ne-1)  
                for index in range(Ne): 
                    Cd = np.zeros([self.dLength, self.dLength])
                    np.fill_diagonal(Cd,phi[:,index]**2)
                    K = Cdd + alpha*Cd
                    Kinv, svd_rank = sla.pinvh(K, return_rank=True)
                    M_update[:,index] = M[:,index]+Cmd@Kinv@(Duc[:,index]-D[:,index])
                    
            if(self.inversion_type == 'subspace'):
                Ud, Wd, Vd = np.linalg.svd(del_D, full_matrices=False, compute_uv=True, hermitian=False)
                Binv = np.diag(Wd**(-2)) 
                for index in range(Ne):
                    aCd = (Ne-1) * alpha * phi[:,index]**2
                    Ainv = np.diag(aCd**(-1))
                    bracket = Binv + Ud.T@Ainv@Ud
                    bracketinv = np.linalg.inv(bracket)
                    Kinv = (Ne-1) * (Ainv - Ainv@Ud@bracketinv@Ud.T@Ainv)
                    M_update[:,index] = M[:,index]+Cmd@Kinv@(Duc[:,index]-D[:,index]) 

            if(self.inversion_type == 'fast_subspace'):
                Ud, Wd, Vd = np.linalg.svd(del_D, full_matrices=False, compute_uv=True, hermitian=False)
                Binv = np.diag(Wd**(-2)) 
                for index in range(Ne):
                    aCd = (Ne-1) * alpha * phi[:,index]**2
                    # Ainv = np.diag(aCd**(-1))
                    AinvUd = ((aCd**(-1))*Ud.T).T
                    #bracket = Binv + Ud.T@Ainv@Ud
                    bracket = Binv + Ud.T@AinvUd
                    bracketinv = np.linalg.inv(bracket)
                    #Kinv = (Ne-1) * (Ainv - Ainv@Ud@bracketinv@Ud.T@Ainv)
                    Kinv = (Ne-1) * (np.diag(aCd**(-1)) - AinvUd@bracketinv@AinvUd.T)
                    M_update[:,index] = M[:,index]+Cmd@Kinv@(Duc[:,index]-D[:,index]) 

            if(self.inversion_type == 'efast_subspace'):
                rand_phi = np.zeros(Nd)
                for i in range(Nd):
                    phi_mean = np.mean(phi[i,:])
                    phi_std = np.std(phi[i,:])
                    rand_phi[i] = truncnorm.rvs(-1,1,phi_mean,phi_std)

                Ud, Wd, Vd = np.linalg.svd(del_D, full_matrices=False, compute_uv=True, hermitian=False)
                Binv = np.diag(Wd**(-2)) 
                # aCd = (Ne-1) * alpha * phi.mean(axis=1)**2
                aCd = (Ne-1) * alpha * rand_phi**2
                # Ainv = np.diag(aCd**(-1))
                AinvUd = ((aCd**(-1))*Ud.T).T
                #bracket = Binv + Ud.T@Ainv@Ud
                bracket = Binv + Ud.T@AinvUd
                bracketinv = np.linalg.inv(bracket)
                #Kinv = (Ne-1) * (Ainv - Ainv@Ud@bracketinv@Ud.T@Ainv)
                Kinv = (Ne-1) * (np.diag(aCd**(-1)) - AinvUd@bracketinv@AinvUd.T)
                M_update = M+Cmd@Kinv@(Duc-D) 

            if(self.inversion_type == 'dask'):
                rand_phi = np.zeros(Nd)
                for i in range(Nd):
                    phi_mean = np.mean(phi[i,:])
                    phi_std = np.std(phi[i,:])
                    rand_phi[i] = truncnorm.rvs(-1,1,phi_mean,phi_std)

                M_da = da.from_array(M, chunks='auto')
                Cmd_da = da.from_array(Cmd, chunks='auto')
                Duc_da = da.from_array(Duc, chunks=(500,Ne))
                D_da = da.from_array(D, chunks=(500,Ne))
                rand_phi_da = da.from_array(rand_phi, chunks=500)
                
                Ud, Wd, Vd = np.linalg.svd(del_D, full_matrices=False, compute_uv=True, hermitian=False)
                Ud = da.from_array(Ud, chunks=(500,Ne))
                Wd = da.from_array(Wd, chunks='auto')
                
                Binv = np.diag(Wd**(-2)) 
                # aCd = (Ne-1) * alpha * phi.mean(axis=1)**2
                aCd = (Ne-1) * alpha * rand_phi_da**2

                AinvUd = ((aCd**(-1))*Ud.T).T
                #bracket = Binv + Ud.T@Ainv@Ud
                bracket = Binv + Ud.T@AinvUd
                bracketinv = np.linalg.inv(bracket)

                Kinv = (Ne-1) * (np.diag(aCd**(-1)) - AinvUd@bracketinv@AinvUd.T)
                M_update_da = M_da+Cmd_da@Kinv@(Duc_da-D_da) 

                M_update = M_update_da.compute() 



            if(self.inversion_type == 'esmda'):

                Ud, Wd, Vd = np.linalg.svd(del_D, full_matrices=False, compute_uv=True, hermitian=False)
                Binv = np.diag(Wd**(-2)) 
                # aCd = (Ne-1) * alpha * phi.mean(axis=1)**2
                aCd = (Ne-1) * alpha * phi**2
                # Ainv = np.diag(aCd**(-1))
                AinvUd = ((aCd**(-1))*Ud.T).T
                #bracket = Binv + Ud.T@Ainv@Ud
                bracket = Binv + Ud.T@AinvUd
                bracketinv = np.linalg.inv(bracket)
                #Kinv = (Ne-1) * (Ainv - Ainv@Ud@bracketinv@Ud.T@Ainv)
                Kinv = (Ne-1) * (np.diag(aCd**(-1)) - AinvUd@bracketinv@AinvUd.T)
                M_update = M+Cmd@Kinv@(Duc-D) 

            if(self.inversion_type == 'esmda_dask'):

                Ud, Wd, Vd = np.linalg.svd(del_D, full_matrices=False, compute_uv=True, hermitian=False)
                Binv = np.diag(Wd**(-2)) 
                # aCd = (Ne-1) * alpha * phi.mean(axis=1)**2
                aCd = (Ne-1) * alpha * phi**2
                # Ainv = np.diag(aCd**(-1))
                AinvUd = ((aCd**(-1))*Ud.T).T
                #bracket = Binv + Ud.T@Ainv@Ud
                bracket = Binv + Ud.T@AinvUd
                bracketinv = np.linalg.inv(bracket)
                #Kinv = (Ne-1) * (Ainv - Ainv@Ud@bracketinv@Ud.T@Ainv)
                Kinv = (Ne-1) * (np.diag(aCd**(-1)) - AinvUd@bracketinv@AinvUd.T)
                M_update = M+Cmd@Kinv@(Duc-D) 


            end = time.time()
            print('')
            print('Time for inversion and M update ', end - start) 
            print('')
    
    # update phi
#==========================================================================================================
            alpha_j += 1/Na
#            phi_update = np.zeros_like(phi)
#            for i in range(self.nEnsemble):
#                phi_update_std = (D[:,i]-d_obs).std()
#                phi_update_mean = (np.abs((D[:,i]-d_obs))).mean()
#                phi_update[i] = phi_update_std
            if self.calculation_type == 'ikea' :
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
        
        if self.calculation_type == 'ikea':
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
        