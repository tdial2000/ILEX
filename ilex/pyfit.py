##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 19/02/2024 
##
##
## 
## 
## General purpose Fitting structure
## TODO: 
##
##===============================================##
##===============================================##
# imports
import inspect
from scipy.optimize import curve_fit
from scipy.stats import chi2
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import bilby
from bilby.core.utils.io import check_directory_exists_and_if_not_mkdir
import glob, os
import math
from bilby.core.utils import infer_parameters_from_function

#----------------------------------------#
# UTILITY FUNCTIONS                      #
#----------------------------------------#

def _dict_init(*dicts):
    """
    Initilise any number of dictionaries

    Returns
    -------
    *dicts
        A number of Dictionaries
    """    

    #assign input dictionaries to empty {} object
    out_ = list(dicts)
    for i,dic in enumerate(out_):
        if dic is None:
            out_[i] = {}

    if len(out_) == 1:
        out_ = out_[0]

    return out_



def _merge_dicts(*dicts):
    """
    Combine multiple Dictionaries into 1 dictionary

    Returns
    -------
    *dicts : 
        A number of dictionaries
    """    

    # combine multiple dictionaries together
    dicts = list(dicts)
    odict = {}
    for i, dic in enumerate(dicts):
        odict = {**odict, **dic}
    
    return odict


def _dict_get(dict, keys):
    """
    Get a sub-set of the Dictionary based on keys

    Parameters
    ----------
    dict : Dict
        Original dictionary
    keys : List(str)
        List of items to retrieve

    Returns
    -------
    new_dict : Dict
        Sub-set of Dictionary
    """    
    new_dict = {}

    for key in keys:
        new_dict[key] = dict[key]
    
    return new_dict





def _priorUniform(p):
    """
    Convert Dictionary of Priors to Bilby Uniform prior instance

    Parameters
    ----------
    p : Dict(List(float))
        Dictionary of priors for Bilby.run_sampler

    Returns
    -------
    Bilby.UniformPrior
        Bilby uniform priors
    """
    priors = {}
    for _,key in enumerate(p.keys()):
        priors[key] = bilby.core.prior.Uniform(p[key][0],p[key][1],key)
    
    return priors




def _clean_bilby_run(outdir, label):
    """
    Remove bilby sample run
    """
    print(f"Removing Bilby sampled run: outdir: {outdir}   label: {label}")

    for file in glob.glob(os.path.join(os.getcwd(), f"{outdir}/{label}") + "*"):
        os.remove(file)

    return 






# likelyhood function for specifying both yerr and xerr
class PyfitLikelihood(bilby.Likelihood):
    def __init__(self, x, y, func, xerr = None, yerr = None, **kwargs):
        """
        A general gaussian likelihood function that estimates variance in posterior 
        given uncertainty in both x and y.

        This class is a basis class the user should use when the user wants to estimate 
        the max likelihood using both x and y errors.

        Parameters
        ----------
        x : np.ndarray or array-like
            x values
        y : np.ndarray or array-like
            y values
        xerr : np.ndarray or array-like
            x uncertainties
        yerr : np.ndarray or array-like
            y uncertainties
        func : 
            callable function to evaluate data with

        """         
        
        #---------- Get function arguments ----------#
        parameters = inspect.getfullargspec(func).args[1:]
        super().__init__(parameters = dict.fromkeys(parameters))
        self.parameters = dict.fromkeys(parameters)
        self.func_keys = self.parameters.keys()        
        
        #---------- Set data -----------#
        self.x = x
        self.y = y
        self.yerr = yerr
        self.xerr = xerr

        # Check if either x or y uncertainties are given, if not then
        self.sigma_flag = False
        if self.yerr is None:
            self.sigma = None
            self.parameters['sigma'] = None
            self.sigma_flag = True


        # if xerr is Undefined, we want to skip the propogation sigma calculation
        self._skip_propogation_sigma = False
        if self.xerr is not None:
            if hasattr(xerr, "__len__"):
                self.xerr = np.array(xerr)
        else:
            self._skip_propogation_sigma = True
            self.xerr = 0.0

        if self.yerr is not None:
            if hasattr(yerr, "__len__"):
                self.yerr = np.array(yerr)
        else:
            self.yerr = 0.0


        self.func = func


    # function
    def propogated_sigma(self):

        return 0.0


    def calc_sigma(self):

        # default way to calculate sigma
        if self._skip_propogation_sigma:
            return self.yerr
        else:
            return (self.yerr**2 + self.propogated_sigma()**2)**0.5


    @property
    def model_parameters(self):

        return {key : self.parameters[key] for key in self.func_keys}
    

    def log_likelihood(self):
        """
        Evaluate the log liklihood of function
    
        """

        sigma = self.sigma
        y_model = self.func(self.x, **self.model_parameters)

        # for a gaussian log likelihood
        return -0.5 * np.sum(((self.y - y_model)/sigma)**2 + np.log(2 * np.pi * sigma**2))

    @property
    def sigma(self):

        if self.sigma_flag:
            return self.parameters.get('sigma', self._sigma)
        else:
            return self.calc_sigma()

    
    def get_sigma(self):

        sigma = self.sigma
        if sigma is None:
            return None
        elif hasattr(sigma, "__len__"):
            return sigma.copy()
        else:
            return sigma




class _PyfitDefault(PyfitLikelihood):
    pass
        

            














class _globals:
    pass

_fit = _globals()

_fit.attr = ["x", "y", "xerr", "yerr", "func", "method",
            "fit_keywords", "prior", "keys",
            "static", "residuals", "likelihood"]

_fit.meth = ["least squares", "bayesian"]




class _posterior:
    # class for posteriors
    def __init__(self, val = 0.0, p = 0.0, m = 0.0):
        self.val = val      # best fit value
        self.p = p          # positive error
        self.m = m         # negative error

    def __str__(self):

        return(f"{self.val:4f}  +{self.p:4f}  -{self.m:.4f}")



# create fitting class
class fit:
    """
    General fitting class, model a function using either bayesian inference using BILBY, or 
    least squares using scipy.curve_fit(). This class has a handful of useful quality of life features,
    including model plotting and statistics such as chi2. 

    Attributes
    ----------
    x : np.ndarray
        X data
    y : np.ndarray
        Y data
    xerr: np.ndarray
        X data error
    yerr : np.ndarray
        Y data error
    func : __func__
        Callable python function to model/fit to
    method : str
        Method to fit \n
        [bayesian] - Use bayesian inference (see https://lscsoft.docs.ligo.org/bilby/) \n
        [least squares] - use least squares method (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
    fit_keywords : Dict
        Dictionary of keywords to apply to fitting method
    prior : Dict
        Priors for modelling
    static : Dict
        Priors to keep constant during modelling
    posterior : Dict
        Sampled posteriors
    likelihood : Bilby.Likelihood
        Likelihood class used for bayesian inference
    keys : List
        parameter names to be sampled
    chi2 : float
        Chi squared statistic
    chi2err : float
        Chi squared error
    rchi2 : float
        reduced Chi squared
    rchi2err : float
        reduced Chi squared error
    nfitpar : int
        number of fitted parameters
    dof : int
        degrees of freedom
    p : float
        p-value statistic
    bic : float
        bayesian information criterion
    bic_err : float
        bic error
    residuals : bool
        if true, plot residuals when .plot() is called
    plotPosterior : bool
        if true, save image of posterior corner plot

    """

    def __init__(self, **kwargs):
        """
        Run through set function
        """

        # data
        self.x = None
        self.y = None
        self.xerr = None
        self.yerr = None
        self.sigma = None

        # proc
        self.mask = None

        # function 
        self.func = None
        self.method = "least squares"
        self.fit_keywords = {}

        # bilby inputs
        self.likelihood = _PyfitDefault

        # params
        self.prior = {}
        self.static = {}
        self.posterior = {}
        self.bounds = {}    # for least squares method
        self.keys = []

        # stats (general)
        self.chi2 = 0.0
        self.chi2err = 0.0
        self.rchi2 = 0.0
        self.rchi2err = 0.0
        self.nfitpar = 0
        self.dof = 0
        self.p = 0.0

        # stats (Bayesian)
        self.max_log_likelihood = 0.0
        self.max_log_likelihood_err = 0.0
        self.bic = 0.0
        self.bic_err = 0.0
        self.log_bayes_factor = 0.0
        self.log_evidence = 0.0
        self.log_evidence_err = 0.0
        self.log_noise_evidence = 0.0

        # other
        self._is_sigma_sampled = False
        self._is_stats = False
        self._is_fit = False
        self.residuals = True
        self.plotPosterior = True



        # run set function for initalised attributes
        self.set(**kwargs)

    


    def set(self, **kwargs):

        # get class kwargs
        fit_kwargs = {}
        for key in kwargs:
            if key in _fit.attr:
                fit_kwargs[key] = kwargs[key]

        # set class attributes
        self._set_attr(**fit_kwargs)

        # check if all params exist
        self._check_params_keys()




    def _set_attr(self, **kwargs):
        """
        Check kwargs
        """

        # prior and statics
        for pkey in ["prior", "static", "fit_keywords"]:
            if pkey in kwargs.keys():
                # check if the right type
                if type(kwargs[pkey]) != dict:
                    raise ValueError(f"[fit]: {pkey} must be of type 'dict'")
                else:
                    setattr(self, pkey, kwargs[pkey])
        
        # func
        if "func" in kwargs.keys():
            if not callable(kwargs["func"]):
                raise ValueError("[fit]: func must be a callable function")
            else:
                self.func = kwargs["func"]
        
        # method
        if "method" in kwargs.keys():
            if kwargs["method"] not in _fit.meth:
                raise ValueError(f"[fit]: Method of fitting must be one of the following types:\n {_fit.meth}")
            else:
                self.method = kwargs["method"]
        
        # data
        data_flag = False
        for pkey in ["x", "y", "xerr", "yerr"]:
            if pkey in kwargs.keys():
                data_flag = True
                setattr(self, pkey, kwargs[pkey])
        if data_flag:
            self._get_mask()

        # other
        if "residuals" in kwargs.keys():
            self.residuals = kwargs['residuals']

        # likelihood
        if "likelihood" in kwargs.keys():
            if issubclass(kwargs['likelihood'], PyfitLikelihood):
                self.likelihood = kwargs['likelihood']
            else:
                raise ValueError("Likelihood Class must be a sub-class of PyfitLikelihood")

                
    


    def _check_attr(self):
        """
        Check attributes and see if they are right types,
        if pass return 1, if fail return 0
        """

        # method 
        if not self.method in _fit.meth:
            raise ValueError(f"[fit]: Method of fitting must be one of the following:\n {_fit.meth}")
            return 0 

        # func
        if not callable(self.func):
            raise ValueError(f"[fit]: func must be a callable function")
            return 0

        # check posteriors and priors and statics
        for pkey in ["prior", "posterior", "static"]:
            if type(getattr(self, pkey)) != dict:
                raise ValueError(f"[fit]: {pkey} must be a dictionary of values")
                return 0

        # check data
        for pkey in ["x", "y"]:
            attr = getattr(self, pkey)
            if attr is None:
                raise ValueError(f"[fit]: Must specify {pkey}")
                return 0
            elif  not hasattr(attr, "__len__"):
                raise ValueError(f"[fit]: {pkey} data must be array-like")
                return 0
        
        if self.x.size != self.y.size:
            raise ValueError("[fit]: Length mismatch between x and y data")
            return 0
        
        
        # check yerr
        for pkey in ["xerr", "yerr"]:
            attr = getattr(self, pkey)
            if attr is not None:
                if hasattr(attr, "__len__"):
                    if attr.size != getattr(self, pkey[0]).size:
                        raise RuntimeError(f"[fit]: Array Length mismatch between {pkey} and {pkey[0]}!")
                        return 0
                elif isinstance(attr, float) or isinstance(attr, int):
                    pass
                else:
                    raise ValueError(f"[{pkey}] must be array-like or a float/int value!")

        # check likelihood
        if not issubclass(self.likelihood, PyfitLikelihood):
            raise ValueError("Likelihood Class must be a sub-class of PyfitLikelihood")

        
        # only if all passed
        return 1






    def _check_params_keys(self):
        """
        Check priors, statics and posteriors 
        """

        # check if the right attr types
        for pkey in ["prior", "static", "posterior", "bounds"]:
            if type(getattr(self, pkey)) != dict:
                print(f"[fit]: {pkey} is not of type 'dict'")
                return 0
        
        if not hasattr(self.keys, "__len__"):
            print("[fit]: Keys attribute must be array-like")
            return 0

        # check if keys of priors, statics and posteriors match
        full_keys = list(_merge_dicts(self.posterior, self.prior, self.static, self.bounds).keys())
        for key in self.keys:
            if key not in full_keys:
                full_keys += [key]

        # check function arguments
        for key in self._get_func_args():
            if key not in full_keys:
                full_keys += [key]
        
        # check sigma 
        if "sigma" not in full_keys:
            full_keys += ["sigma"]

        def _if_no_key_set_none(dic, keys):

            dict_keys = dic.keys()
            for key in full_keys:
                if key not in dic.keys():
                    dic[key] = None
            
            return dic
        
        # set params
        self.prior = _if_no_key_set_none(self.prior, full_keys)
        self.posterior = _if_no_key_set_none(self.posterior, full_keys)
        self.static = _if_no_key_set_none(self.static, full_keys)

        # set bounds
        for key in full_keys:
            if key not in self.bounds.keys():
                self.bounds[key] = [-math.inf, math.inf]

        self.keys = full_keys
    
        return 1

    




    def _check_params_types(self, func = True):
        """
        Check if params are the right type, 1 if passed, 0 if failed
        """

        # type of prior given fitting method
        if self.method == "least squares":
            typ = float
        elif self.method == "bayesian":
            typ = list
        
        # get arguments to check
        args = self.keys
        if func:
            args = self._get_func_args()

        # sigma parameter
        if self.yerr is None and self.method == "bayesian":
            if "sigma" not in args:
                args += ["sigma"]

        for key in args:
            # check if a static variable
            if self.static[key] is not None:
                # check if type float
                if not isinstance(self.static[key], float):
                    print(f"[fit]: {key} (static) must be of type float")
                    return 0
                
            # check
            else:

                if self.method == "least squares":
                    if isinstance(self.prior[key], list):
                        print(f"Converting bounds of least squares parameter {key} to median prior")
                        self.bounds[key] = self.prior[key].copy()
                        self.prior[key] = (self.prior[key][1] - self.prior[key][0])/2 + self.prior[key][0]
                    if not isinstance(self.prior[key], float):
                        print(f"[fit]: prior [{key}] = {self.prior[key]} must be of type 'float' for '{self.method}' fitting method")
                        return 0
                if self.method == "bayesian":
                    if not isinstance(self.prior[key], list):
                        print(f"[fit]: prior [{key}] = {self.prior[key]} must be of type 'list' for '{self.method}' fitting method")
                        return 0

        # passed
        return 1
                    





    def _init_params(self, func = True):
        """
        In the case that you use .fit() and no priors have been specified, 
        initilaise them
        """

        if not self._check_params_keys():
            return {}

        self._check_params_keys()

        args = self.keys

        if func:
            args = self._get_func_args()

        # sigma parameter
        if self.yerr is None and self.method == "bayesian":
            if "sigma" not in args:
                args += ["sigma"]

        for key in args:
            if self.prior[key] is None:
                if self.method == "least squares":
                    print(f"[fit]: unspecified parameter [{key}], setting {key} -> 1.0")
                    self.prior[key] = 1.0
                elif self.method == "bayesian":
                    print(f"[fit]: unspecified parameter [{key}], setting {key} -> [0.0, 1.0]")
                    self.prior[key] = [0.0, 1.0]
            
                
    


    def _iserr(self):
        """
        check if error is present to perform statistics

        """
        # if using lsq method
        if self.method == "least squares":
            if self.yerr is None:
                return 0
        
        elif self.method == "bayesian":
            if self.yerr is None and not self._is_sigma_sampled:
                return 0
        
        return 1




    # def _isnan(self):
    #     """
        
    #     Make mask of data incorporating any nan values
    #     """
    #     # nans in [x]
    #     mask = np.isnan(self.x)
    #     mask = np.concatenate((mask, np.isnan(self.y)))
        
    #     if self.yerr:
    #         mask = np.concatenate((mask, np.isnan(self.yerr)))

    #     self.mask = mask.copy()
    #     return

    def _get_mask(self):
        """
        Mask data, find common mask amoung x, y, xerr and yerr
        """
        # Nans in [x]
        self.mask = ~np.isnan(self.x)

        # Nans in [y]
        self.mask[np.isnan(self.y)] = False

        # nans in [xerr?]
        if self.xerr is not None:
            if hasattr(self.xerr, "__len__"):
                self.mask[np.isnan(self.xerr)] = False
        
        # nans in [yerr?]
        if self.yerr is not None:
            if hasattr(self.yerr, "__len__"):
                self.mask[np.isnan(self.yerr)] = False




        
    
    def _proc_data(self):
        """
        Perform pre-processing tasks on data before passing through fitting
        methods

        1. Mask any nans
        """
        xerr = None
        yerr = None
        
        # mask data

        if self.xerr is not None:
            if hasattr(self.xerr, "__len__"):
                xerr = self.xerr[self.mask]
            else:
                xerr = self.xerr
        if self.yerr is not None:
            if hasattr(self.yerr, "__len__"):
                yerr = self.yerr[self.mask]
            else:
                yerr = self.yerr
        
        return self.x[self.mask], self.y[self.mask], xerr, yerr

        

        
    
        




    def _get_func_args(self):
        """
        get fitting function arguments
        """
        if callable(self.func):
            return inspect.getfullargspec(self.func)[0][1:]
        else:
            return []
        

    def get_model(self, x = None):
        """
        Get model fit

        Parameters
        ----------
        x : array-like
            datapoints to evaluate model, if None will use x values already given to current instance

        Returns
        -------
        x : array-like
            x data
        y : array-like
            y data - evaluated model values
        """

        model_vals = self.get_post_val()
        if model_vals is None:
            return None

        if x is None:
            x = self.x

        return x, self.func(x, **model_vals)


    
    def get_post_val(self, func = True):
        """
        Get values/betfit of posterior
        """

        # check attributes
        if not self._check_attr():
            return

        self._check_params_keys()

        # check if posteriors valid
        func_keys = self._get_func_args()

        for key in func_keys:
            if self.posterior[key] is None:
                print(f"[fit] '{key}' posterior must be valid to plot model")
                return None
            elif not isinstance(self.posterior[key], _posterior):
                print(f"[fit] '{key}' posterior is of wrong type, must be of type '_posterior', something went wrong")
                return None

        args = self.keys

        if func:
            args = self._get_func_args()

        _vals = {}
        for key in args:
            if self.posterior[key] is None:
                _vals[key] = None
            else:
                _vals[key] = self.posterior[key].val
        
        return _vals


    def get_mean_err(self, func = True):
        """
        Get mean errors of posterior

        Parameters
        ----------
        func : bool, optional
            if true, retrieve errors of posteriors defined in function

        Returns
        -------
        _errs : dict
            dictionary of errors for posteriors

        """
        # check attributes
        if not self._check_attr():
            return

        self._check_params_keys()

        # check if posteriors valid
        func_keys = self._get_func_args()

        for key in func_keys:
            if self.posterior[key] is None:
                print(f"[fit] '{key}' posterior must be valid to plot model")
                return None
            elif not isinstance(self.posterior[key], _posterior):
                print(f"[fit] '{key}' posterior is of wrong type, must be of type '_posterior', something went wrong")
                return None

        args = self.keys

        if func:
            args = self._get_func_args()

        _errs = {}
        for key in args:
            if self.posterior[key] is None:
                _errs[key] = None
            else:
                _errs[key] = (abs(self.posterior[key].p) + abs(self.posterior[key].m))/2
        
        return _errs



    def get_priors(self, func = True):
        """
        get priors and statics

        Parameters
        ----------
        func : bool
            if true, only retrieve priors of parameters of function to be modelled, default is True
        """

        if not self._check_params_keys():
            return {}, {}

        args = self.keys

        if func:
            args = self._get_func_args()

        # get sigma arg
        if self.yerr is None and self.method == "bayesian":
            if "sigma" not in args:
                args += ["sigma"]
        
        priors = _dict_get(self.prior, args)
        statics = _dict_get(self.static, args)

        return priors, statics

    


    def get_posteriors(self, func = True):
        """
        Get posteriors

        Parameters
        ----------
        func : bool
            if true, only retrieve posteriors of parameters of function to be modelled, default is True

        Returns
        -------
        posteriors : dict
            Dictionary of posterior median/mean best values
        """

        if not self._check_params_keys():
            return {}

        self._check_params_keys()

        args = self.keys

        if func:
            args = self._get_func_args()

        # get sigma arg
        if self.yerr is None and self.method == "bayesian":
            if "sigma" not in args:
                args += ["sigma"]

        posteriors = _dict_get(self.posterior, args)

        return posteriors


    
    def set_prior(self, name, pr):
        """
        Set prior

        Parameters
        ----------
        name : str
            name of parameter
        pr : float or 2-element list
            prior value
        """

        self.prior[name] = pr
        return

    
    def set_posterior(self, name, val, plus, minus):
        """
        Set posterior

        Parameters
        ----------
        name : str
            name of posterior
        val : float
            bestfit value/median
        plus : float 
            positive std/err
        minus : float
            negative std/err

        """

        self.posterior[name] = _posterior(val, plus, minus)
        return




    def _curve_fit2posterior(self, val, err, keys):
        """
        Convert output of curve_fit to posteriors
        
        """
        # calc uncorrelated errors
        err = np.sqrt(np.diag(err))

        for i, key in enumerate(keys):
            self.posterior[key] = _posterior(val[i], err[i], err[i])
        
       

        # set statics to posterior
        func_args = self._get_func_args()
        for key in func_args:
            if self.static[key] is not None:
                self.posterior[key] = _posterior(self.static[key]) 
                
            
        return
        
    





    def _results2posterior(self, result):
        """
        Convert bilby .result output to posteriors
        """

        # get all parameters excluding log_likelihood
        par_keys = result.posterior.columns[:-2].tolist()       

        for key in par_keys:
            vals = result.get_one_dimensional_median_and_error_bar(key)

            # posterior
            self.posterior[key] = _posterior(vals.median, vals.plus, vals.minus)
        
        # set statics to posterior
        func_args = self._get_func_args()
        for key in func_args:
            if self.static[key] is not None:
                self.posterior[key] = _posterior(self.static[key])  

        if "sigma" in par_keys:
            self._is_sigma_sampled = True

        # set statistics
        self.log_bayes_factor = result.log_10_bayes_factor
        self.log_evidence = result.log_10_evidence
        self.log_evidence_err = result.log_10_evidence_err
        self.log_noise_evidence = result.log_10_noise_evidence

        # get log_likelihood estimates
        llval = result.get_one_dimensional_median_and_error_bar('log_likelihood')

        self.max_log_likelihood = llval.median
        self.max_log_likelihood_err = (llval.plus**2 + llval.minus**2)**0.5







    def fit(self, redo = False, **kwargs):
        """
        Fit to function

        Parameters 
        ----------
        redo : bool
            redo fitting (in case BILBY is being used, remove cached results of any previous fit), default is False
        """

        self._is_sigma_sampled = False

        #--------------------------#
        # data and parameter check #
        #--------------------------#

        # set attributes using given keywords
        self._set_attr(**kwargs)

        # check if attributes are in right format
        if not self._check_attr():
            return 

        # check if all fit params are present
        self._check_params_keys()

        # initialise parameters if unspecified
        self._init_params()
        
        # check if priors are in the right format given the method of fitting
        if not self._check_params_types():
            return

        # check if fit params attributes to use are in the right format
        priors, statics = self.get_priors()

        # # proc data before hand (i.e. masking)
        x, y, xerr, yerr = self._proc_data()



        #-----------------------------#
        # wrap function using statics #
        #-----------------------------#
        args_str = "lambda x"
        func_str = "func(x = x"

        # NOTE I am aware that Bilby has DeltaFunction priors I could use for static paramters,
        # however, trhis method, athough scary, works for both least squares and bilby. 


        priors_wrap = {}
        statics_wrap = {}
        # check each key if it is static or not, we don't want to add "sigma"
        # prior to wrapped function in case we are sampling that as well
        for key in priors.keys():
            if statics[key] is not None:
                if key != "sigma":
                    func_str += f", {key} = {statics[key]}"
                statics_wrap[key] = statics[key]
            
            else:
                if key != "sigma":
                    args_str += f", {key}"
                    func_str += f", {key} = {key}"
                priors_wrap[key] = priors[key]


        # print priors and statics infomation
        print(f"\nFitting [{self.func.__name__}] using [{self.method}]")
        print("The following priors will be sampled/fitted for")
        print("-----------------------------------------------")
        for key in priors_wrap.keys():
            print(f"{key}:  {priors_wrap[key]}")

        if len(statics_wrap) > 0:
            print("\n The following priors will be kept constant")
            print("---------------------------------------------")
            for key in statics_wrap.keys():
                print(f"{key}:  {statics_wrap[key]}")



        
        # put all together (NOTE: The executed function is CAREFULLY constructed, else I 
        # wouldn't even think about using the eval function)
        func_wrap = eval(args_str + " : " + func_str + ")", {'func':self.func})

        # fit using least squares
        if self.method == "least squares":

            # convert prior dict to list in order! convert bounds to curve_fit bounds
            priors_list = []
            b_min, b_max = [], []
            for key in inspect.getfullargspec(func_wrap)[0][1:]:
                priors_list += [priors_wrap[key]]                
                b_min += [self.bounds[key][0]]
                b_max += [self.bounds[key][1]]

            

            _val, _err = curve_fit(func_wrap, x, y, p0 = priors_list, 
                                    sigma = yerr, bounds = (b_min, b_max), **self.fit_keywords)

            self._curve_fit2posterior(_val, _err, inspect.getfullargspec(func_wrap)[0][1:])
            self.sigma = yerr


        # fit using bilby bayesian inference
        elif self.method == "bayesian":

            # clean previous sampling, if any
            if redo:
                outdir = "outdir"
                if "outdir" in self.fit_keywords.keys():
                    outdir = self.fit_keywords["outdir"]
                label = "label"
                if "label" in self.fit_keywords.keys():
                    label = self.fit_keywords["label"]
                _clean_bilby_run(outdir, label)


            keys = inspect.getfullargspec(func_wrap)[0][1:]
            # if yerr not specified, include sigma for sampling
            if self.yerr is None:
                keys += ['sigma']
            bil_priors = _dict_get(priors, keys)

            
            likelihood = self.likelihood(x = x, y = y,
                                        func = func_wrap, xerr = xerr, yerr = yerr)
            
            # run sampler
            result_ = bilby.run_sampler(likelihood = likelihood, priors = _priorUniform(bil_priors),
                                    **self.fit_keywords)

            # get posteriors
            self._results2posterior(result_)

            # REUSE LIKELIHOOD INSTANCE TO CALCULATE SIGMA
            likelihood.parameters = self.get_post_val()
            self.sigma = likelihood.get_sigma()

            # plot bilby outputs for diagnostic purposes
            if self.plotPosterior:
                result_.plot_with_data(func_wrap, x, y)
                result_.plot_corner()


        # last stats to save 
        self.nfitpar = len(priors_wrap)

        # run stats
        self._get_stats()

        self._is_fit = True
        
        return













    def _get_stats(self):
        """
        Get stats of last function fitting call
        """

        #-----------------------------#
        # Calculate Chi squared stats #
        #-----------------------------# 

        if not self._iserr():
            print(f"[fit]: No error/sigma present to perform statistics")
            return

        # check which sigma to use
        if self._is_sigma_sampled:
            sigma = self.posterior['sigma'].val
        else:
            sigma = self.sigma

        
        x, y, _, _ = self._proc_data()


        # degrees of freedom
        self.dof = y.size - self.nfitpar

        # calculate chi squared and reduced chi squared
        self.chi2 = np.sum((y - self.get_model(x = x)[1])**2/(sigma**2))
        self.rchi2 = self.chi2/self.dof

        # calculate errors in chi squared and reduced chi squared
        self.chi2err = (2*self.dof)**0.5
        self.rchi2err = (2/self.dof)**0.5

        # calculate p value
        self.p = chi2.sf(self.chi2, self.dof)

        

        #---------------------#
        # Bayesian statistics #
        #---------------------#

        if self.method == "bayesian":
            # bayesian Information Criterion
            self.bic = self.nfitpar*np.log(y.size) - 2*self.max_log_likelihood
            self.bic_err = 2*self.max_log_likelihood_err

        
        self._is_stats = True







    def stats(self):
        """
        Print statistics
        """

        if not self._is_stats:
            return

        # print general statistics
        print("\nModel Statistics:")
        print("---------------------------")
        print("chi2:".ljust(30) + f"{self.chi2:.4f}".ljust(10) + f"+/- {self.chi2err:.4f}")
        print("rchi2:".ljust(30) + f"{self.rchi2:.4f}".ljust(10) + f"+/- {self.rchi2err:.4f}")
        print("p-value:".ljust(30) + f"{self.p:.4f}")
        print("v (degrees of freedom):".ljust(30) + f"{self.dof}")
        print("# free parameters:".ljust(30) + f"{self.nfitpar}")

        if self.method != "bayesian":
            return

        # print bayesian statistics if nessesary
        print("\nBayesian Statistics:")
        print("---------------------------")
        print("Max Log Likelihood:".ljust(30) + 
            f"{self.max_log_likelihood:.4f}".ljust(10) + f"+/- {self.max_log_likelihood_err:.4f}")
        print("Bayes Info Criterion (BIC):".ljust(30) + 
            f"{self.bic:.4f}".ljust(10) + f"+/- {self.bic_err:.4f}")
        print("Bayes Factor (log10):".ljust(30) + f"{self.log_bayes_factor:.4f}")
        print("Evidence (log10):".ljust(30) + 
            f"{self.log_evidence:.4f}".ljust(10) + f"+/- {self.log_evidence_err:.4f}")
        print("Noise Evidence (log10):".ljust(30) + f"{self.log_noise_evidence:.4f}")

        














    def plot(self, show = True, filename = None, **ax_kw):
        """
        Plot fitted model and data

        Parameters
        ----------
        show : bool, optional
            if true, display plot 
        filename : str
            name of output image file, default is None
        **ax_kw : Dict
            keyword parameters for plotting

        Returns
        -------
        fig : plt._figure_
            figure instance
        """

        if not self._is_fit:
            print("[fit]: No Model fit found, fit to data before plotting")
            return
        
        rows = 1
        if self.residuals:
            rows = 2

        # now plot, create axis
        fig, AX = plt.subplots(rows, 1, figsize = (12, 10), 
                    num = f"Figure model: {self.func.__name__}", sharex = True)

        if self.residuals:
            AX = AX.flatten()
        else:
            AX = [AX]
        
        # seperate keywords
        AX[0].set(**ax_kw)

        if self.residuals:
            AX[1].set(**ax_kw)
            AX[0].get_xaxis().set_visible(False)

        # plot data
        AX[0].scatter(self.x, self.y, c = 'k', s = 10)

        # plot model
        _, y_fit = self.get_model()
        if y_fit is None:
            return
        AX[0].plot(self.x, y_fit, color = [0.9098, 0.364, 0.3961], linewidth = 2.5)

        # plot errorbars if specified
        if (self.yerr is not None) or (self.xerr is not None):
            AX[0].errorbar(x = self.x, y = self.y, xerr = self.xerr, yerr = self.yerr, color = 'k', 
                        alpha = 0.4, linestyle = '')

        # plot residuals
        if self.residuals:
            AX[1].scatter(self.x, self.y - y_fit, c = 'k', s = 10)
            if (self.yerr is not None) or (self.xerr is not None):
                AX[1].errorbar(x = self.x, y = self.y - y_fit, xerr = self.xerr,
                                 yerr = self.yerr, color = 'k', alpha = 0.4, linestyle = '')

        # last figure changes
        fig.tight_layout()        
        fig.subplots_adjust(hspace = 0)

        # save file
        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()

        return fig



        






    def __str__(self):
        """
        Print infomation abou fit instance
        """

        self._check_params_keys()

        pstr = ""

        # function being used to fit data
        func_name = None
        if self.func is not None:
            func_name = self.func.__name__
        pstr += f"\nFitting data to [{func_name}] using [{self.method}]\n"
        
        # add data info to string
        pstr += "\nDATA\n"
        pstr += "--------\n"
        for d in ["x", "y", "yerr"]:
            attr = getattr(self, d)
            if attr is not None:
                if hasattr(attr, "__len__"):
                    o = list(attr.shape)
                elif isinstance(attr, float) or isinstance(attr, int):
                    o = attr
            else:
                o = None
            pstr += f"{d}:".ljust(10) + f"{o}\n"
        
        # priors and posteriors
        pstr += "\nParameter".ljust(25) + "Priors".ljust(26) + "Posteriors".ljust(30) + "\n"
        pstr += "-"*81 + "\n"
        for key in self.keys:
            name = key
            val = self.prior[key]
            if self.static[key] is not None:
                name += "  (static)"
                val = self.static[key]
            pstr += f"{name}".ljust(25) + f"{val}".ljust(26) + f"{self.posterior[key]}".ljust(30) + "\n"

        return pstr



















