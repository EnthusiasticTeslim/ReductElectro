
from botorch.models import SingleTaskGP
from gpytorch.kernels import RBFKernel, LinearKernel, ScaleKernel
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model

# construct and fit GP model
def construct_and_fit_gp_model(X_train, y_train):      
    model = SingleTaskGP(X_train, y_train, covar_module=ScaleKernel(RBFKernel()+LinearKernel()), outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_model(mll)
    return model