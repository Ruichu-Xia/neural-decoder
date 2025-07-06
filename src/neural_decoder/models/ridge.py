import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge


class RidgeWrapper(nn.Module):
    """
    Pytorch wrapper for Ridge regression.
    """
    def __init__(self, alpha: float = 1, 
                 fit_intercept: bool = True, 
                 copy_X: bool = True, 
                 max_iter: int = 1000, 
                 tol: float = 0.0001, 
                 solver: str = "auto", 
                 positive: bool = False, 
                 random_state: int = 42):
        super().__init__()
        self.ridge = Ridge(
            alpha=alpha, 
            fit_intercept=fit_intercept, 
            copy_X=copy_X,  
            max_iter=max_iter, 
            tol=tol, 
            solver=solver, 
            positive=positive, 
            random_state=random_state)

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return torch.from_numpy(self.ridge.predict(x))

    def fit(self, x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray) -> None:
        """
        Fit the Ridge regression model on numpy arrays.

        Args:
            x: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        self.ridge.fit(x, y)

    def state_dict(self):
        """
        Custom state_dict to save scikit-learn's parameters.
        """
        if not hasattr(self.ridge, 'coef_'):
            return {}

        state = {
            'coef': torch.from_numpy(self.ridge.coef_),
        }
        if self.ridge.fit_intercept:
            state['intercept'] = torch.from_numpy(np.array(self.ridge.intercept_))
        return state

    def load_state_dict(self, state_dict):
        """
        Custom load_state_dict to load scikit-learn's parameters.
        """
        self.ridge.coef_ = state_dict['coef'].cpu().numpy()
        if 'intercept' in state_dict:
            self.ridge.intercept_ = state_dict['intercept'].cpu().numpy()