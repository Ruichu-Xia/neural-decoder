import torch
import torch.nn as nn
import numpy as np
import joblib
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