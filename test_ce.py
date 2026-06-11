import sys
import pandas as pd
import numpy as np
sys.path.append('.')
import new_liquidation_sharpe_v3 as mod

M = ['A','B','C','D','E','F']
mean_returns = pd.Series([0.20,0.15,0.18,0.12,0.10,0.16], index=M)
cov_matrix = pd.DataFrame(np.eye(len(M))*0.04, index=M, columns=M)

alloc,best = mod.cross_entropy_portfolio_selection(
    M, {}, 1000.0, mean_returns, cov_matrix,
    n_stocks=4, num_samples=200, elite_frac=0.2, alpha=0.7, max_iter=10, tol=1e-4, random_state=1
)
print('BEST', best)
print('ALLOC', alloc)
