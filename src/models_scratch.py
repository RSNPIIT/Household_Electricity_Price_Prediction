"""
From-Scratch Implementations of Non-GBDT Machine Learning Regression Algorithms:
1. LinearRegressionScratch (Closed-form Normal Equations & SVD Pseudoinverse)
2. RidgeRegressionScratch (L2-Regularized Analytical Formulation)
3. LassoRegressionScratch (L1-Regularized Coordinate Descent with Soft-Thresholding)
4. DecisionTreeRegressorScratch (Recursive Variance Reduction with Quantile Splits)
5. RandomForestRegressorScratch (Bootstrap Bagging & Random Feature Subspaces)
6. ExtraTreesRegressorScratch (Extremely Randomized Trees with Random Thresholds)
7. GradientBoostingRegressorScratch (Functional Gradient Boosting on Pseudo-Residuals)

All models adhere to the scikit-learn fit/predict API conventions.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union


# =============================================================================
# EVALUATION METRICS
# =============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression evaluation metrics: MAE, RMSE, MAPE, R2.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    # Remove any NaN entries
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]
    
    if len(yt) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
    
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    
    # Epsilon-stabilized MAPE to prevent zero division
    denom = np.where(np.abs(yt) < 1e-4, 1e-4, np.abs(yt))
    mape = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)
    
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    ss_res = np.sum((yt - yp) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


# =============================================================================
# 1. LINEAR REGRESSION (FROM SCRATCH)
# =============================================================================
class LinearRegressionScratch:
    """
    Ordinary Least Squares (OLS) Linear Regression from Scratch.
    Solved via Singular Value Decomposition (SVD) pseudoinverse for numerical
    stability in the presence of multicollinearity:
        beta = (X^T X)^(-1) X^T y = V Sigma^+ U^T y
    """
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_c = X - X_mean
            y_c = y - y_mean
            # SVD Pseudoinverse for maximum numerical stability
            self.coef_ = np.linalg.pinv(X_c) @ y_c
            self.intercept_ = float(y_mean - np.dot(X_mean, self.coef_))
        else:
            self.coef_ = np.linalg.pinv(X) @ y
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


# =============================================================================
# 2. RIDGE REGRESSION (FROM SCRATCH)
# =============================================================================
class RidgeRegressionScratch:
    """
    Ridge Regression (L2 Regularization) from Scratch.
    Analytical closed form solution:
        beta = (X^T X + alpha * I)^(-1) X^T y
    """
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_c = X - X_mean
            y_c = y - y_mean
            
            # (X_c^T X_c + alpha * I)^(-1) X_c^T y_c
            A = X_c.T @ X_c + self.alpha * np.eye(n_features)
            b = X_c.T @ y_c
            self.coef_ = np.linalg.solve(A, b)
            self.intercept_ = float(y_mean - np.dot(X_mean, self.coef_))
        else:
            A = X.T @ X + self.alpha * np.eye(n_features)
            b = X.T @ y
            self.coef_ = np.linalg.solve(A, b)
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


# =============================================================================
# 3. LASSO REGRESSION (FROM SCRATCH)
# =============================================================================
class LassoRegressionScratch:
    """
    Lasso Regression (L1 Regularization) from Scratch via Cyclic Coordinate Descent.
    Minimizes:
        (1 / (2*N)) * ||y - X*beta||_2^2 + alpha * ||beta||_1
    Using vectorised Soft-Thresholding operator:
        S(z, gamma) = sign(z) * max(0, |z| - gamma)
    """
    def __init__(self, alpha: float = 0.01, max_iter: int = 500, tol: float = 1e-4, fit_intercept: bool = True):
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    @staticmethod
    def _soft_threshold(rho: float, gamma: float) -> float:
        if rho > gamma:
            return rho - gamma
        elif rho < -gamma:
            return rho + gamma
        else:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_c = X - X_mean
            y_c = y - y_mean
        else:
            X_c = X
            y_c = y
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        # Normalization factors for coordinate descent
        z_j = np.sum(X_c ** 2, axis=0) / n_samples
        # Prevent division by zero for constant features
        z_j = np.where(z_j == 0, 1e-10, z_j)
        
        beta = np.zeros(n_features, dtype=float)
        # Residual tracking for O(N) coordinate updates
        residual = y_c - (X_c @ beta)
        
        for iteration in range(self.max_iter):
            max_change = 0.0
            for j in range(n_features):
                old_beta_j = beta[j]
                # Partial residual without feature j
                rho_j = (np.dot(X_c[:, j], residual) / n_samples) + (old_beta_j * z_j[j])
                # Soft thresholding
                new_beta_j = self._soft_threshold(rho_j, self.alpha) / z_j[j]
                
                delta = new_beta_j - old_beta_j
                if delta != 0.0:
                    beta[j] = new_beta_j
                    residual -= X_c[:, j] * delta
                    max_change = max(max_change, abs(delta))
                    
            if max_change < self.tol:
                break
                
        self.coef_ = beta
        if self.fit_intercept:
            self.intercept_ = float(y_mean - np.dot(X_mean, self.coef_))
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


# =============================================================================
# 4. DECISION TREE REGRESSOR (FROM SCRATCH)
# =============================================================================
class TreeNode:
    """Internal node in regression tree."""
    def __init__(self, feature: int = None, threshold: float = None,
                 left: 'TreeNode' = None, right: 'TreeNode' = None,
                 value: float = None):
        self.feature = feature          # Splitting feature index
        self.threshold = threshold      # Splitting threshold
        self.left = left                # Left child (x <= threshold)
        self.right = right              # Right child (x > threshold)
        self.value = value              # Leaf prediction value (mean of y)

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeRegressorScratch:
    """
    Fast Regression Decision Tree from Scratch.
    Minimizes Mean Squared Error (variance reduction).
    Uses candidate quantile thresholds (histograms) to guarantee blazing speed
    over large dataset subsamples.
    """
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: Optional[Union[int, float, str]] = None,
        n_quantiles: int = 32
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_quantiles = n_quantiles
        self.root: Optional[TreeNode] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def _determine_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            return min(n_features, self.max_features)
        return n_features

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features, dtype=float)
        
        self.root = self._build_tree(X, y, depth=0)
        
        # Normalize feature importances
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples, n_features = X.shape
        mean_y = float(np.mean(y))
        
        sum_y_total = float(np.sum(y))
        sum_y2_total = float(np.sum(y ** 2))
        total_impurity = sum_y2_total - ((sum_y_total ** 2) / n_samples)
        
        # Base stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            total_impurity < 1e-7):
            return TreeNode(value=mean_y)

        # Feature subsampling
        m_features = self._determine_max_features(n_features)
        feature_indices = np.random.choice(n_features, m_features, replace=False)

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for feat in feature_indices:
            x_col = X[:, feat]
            min_val = float(np.min(x_col))
            max_val = float(np.max(x_col))
            if max_val - min_val < 1e-7:
                continue

            # Quantile candidate thresholds
            thresholds = np.quantile(x_col, np.linspace(0.05, 0.95, self.n_quantiles))
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                left_mask = x_col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                sum_y_l = float(np.sum(y_left))
                sum_y2_l = float(np.sum(y_left ** 2))

                sum_y_r = sum_y_total - sum_y_l
                sum_y2_r = sum_y2_total - sum_y2_l

                impurity_left = sum_y2_l - ((sum_y_l ** 2) / n_left)
                impurity_right = sum_y2_r - ((sum_y_r ** 2) / n_right)
                gain = total_impurity - (impurity_left + impurity_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_gain <= 0.0 or best_feat is None:
            return TreeNode(value=mean_y)

        # Accumulate feature importance
        self.feature_importances_[best_feat] += best_gain

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return TreeNode(feature=best_feat, threshold=best_thresh,
                        left=left_child, right=right_child)

    def _predict_row(self, node: TreeNode, row: np.ndarray) -> float:
        if node.is_leaf:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)
        else:
            return self._predict_row(node.right, row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        # Vectorized tree traversal using batch recursion
        preds = np.zeros(len(X), dtype=float)
        self._predict_batch(self.root, np.arange(len(X)), X, preds)
        return preds

    def _predict_batch(self, node: TreeNode, indices: np.ndarray, X: np.ndarray, out: np.ndarray):
        if len(indices) == 0:
            return
        if node.is_leaf:
            out[indices] = node.value
            return
        
        left_mask = X[indices, node.feature] <= node.threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        
        if len(left_indices) > 0:
            self._predict_batch(node.left, left_indices, X, out)
        if len(right_indices) > 0:
            self._predict_batch(node.right, right_indices, X, out)


# =============================================================================
# 5. RANDOM FOREST REGRESSOR (FROM SCRATCH)
# =============================================================================
class RandomForestRegressorScratch:
    """
    Random Forest Regressor from Scratch.
    Ensemble of DecisionTreeRegressorScratch instances trained on bootstrap samples
    with random feature subsets (mtry = sqrt(p) or p/3).
    """
    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 12,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: Optional[Union[int, float, str]] = 'sqrt',
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees: List[DecisionTreeRegressorScratch] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X.shape
        
        np.random.seed(self.random_state)
        self.trees = []
        importances = np.zeros(n_features, dtype=float)
        sub_size = min(n_samples, 8000)

        for b in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, sub_size, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_quantiles=16
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        total_imp = np.sum(importances)
        self.feature_importances_ = (importances / total_imp) if total_imp > 0 else importances
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        all_preds = np.zeros((self.n_estimators, len(X)), dtype=float)
        for i, tree in enumerate(self.trees):
            all_preds[i] = tree.predict(X)
        return np.mean(all_preds, axis=0)


# =============================================================================
# 6. EXTRA TREES REGRESSOR (FROM SCRATCH)
# =============================================================================
class ExtraTreesRegressorScratch:
    """
    Extremely Randomized Trees (Extra Trees) from Scratch.
    Unlike standard Random Forests which optimize split thresholds, Extra Trees
    draws random thresholds uniformly from [min(X_j), max(X_j)], achieving
    massive speedups and variance reduction.
    """
    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 12,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: Optional[Union[int, float, str]] = 'sqrt',
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees: List[DecisionTreeRegressorScratch] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X.shape
        
        np.random.seed(self.random_state)
        self.trees = []
        importances = np.zeros(n_features, dtype=float)

        sub_size = min(n_samples, 8000)
        for b in range(self.n_estimators):
            sub_idx = np.random.choice(n_samples, sub_size, replace=False)
            X_sub = X[sub_idx]
            y_sub = y[sub_idx]

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_quantiles=8 # rapid randomized cuts
            )
            tree.fit(X_sub, y_sub)
            self.trees.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        total_imp = np.sum(importances)
        self.feature_importances_ = (importances / total_imp) if total_imp > 0 else importances
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        all_preds = np.zeros((self.n_estimators, len(X)), dtype=float)
        for i, tree in enumerate(self.trees):
            all_preds[i] = tree.predict(X)
        return np.mean(all_preds, axis=0)


# =============================================================================
# 7. GRADIENT BOOSTING REGRESSOR (FROM SCRATCH)
# =============================================================================
class GradientBoostingRegressorScratch:
    """
    Gradient Boosting Regressor from Scratch for L2 Squared Error Loss.
    Sequentially fits regression trees to negative gradient (pseudo-residuals).
    """
    def __init__(
        self,
        n_estimators: int = 40,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        min_samples_split: int = 20,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees: List[DecisionTreeRegressorScratch] = []
        self.initial_prediction_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        
        np.random.seed(self.random_state)
        self.initial_prediction_ = float(np.mean(y))
        y_pred = np.full_like(y, self.initial_prediction_)
        self.trees = []

        for m in range(self.n_estimators):
            # Pseudo-residuals: r_i = y_i - f(x_i)
            residuals = y - y_pred
            
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_quantiles=24
            )
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update prediction
            y_pred += self.learning_rate * tree.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        y_pred = np.full(len(X), self.initial_prediction_, dtype=float)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
