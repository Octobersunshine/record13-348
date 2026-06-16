import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, Union

METRIC_MAPPING = {
    "manhattan": "cityblock",
    "euclidean": "euclidean",
    "cosine": "cosine",
    "chebyshev": "chebyshev",
    "minkowski": "minkowski",
}


class NearestNeighborMatcher:
    def __init__(
        self,
        n_neighbors: int = 1,
        metric: str = "euclidean",
        with_replacement: bool = False,
        standardize: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.with_replacement = with_replacement
        self.standardize = standardize
        self.random_state = random_state
        self.scaler_ = None
        self.treated_indices_ = None
        self.control_indices_ = None
        self.matches_ = None
        self.distances_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        treated: Union[np.ndarray, pd.Series, list],
    ) -> "NearestNeighborMatcher":
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(treated, (pd.Series, list)):
            treated = np.array(treated)

        self.treated_indices_ = np.where(treated == 1)[0]
        self.control_indices_ = np.where(treated == 0)[0]

        if len(self.treated_indices_) == 0 or len(self.control_indices_) == 0:
            raise ValueError("Both treated and control groups must have at least one sample.")

        X_treated = X[self.treated_indices_]
        X_control = X[self.control_indices_]

        if self.standardize:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)
            X_treated = self.scaler_.transform(X_treated)
            X_control = self.scaler_.transform(X_control)

        self._X_treated = X_treated
        self._X_control = X_control

        return self

    def transform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.treated_indices_ is None or self.control_indices_ is None:
            raise ValueError("Matcher has not been fitted. Call fit() first.")

        if self.with_replacement:
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
            nbrs.fit(self._X_control)
            distances, indices = nbrs.kneighbors(self._X_treated)
            matched_control_indices = self.control_indices_[indices]
            self.distances_ = distances
        else:
            matched_control_indices = self._match_without_replacement()
            distances = self.distances_

        self.matches_ = matched_control_indices

        return self.treated_indices_, matched_control_indices, distances

    def _match_without_replacement(self) -> np.ndarray:
        n_treated = len(self.treated_indices_)
        n_control = len(self.control_indices_)
        n_neighbors = min(self.n_neighbors, n_control)

        if n_treated * n_neighbors > n_control:
            raise ValueError(
                f"Not enough control samples. Need {n_treated * n_neighbors} but only {n_control} available."
            )

        if n_neighbors == 1:
            return self._match_one_to_one()
        else:
            return self._match_k_to_one(n_neighbors)

    def _get_scipy_metric(self) -> str:
        return METRIC_MAPPING.get(self.metric, self.metric)

    def _match_one_to_one(self) -> np.ndarray:
        n_treated = len(self.treated_indices_)
        n_control = len(self.control_indices_)

        dist_matrix = cdist(self._X_treated, self._X_control, metric=self._get_scipy_metric())

        if n_treated <= n_control:
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
        else:
            row_ind, col_ind = linear_sum_assignment(dist_matrix.T)
            col_ind, row_ind = row_ind, col_ind

        result = col_ind.reshape(-1, 1)

        self._update_distances(result)

        return self.control_indices_[result]

    def _match_k_to_one(self, n_neighbors: int) -> np.ndarray:
        n_treated = len(self.treated_indices_)
        n_control = len(self.control_indices_)

        dist_matrix = cdist(self._X_treated, self._X_control, metric=self._get_scipy_metric())

        result = np.zeros((n_treated, n_neighbors), dtype=int)
        all_distances = np.zeros((n_treated, n_neighbors))
        used_control = set()

        all_pairs = []
        for i in range(n_treated):
            for j in range(n_control):
                all_pairs.append((dist_matrix[i, j], i, j))

        all_pairs.sort(key=lambda x: x[0])

        matched_count = np.zeros(n_treated, dtype=int)
        filled = 0

        for dist, i, j in all_pairs:
            if matched_count[i] >= n_neighbors:
                continue
            if j in used_control:
                continue

            result[i, matched_count[i]] = j
            all_distances[i, matched_count[i]] = dist
            matched_count[i] += 1
            used_control.add(j)
            filled += 1

            if filled >= n_treated * n_neighbors:
                break

        self.distances_ = all_distances

        return self.control_indices_[result]

    def _update_distances(self, matched_indices: np.ndarray):
        n_treated = len(self.treated_indices_)
        dist_matrix = cdist(self._X_treated, self._X_control, metric=self._get_scipy_metric())

        distances = np.zeros((n_treated, 1))
        for i in range(n_treated):
            distances[i, 0] = dist_matrix[i, matched_indices[i, 0]]

        self.distances_ = distances

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        treated: Union[np.ndarray, pd.Series, list],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.fit(X, treated)
        return self.transform()

    def get_matched_dataframe(
        self,
        df: pd.DataFrame,
        treated_col: str = "treated",
    ) -> pd.DataFrame:
        if self.matches_ is None:
            raise ValueError("No matches found. Call fit_transform() first.")

        treated_df = df.iloc[self.treated_indices_].copy()
        treated_df["match_id"] = np.arange(len(self.treated_indices_))
        treated_df["group"] = "treated"
        treated_df["match_distance"] = self.distances_[:, 0] if self.n_neighbors == 1 else None

        control_rows = []
        for i in range(len(self.treated_indices_)):
            for j in range(self.n_neighbors):
                control_idx = self.matches_[i, j]
                row = df.iloc[control_idx].copy()
                row["match_id"] = i
                row["group"] = "control"
                row["match_distance"] = self.distances_[i, j]
                control_rows.append(row)

        control_df = pd.DataFrame(control_rows)

        result_df = pd.concat([treated_df, control_df], ignore_index=True)
        result_df = result_df.sort_values(["match_id", "group"]).reset_index(drop=True)

        return result_df

    def get_propensity_matched(
        self,
        propensity_scores: Union[np.ndarray, pd.Series],
        treated: Union[np.ndarray, pd.Series, list],
        caliper: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(propensity_scores, pd.Series):
            propensity_scores = propensity_scores.values
        if isinstance(treated, (pd.Series, list)):
            treated = np.array(treated)

        X = propensity_scores.reshape(-1, 1)

        self.fit(X, treated)

        if caliper is not None and self.scaler_ is not None:
            pass

        treated_idx, control_idx, distances = self.transform()

        if caliper is not None:
            mask = distances[:, 0] <= caliper
            treated_idx = treated_idx[mask]
            control_idx = control_idx[mask]
            distances = distances[mask]

        return treated_idx, control_idx, distances
