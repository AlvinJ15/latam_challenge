import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Tuple, Union, List
from datetime import datetime

from sklearn.model_selection import train_test_split


class DelayModel:

    def __init__(
            self
    ):
        self._columns = None
        self._threshold_in_minutes = 15
        self._columns_chosen = ['OPERA', 'TIPOVUELO', 'MES']
        self._top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        self.data = pd.read_csv(filepath_or_buffer="data/data.csv")
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        features, target = self.preprocess(
            data=self.data,
            target_column="delay"
        )
        _, features_validation, _, target_validation = train_test_split(features, target, test_size=0.33,
                                                                        random_state=42)

        self.fit(
            features=features,
            target=target
        )

    def _get_min_diff(self, data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def _get_scale(self, y_values):
        n_y0 = len(y_values[y_values == 0])
        n_y1 = len(y_values[y_values == 1])
        scale = n_y0 / n_y1
        return scale

    def preprocess(
            self,
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        if target_column is not None:
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)
            data[target_column] = np.where(data['min_diff'] > self._threshold_in_minutes, 1, 0)
            scale = self._get_scale(data[target_column])
            self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)

            features = pd.concat(
                [pd.get_dummies(data[column], prefix=column) for column in self._columns_chosen],
                axis=1
            )
            self._columns = features.columns.tolist()
            features = features[self._top_10_features]
            return features, pd.DataFrame(data[target_column])
        else:
            features = pd.concat(
                [pd.get_dummies(data[column], prefix=column) for column in self._columns_chosen],
                axis=1
            )
            features = features.reindex(columns=self._columns).fillna(0.00)
            features = features[self._top_10_features]
            return features

    def fit(
            self,
            features: pd.DataFrame,
            target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target)

    def predict(
            self,
            features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        predictions = self._model.predict(features)
        predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        return predictions
