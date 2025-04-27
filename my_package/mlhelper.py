import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from lightgbm import LGBMClassifier
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours, NearMiss, NeighbourhoodCleaningRule, RandomUnderSampler, TomekLinks
from IPython.display import HTML
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2, kurtosis, shapiro, skew
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, IsolationForest, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
from umap import UMAP
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)

class MLHelper():
    def __init__(self, dataset_name:str, labels_dict: dict[int: dict[str: str]]):
        self.dataset_name:str = dataset_name
        self.labels_dict: dict[int: dict[str: str]] = labels_dict

    def get_label_balance(self, label_column: pd.Series, should_return_label_counts=False) -> pd.Series:
        label_counts = label_column.value_counts().sort_index(ascending=True)
        num_labels = len(self.labels_dict.keys())
        labels = [f"{self.labels_dict.get(i).get('name')} {label_counts.loc[i]}" for i in range(num_labels)]
        cmap = cm.get_cmap('Set3')  # tab20 tab10 Set3
        colors = [cmap(i / num_labels) for i in range(num_labels)]
        plt.figure(figsize=(12, 10))
        plt.pie(label_counts, labels=labels, autopct='%1.1f%%', colors=colors, explode=[0.05, 0])
        plt.title(self.dataset_name + ' Labels Balance')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        return label_counts if should_return_label_counts else None
        
    def get_data_skew_kurtosis_shapiro(self, df: pd.DataFrame) -> pd.DataFrame:
        data_info = {
            'Skew': [],
            'Kurtosis': [],
            'Shapiro stat': [],
            'Shapiro p': []
        }
        for feature in df.columns[:-1]:
            stat, p = shapiro(df[feature].values)
            data_info.get('Skew').append(skew(df[feature]))
            data_info.get('Kurtosis').append(kurtosis(df[feature]))
            data_info.get('Shapiro stat').append(stat)
            data_info.get('Shapiro p').append(p)
        return pd.DataFrame().from_dict(data_info).set_index(df.columns[:-1])
            
    def get_distribution(self, df: pd.DataFrame):
        fig, axs = plt.subplots(figsize=(35,20), nrows=5, ncols=6)
        for i, feature in enumerate(df.iloc[:, 1:-1]):
            row = int(i/6)
            col = i%6
            sns.histplot(df[feature], bins=100, kde=True, ax=axs[row,col])
        plt.title(self.dataset_name + ' Distribution of Features')
        plt.tight_layout()
        plt.show()
        fig, axs = plt.subplots(figsize=(35,20), nrows=5, ncols=6)
        for i, feature in enumerate(df.iloc[:, 1:-1]):
            row = int(i/6)
            col = i%6
            axs[row,col].grid()
            stats.probplot(df[feature].values, dist='norm', plot=axs[row,col])
            axs[row, col].set_ylabel(feature)
        plt.title(self.dataset_name + ' Distribution of Features')
        plt.tight_layout()
        plt.show()
        
    def get_feature_correlation(self, df: pd.DataFrame):
        plt.figure(figsize=(12,10))
        plt.title(self.dataset_name + ' Correlations of Features')
        sns.heatmap(df.corr(), cmap='RdBu')
        plt.tight_layout()
        plt.show()
        
    def _visualize_lda(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       full_data: bool=True,
                       is_training: bool=True):
        training_or_testing = ''
        if not full_data:
            training_or_testing = ' Training' if is_training else ' Testing'
        plt.figure(figsize=(12, 10))
        for label, info in self.labels_dict.items():
            plt.hist(X[y == label], alpha=0.6, label=info.get('name'), bins=20, color=info.get('color'))
        plt.xlabel("LDA Component")
        plt.ylabel("Frequency")
        plt.title(self.dataset_name + training_or_testing + " Data LDA")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def full_data_extract_features_lda(self,
                                       X: pd.DataFrame,
                                       y: pd.Series,
                                       params) -> pd.DataFrame:
        lda = LinearDiscriminantAnalysis(**params)
        X_lda = lda.fit_transform(X, y)
        self._visualize_lda(X_lda, y)
    
    def extract_features_lda(self,
                             X_train: pd.DataFrame,
                             X_test: pd.DataFrame,
                             y_train: pd.Series,
                             y_test: pd.Series,
                             params,
                             should_return_fitted_data=False) -> list[pd.DataFrame]:
        lda = LinearDiscriminantAnalysis(**params)
        X_lda_train = lda.fit_transform(X_train, y_train)
        X_lda_test = lda.transform(X_test)
        self._visualize_lda(X_lda_train, y_train, False, True)
        self._visualize_lda(X_lda_test, y_test, False, False)
        return pd.DataFrame(X_lda_train), pd.DataFrame(X_lda_test) if should_return_fitted_data else None
    
    def _visualize_feature_extraction(self, 
                                      X: np.ndarray, 
                                      y: pd.Series, 
                                      method: str,
                                      full_data: bool=True,
                                      is_training: bool=True):
        training_or_testing = ''
        if not full_data:
            training_or_testing = ' Training' if is_training else ' Testing'
        plt.figure(figsize=(12, 10))
        for label, info in self.labels_dict.items():
            plt.scatter(
                X[y == label, 0],
                X[y == label, 1],
                color=info.get('color'),
                marker=info.get('marker'),
                s=10,
                label=info.get('name')
            )
        plt.title("2D " + self.dataset_name + training_or_testing + " Data " + method)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def full_data_extract_features_pca(self,
                             X: pd.DataFrame,
                             y: pd.Series,
                             params) -> pd.DataFrame:
        pca = PCA(**params)
        X_pca = pca.fit_transform(X)
        self._visualize_feature_extraction(X_pca, y, 'PCA')
        
    def extract_features_pca(self,
                             X_train: pd.DataFrame,
                             X_test: pd.DataFrame,
                             y_train: pd.Series,
                             y_test: pd.Series,
                             params,
                             should_return_fitted_data=False) -> list[pd.DataFrame]:
        pca = PCA(**params)
        X_pca_train = pca.fit_transform(X_train)
        X_pca_test = pca.transform(X_test)
        self._visualize_feature_extraction(X_pca_train, y_train, 'PCA', False, True)
        self._visualize_feature_extraction(X_pca_test, y_test, 'PCA', False, False)
        plt.figure(figsize=(12, 10))
        plt.title('PCA Explained Variance Ratio')
        plt.bar(range(0, len(X_pca_train[0])), pca.explained_variance_ratio_, align='center')
        plt.step(range(0, len(X_pca_train[0])), np.cumsum(pca.explained_variance_ratio_), where='mid')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.show()
        return pd.DataFrame(X_pca_train), pd.DataFrame(X_pca_test) if should_return_fitted_data else None

    def full_data_extract_features_tsne(self, 
                              X: pd.DataFrame,
                              y:pd.Series,
                              params):
        '''
        Only for visualizing data
        '''
        tsne = TSNE(**params)
        X_tsne = tsne.fit_transform(X)
        self._visualize_feature_extraction(X_tsne, y, 't-SNE')

    def full_data_extract_features_umap(self, 
                              X: pd.DataFrame,
                              y: pd.Series,
                              params) -> pd.DataFrame:
        umap = UMAP(**params)
        X_umap = umap.fit_transform(X)
        self._visualize_feature_extraction(X_umap, y, 'UMAP')
    
    def extract_features_umap(self,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series,
                              params,
                              should_return_fitted_data=False) -> list[pd.DataFrame]:
        umap = UMAP(**params)
        X_umap_train = umap.fit_transform(X_train)
        X_umap_test = umap.transform(X_test)
        self._visualize_feature_extraction(X_umap_train, y_train, 'UMAP', False, True)
        self._visualize_feature_extraction(X_umap_test, y_test, 'UMAP', False, False)
        
        return pd.DataFrame(X_umap_train), pd.DataFrame(X_umap_test) if should_return_fitted_data else None
    
    def animte_data_3D(self, X: pd.DataFrame, y: pd.Series, method: str) -> HTML:
        '''
        - method
            - PCA
            - t-SNE
            - UMAP
        '''
        METHODS = {
            'PCA': PCA(n_components=3),
            't-SNE': TSNE(n_components=3, random_state=42, n_jobs=-1),
            'UMAP': UMAP(n_components=3, n_jobs=-1)
        }
        if extraction_method := METHODS.get(method):
            X_reduced = extraction_method.fit_transform(X)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            for label, info in self.labels_dict.items():
                target = X_reduced[y == label]
                ax.scatter(target[:, 0], 
                        target[:, 1], 
                        target[:, 2],
                        c=info.get('color'), 
                        marker=info.get('marker'), 
                        edgecolor='k',
                        s=40, 
                        label=info.get('name'))
            ax.set_title("3D " + self.dataset_name + " Data " + method)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.legend([info.get('name') for info in self.labels_dict.values()])

            def rotate(angle):
                ax.view_init(elev=30, azim=angle)

            ani = FuncAnimation(fig, rotate, frames=range(0, 360, 2), interval=50)
            plt.close(fig)  # prevent static plot
            return HTML(ani.to_jshtml())
        else:
            print(method, 'is not a valid plotting method')
    
    def get_transformed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for feature in df_copy.columns:
            feature_series = df_copy[feature]
            skew_ = skew(feature_series)
            kurtosis_ = kurtosis(feature_series)
            if skew_ > 1 and kurtosis_ > 3:
                df_copy[feature] = np.log1p(feature_series)
            elif skew_ < -1 and kurtosis_ > 3:
                df_copy[feature] = np.sqrt(feature_series)
            else:
                df_copy[feature] = feature_series
        return df_copy
    
    def get_scaled_data(self,
                        X_train: pd.DataFrame, 
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        scaler: str,
                        params=None):
        '''
        - mas: MaxAbsScaler
            - range becomes -1 to 1
            - not sensitive to outliers
            - data does not have to follow Gaussian distribution
            - use when data is centered around 0
        - mm: MinMaxScaler
            - range becomes 0 to 1
            - sensitive to outliers
            - use when using models sensitive to scale (linear, distance, gradient descent based models)
        - pt: PowerTransformer
            - params
                - method: ["yeo-johnson", "box-cox"]]
            - scales data and make it follow Gaussian distribution
            - mean becomes 0
            - standard deviation becomes 1
            - use when data is skewed
        - rs: RobustScaler
            - range becomes IQR (25th and 75th) centered at median
            - even less sensitive to outliers
            - no assumption about data distribution
            - use when data has significant outliers or is skewed
        - ss: StandardScaler
            - range becomes -1 to 1
            - less sensitive to outliers
            - use when data follows Gaussian distribution
        '''
        scalers = {
            'mas': MaxAbsScaler(),
            'mm': MinMaxScaler(),
            'pt': PowerTransformer(**params),
            'rb': RobustScaler(),
            'ss': StandardScaler()
        }
        if scaler := scalers.get(scaler):
            X_train_scaled: pd.DataFrame = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=y_train.index)
            X_test_scaled: pd.DataFrame = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=y_test.index)
            return X_train_scaled, X_test_scaled
        else:
            print(scaler, 'is not a valid scaler')
            return None

    def _get_outlier_dbscan(self, df: pd.DataFrame, params) -> pd.Index:
        dbscan = DBSCAN(**params)
        labels = dbscan.fit_predict(df)
        return df[labels == -1].index

    def _get_outlier_isolation_forest(self, df: pd.DataFrame, params) -> pd.Index:
        isolation_forest = IsolationForest(**params)
        labels = isolation_forest.fit_predict(df)
        return df[labels == -1].index

    def _get_outlier_iqr(self, df: pd.DataFrame, params) -> np.ndarray:
        weight=1.5
        outliers = []
        for feature in df.columns:
            series = df[feature]
            q_25 = np.percentile(series.values, 25)
            q_75 = np.percentile(series.values, 75)
            iqr = q_75 - q_25
            iqr_weight = iqr * weight
            low_value = q_25 - iqr_weight
            high_value = q_75 + iqr_weight
            outliers += (series[(series < low_value) | (series > high_value)].index).tolist()
        return np.unique(np.array(outliers))

    def _get_outlier_local_outlier_factor(self, df: pd.DataFrame, params) -> pd.Index:
        lof = LocalOutlierFactor(**params)
        y_pred = lof.fit_predict(df)  # -1 for outliers, 1 for inliers
        return df[y_pred == -1].index

    def _get_outlier_mahalanobis(self, df: pd.DataFrame, params) -> pd.Index:
        mean_vec = np.mean(df, axis=0)
        cov_matrix = np.cov(df, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mahal_distances = np.array([mahalanobis(x, mean_vec, inv_cov_matrix) for x in df.values])
        threshold = np.sqrt(chi2.ppf(0.999, df=df.shape[1])) 
        return df[mahal_distances > threshold].index
            
    def remove_outliers(self, 
                        df: pd.DataFrame, 
                        method: str, 
                        params):
        '''
        - method
            - dbscan
                - use when 
                    - data has clusters of varying shapes and densities and want to detect outliers that do not belong to any cluster
                    - data doesn't follow Gaussian distribution
                - dont use when 
                    - data is uniformly distributed
                    - clusters are similar sizes
                - feature transformation recommended
                - feature scaling recommended
            - if
                - use when 
                    - data is large and high-dimensional
                    - data contains outliers that are distinct and separate
                    - data is large
                - dont use when
                    - data is small
                    - data has few outliers
            - iqr
                - use when
                    - data is concentrated within a known range
                    - data is univariate (single feature)
                - dont use when
                    - data contains many extreme values
                    - data is skewed
                    - data is multi-dimensional
            - loc
                - use when 
                    - data is high-dimensional
                    - data is heterogeneous
                        - has both discrete and continuous data
                        - features have different statistical properties (distribution, etc)
                        - features have varying relationships
                - dont use when
                    - data has large, uniform regions without much local density variation
                - feature transformation recommended
                - feature scaling recommended
            - m
                - use when
                    - data is multivariate (multi-feature)
                    - data follows Gaussian distribution
                - dont use when
                    - data is not multivariate
                    - data does not follow Gaussian distribution
                - feature transformation recommended
                - feature scaling recommended
        '''
        METHODS = {
            'dbscan': self._get_outlier_dbscan,
            'if': self._get_outlier_isolation_forest,
            'iqr': self._get_outlier_iqr,
            'loc': self._get_outlier_local_outlier_factor,
            'm': self._get_outlier_mahalanobis
        }
        if outlier_method := METHODS.get(method):
            return df.drop(outlier_method(df, params), axis=0, inplace=False)
        elif method:
            print(method, 'is not a valid outlier method')
            return None
        
    def undersample(self, 
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    method: str) -> list[np.ndarray]:
        '''
        - method
            - cc: ClusterCentroids
                - when majority class forms a discernable shape other than a big clump
            - enn: EditedNearestNeighbours
                - when majority class has many outliers
            - nm: NearMiss
                - when minority class is heavily surrounded
            - ncr: NeighbourhoodCleaningRule
                - when majority forms a giant blob and minority is embedded inside
            - rus: RandomUnderSampler
                - when majority is large with no clear structure
            - tl: TomekLinks
                - when majority and minority classes strongly overlap and decision boundery is fuzzy
        '''
        METHODS = {
            'cc': ClusterCentroids(random_state=42),
            'enn': EditedNearestNeighbours(n_jobs=-1),
            'nm': NearMiss(n_jobs=-1),
            'ncr': NeighbourhoodCleaningRule(n_jobs=-1),
            'rus': RandomUnderSampler(random_state=42),
            'tl': TomekLinks(n_jobs=-1),
        }
        if undersample_method := METHODS.get(method):
            return undersample_method.fit_resample(X_train, y_train)
        elif method:
            print(method, 'is not a valid undersampling method')
            return None
        
    def oversample(self, 
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   method: str) -> list[np.ndarray]:
        '''
        - method
            - a: ADASYN
                - when minority class is fully surrounded and tightly clumped
            - bs: BorderlineSMOTE
                - when minority class is fully surrounded and tightly clumped
            - kms: KMeansSMOTE
                - when minority class is made of multiple tiny groups
            - ros: RandomOverSampler
                - when classes are separable
            - s: SMOTE
                - when all else fails
            - se: SMOTEENN
                - when minority class is scattered everywhere (noisy)
            - st: SMOTETomek
                - when minority class are close to the decision boundary (overlapping)
        '''
        METHODS = {
            'a': ADASYN(random_state=42),
            'bs': BorderlineSMOTE(random_state=42),
            'kms': KMeansSMOTE(random_state=42),
            'ros': RandomOverSampler(random_state=42),
            's': SMOTE(random_state=42),
            'se': SMOTEENN(random_state=42),
            'st': SMOTETomek(random_state=42)
        }
        if oversample_method := METHODS.get(method):
            return oversample_method.fit_resample(X_train, y_train)
        elif method:
            print(method, 'is not a valid oversampling method')
            return None
            
    def extract_features(self, 
                         X_train: pd.DataFrame, 
                         X_test: pd.DataFrame, 
                         y_train: pd.Series, 
                         y_test: pd.Series, 
                         method: str, 
                         params) -> list[pd.DataFrame]:
        '''
        - method
            - lda
            - pca
            - umap
        '''
        METHODS = {
            'lda': self.extract_features_lda,
            'pca': self.extract_features_pca,
            'umap': self.extract_features_umap
        }
        if extraction_method := METHODS.get(method):
            return extraction_method(X_train, X_test, y_train, y_test, params, True)
        elif method:
            print(method, 'is not a valid feature extraction method')
            return None
        
    def get_split_data(self, 
                       df: pd.DataFrame, 
                       test_size: float, 
                       shuffle: bool, 
                       stratify: bool):
        X_features = df.iloc[:,:-1]
        y_labels = df.iloc[:,-1]
        return train_test_split(
            X_features
            , y_labels
            , test_size=test_size
            , random_state=42
            , shuffle=shuffle
            , stratify=y_labels if stratify else None
        )

    def _get_optimized_params(self, 
                              model: str, 
                              X_train: pd.DataFrame, 
                              y_train: pd.Series, 
                              should_randomize: bool=False,
                              trials: int=100):
        random_seed = 42 if should_randomize else None
        def objective(trial):
            match model:
                case 'knn':
                    params = {
                        'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
                        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                        'leaf_size': trial.suggest_int('leaf_size', 10, 50),
                        'p': trial.suggest_int("p", 1, 2)
                    }
                    clf = KNeighborsClassifier(**params, n_jobs=-1)
                case 'lgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
                    }
                    clf = LGBMClassifier(**params, random_state=random_seed, verbose=-1, n_jobs=-1)
                case 'lr':
                    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
                    C = trial.suggest_float('C', 1e-4, 1e2)
                    solver = 'liblinear' if penalty == 'l1' else trial.suggest_categorical('solver', ['lbfgs', 'saga', 'liblinear'])
                    max_iter = trial.suggest_int('max_iter', 100, 10000)
                    clf = LogisticRegression(penalty=penalty, C=C,  solver=solver, max_iter=max_iter, random_state=42, n_jobs=-1)
                case 'rf':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 2, 32),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 20)
                    }
                    clf = RandomForestClassifier(**params, random_state=random_seed, n_jobs=-1
                    )
                case 'svc':
                    C = trial.suggest_float('C', 0.001, 100, log=True)
                    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
                    gamma = trial.suggest_float('gamma', 0.0001, 10, log=True) if kernel != 'linear' else 'auto'
                    degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
                    max_iter = trial.suggest_int('max_iter', 100, 10000)
                    clf = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, max_iter=max_iter, probability=True, random_state=random_seed)
                case 'xgb':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                        'eval_metric': 'logloss'
                    }
                    clf = XGBClassifier(**params, random_state=random_seed, n_jobs=-1)
                case _:
                    print(model, 'is not a valid model')
                    return
                    
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
            score = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
            return score.mean()
        
        # Run Optuna optimization
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_seed))
        study.optimize(objective, n_trials=trials)

        # Show best result
        trial = study.best_trial
        print("Best trial:")
        print(f"Accuracy: {trial.value}")
        print("Best hyperparameters: ", trial.params)

        # Train and test the best model
        return trial.params

    def _get_clf_eval(self, 
                      y_test: pd.Series, 
                      pred: np.ndarray, 
                      pred_proba: np.ndarray=None):
        confusion = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test , pred)
        precision = precision_score(y_test , pred)
        recall = recall_score(y_test , pred)
        f1 = f1_score(y_test, pred)
        print('Confusion Matrix')
        print(confusion)
        if pred_proba is not None:
            roc_auc = roc_auc_score(y_test, pred_proba)
            print('Accuracy: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, F1: {3:.4f}, AUC: {4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
        else:
            print('Accuracy: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, F1: {3:.4f}'.format(accuracy, precision, recall, f1))
        
    def _get_model_train_eval(self, 
                              model,        
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame, 
                              y_train: pd.Series,
                              y_test: pd.Series,
                              should_get_pred_proba: bool=True):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if should_get_pred_proba:
            pred_proba=model.predict_proba(X_test)[:,1]
            self._get_clf_eval(y_test, pred, pred_proba)
        else:
            self._get_clf_eval(y_test, pred)
        
    def _plot_learning_curve(self, 
                             model: str, 
                             X_train: pd.DataFrame, 
                             y_train: pd.Series):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model
            , X=X_train
            , y=y_train
            , train_sizes=np.linspace(0.1, 1.0, 10) 
            , cv=10
            , n_jobs=-1
        )
        train_mean = np.mean(train_scores,  axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(12, 10))
        # training results line graph
        plt.plot(
            train_sizes # x axis
            , train_mean # y axis
            , color='blue'
            , marker='o'
            , markersize=5
            , label='Training Accuracy Per Sample'
        )

        # training data standard deviation
        plt.fill_between(
            train_sizes # x
            , train_mean+train_std # y1
            , train_mean-train_std # y2
            , alpha=0.15
            , color='blue'
        )

        # validation results line graph
        plt.plot(
            train_sizes
            , test_mean
            , color='green'
            , linestyle='--'
            , marker='s'
            , markersize=5
            , label='Validation Accuracy Per Sample'
        )

        # validation data standard deviation
        plt.fill_between(
            train_sizes # x
            , test_mean+test_std # y1
            , test_mean-test_std # y2
            , alpha=0.15
            , color='red'
        )

        plt.xlabel('Number of samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def _get_model(self, model: str, params):
        match model:
            case 'knn':
                return KNeighborsClassifier(**params, n_jobs=-1)
            case 'lgbm':
                return LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
            case 'lr':
                if params.get('penalty') == 'l1' and not params.get('solver'):
                    params['solver'] = 'liblinear'
                return LogisticRegression(**params, random_state=42, n_jobs=-1)
            case 'rf':
                return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            case 'svc':
                return SVC(**params, probability=True, random_state=42)
            case 'xgb':
                return XGBClassifier(**params, random_state=42, n_jobs=-1)
            case _:
                print(model, 'is not a valid model')
                return None

    def train_and_evaluate_model(self, 
                                 model: str, 
                                 X_train: pd.DataFrame, 
                                 X_test: pd.DataFrame, 
                                 y_train: pd.Series, 
                                 y_test: pd.Series,
                                 should_randomize: bool=False,
                                 trials: int=100):
        '''
        - model
            - knn: KNeighborsClassifier
                - use when
                    - data's decision boundary is irregular and non-linear
                    - data is small to medium sized datasets 
                    - data has localized patterns
                - examples
                    - image classification
                    - geospacial data
                    - anomaly detection
            - lgbm: LGBMClassifier
                - use when 
                    - data is large
                    - data is structured (rows and columns)
                    - data is imbalanced
                    - data is high-dimensional
                    - features have varying relationships
                = examples
                    - custumer churn prediction
                    - fraud detection
                    - risk prediction
            - lr: LogisticRegression
                - use when
                    - binary classification
                    - data is low-dimensional
                    - data's decision boundary is linear
                - examples
                    - spam detection
                    - medical diagnosis
                    - customer binary clustering
            - rf: RandomForestClassifier
                - use when
                    - data is imbalanced
                    - data is high-dimensional
                    - discrete and continuous features have varying relationships
                    - data's decision boundary is non-linear
                - examples
                    - customer clustering
                    - credit scoring
                    - feature importance anlysis
            - svc: SVC
                - use when
                    - data is high-dimensional and non-linear
                    - data has outliers that can't be removed
                - examples
                    - text classification
                    - image classification
                    - biological data
            - xgb: XGBClassifier
                - use when
                    - data is large and high-dimensional
                    - data is imbalanced
                    - data has null values
                    - features have varying relationships
                    - discrete and continuous features interact in non-linear ways (think discrete on x axis and continuous on y)
                - examples
                    - predicting load defaults
                    - fraud detection
                    - product recommendation
                    - predicting disease progression
        '''
        params = self._get_optimized_params(model, X_train, y_train, should_randomize, trials)
        clf = self._get_model(model, params)
        self._plot_learning_curve(clone(clf), X_train, y_train)
        self._get_model_train_eval(clf, X_train, X_test, y_train, y_test)
        
    def _get_base_ensemble_models(self, 
                                  models: list[str], 
                                  X_train: pd.DataFrame, 
                                  y_train: pd.Series):
        estimators = []
        for model in models:
            params = self._get_optimized_params(model, X_train, y_train)
            estimators.append((model, self._get_model(model, params)))
        return estimators
    
    def train_and_evaluate_stacking_ensemble_model(self, 
                                                   final_estimator: str, 
                                                   models: list[str], 
                                                   passthrough: bool, 
                                                   X_train: pd.DataFrame, 
                                                   X_test: pd.DataFrame, 
                                                   y_train: pd.Series, 
                                                   y_test: pd.Series):
        estimators = self._get_base_ensemble_models(models, X_train, y_train)
        if estimators:
            return
        params = self._get_optimized_params(final_estimator, X_train, y_train, True)
        final = self._get_model(final_estimator, params)
        clf = StackingClassifier(estimators=estimators, final_estimator=final, cv=5, n_jobs=-1, passthrough=passthrough)
        self._plot_learning_curve(clone(clf), X_train, y_train)
        self._get_model_train_eval(clf, X_train, X_test, y_train, y_test)
        
    def train_and_evaluate_stacking_ensemble_model_with_optimized_models(self, 
                                                                         models,
                                                                         passthrough: bool, 
                                                                         X_train: pd.DataFrame, 
                                                                         X_test: pd.DataFrame, 
                                                                         y_train: pd.Series, 
                                                                         y_test: pd.Series):
        models_list = list(models.items())
        estimators = [(model, self._get_model(model, params)) for model, params in models_list[:-1]]
        final = self._get_model(models_list[-1][0], models_list[-1][1])
        clf = StackingClassifier(estimators=estimators, final_estimator=final, cv=5, n_jobs=-1, passthrough=passthrough)
        self._plot_learning_curve(clone(clf), X_train, y_train)
        self._get_model_train_eval(clf, X_train, X_test, y_train, y_test)
            
    def train_and_evaluate_voting_ensemble_model(self, 
                                                 models: list[str], 
                                                 voting: str, 
                                                 X_train: pd.DataFrame, 
                                                 X_test: pd.DataFrame, 
                                                 y_train: pd.Series, 
                                                 y_test: pd.Series):
        estimators = self._get_base_ensemble_models(models, X_train, y_train)
        if estimators:
            return
        clf = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
        self._plot_learning_curve(clone(clf), X_train, y_train)
        self._get_model_train_eval(clf, X_train, X_test, y_train, y_test, voting=='soft')
        
    def train_and_evaluate_voting_ensemble_model_with_optimized_models(self, 
                                                                       models,
                                                                       voting: str, 
                                                                       X_train: pd.DataFrame, 
                                                                       X_test: pd.DataFrame, 
                                                                       y_train: pd.Series, 
                                                                       y_test: pd.Series):
        estimators = [(model, self._get_model(model, params)) for model, params in models.items()]
        clf = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
        self._plot_learning_curve(clone(clf), X_train, y_train)
        self._get_model_train_eval(clf, X_train, X_test, y_train, y_test, voting=='soft')
    
    def train_and_evaluate_bagging_ensemble_model(self, 
                                                  model: str, 
                                                  X_train: pd.DataFrame, 
                                                  X_test: pd.DataFrame, 
                                                  y_train: pd.Series, 
                                                  y_test: pd.Series):
        params = self._get_optimized_params(model, X_train, y_train)
        estimator = self._get_model(model, params)
        if estimator:
            return
        clf = BaggingClassifier(estimator=estimator, n_jobs=-1, random_state=42)
        self._plot_learning_curve(clone(clf), X_train, y_train)
        self._get_model_train_eval(clf, X_train, X_test, y_train, y_test)
    
    def train_and_evaluate_bagging_ensemble_model_with_optimized_model(self, 
                                                                       model: str,
                                                                       params, 
                                                                       X_train: pd.DataFrame, 
                                                                       X_test: pd.DataFrame, 
                                                                       y_train: pd.Series, 
                                                                       y_test: pd.Series):
        estimator = self._get_model(model, params)
        clf = BaggingClassifier(estimator=estimator, n_jobs=-1, random_state=42)
        self._plot_learning_curve(clone(clf), X_train, y_train)
        self._get_model_train_eval(clf, X_train, X_test, y_train, y_test)