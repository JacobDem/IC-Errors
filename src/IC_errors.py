"""
@author: Jacob Demuynck
"""
import random
import math

import pandas as pd
import numpy as np
from openTSNE import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
import pyflann

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class IC_errors:
    """ IC-Errors class used for outlier detection \
        and label error correction in numerical datasets

    Attributes:
        df : pandas.DataFrame
            data set to be used processed by IC-Errors
        target_col : str
            name of the column in the dataset that is the target for classification
        batch_col : str, default None
            optional name of the column in the dataset that depicts the batch a sample belongs to
    """

    def __init__(self, df, target_col, batch_col=None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Invalid type for dataframe, type should be " + str(pd.DataFrame))

        if not isinstance(target_col, str):
            raise TypeError(
                "Invalid type for target column name, type should be " + str(str))

        if batch_col is not None and not isinstance(batch_col, str):
            raise TypeError(
                    "Invalid type for batch column name, type should be " + str(str))

        self.df = df
        self.target_col = target_col
        try:
            self.df[target_col] = self.df[target_col].astype(str)
        except Exception as exc:
            raise TypeError(
                "Values in target column of type " + 
                str(self.df[target_col].dtypes) + 
                " can't be cast to string."
            ) from exc

        self.classes = df[target_col].unique()
        self.classes.sort()
        self.y = np.array(list(map(lambda x: list(self.classes).index(x),
                                   list(self.df[self.target_col]))))

        if batch_col is not None:
            self.X = df.drop(columns=[target_col, batch_col])
            self.batch_col = batch_col
        else:
            self.X = df.drop(columns=[target_col])
            self.batch_col = None

        (self.n_samples, self.n_attributes) = self.X.shape

        # Data to compute        
        self.feature_bagging_result = np.array([])
        self.random_forest_result = np.array([])
        self.neural_training_values_result = np.array([])
        self.neural_training_switches_result = np.array([])
        self.cleanlab_result = np.array([])
        self.df_tsne = pd.DataFrame()
        self.nearest_neighbors_result = None


    def construct_tsne(self, perplexity=30, overwrite=False):
        """ Constructs and saves a two-dimensional t-SNE embedding of the dataframe

        Args:
            perplexity : int, default 30
                The perplexity parameter to be used for the t-SNE embedding
            overwrite : bool, default False
                Allow an existing t-SNE embedding to be overwritten
        
        Returns:
            Dataframe representing the two-dimensional t-SNE embedding
        """

        if not overwrite and not self.df_tsne.empty:
            raise PermissionError('A t-SNE embedding of the dataframe already exists. \
                                  Use overwrite=True to overwrite this embedding.')

        self.df_tsne = pd.DataFrame(TSNE(n_components=2, n_jobs=-1, 
                                         perplexity=perplexity).fit(self.X.to_numpy()))
        self.df_tsne.columns = ['x', 'y']
        self.df_tsne[self.target_col] = np.array(self.df[self.target_col])
        if self.batch_col is not None:
            self.df_tsne[self.batch_col] = self.df[self.batch_col]

        return self.df_tsne

    def set_tsne(self, df_tsne):
        """Set the t-SNE embedding of the dataframe
        
        Args:
            df_tsne : pandas.DataFrame
                A previously calculated two-dimensional t-SNE embedding of the data set. \
                This dataframe should contain two columns representing the x and y coordinates, \
                and the same target classification and optional batch columns, \
                as defined in the class constructor.
        """

        if self.batch_col is not None:
            if df_tsne.shape[1] != 4:
                raise ValueError('Number of columns of TSNE dataframe is incorrect. \
                                 Expected 4 columns, but received ' +
                                str(df_tsne.shape[1]))
            self.df_tsne = pd.DataFrame()
            self.df_tsne[self.target_col] = df_tsne[self.target_col]
            try:
                self.df_tsne[self.target_col] = self.df_tsne[self.target_col].astype(str)
            except Exception as exc:
                raise TypeError(
                    "Values in target column of type " + str(self.df_tsne[self.target_col].dtypes) + " can't be cast to string."
                ) from exc
        
            self.df_tsne[self.batch_col] = df_tsne[self.batch_col]
            self.df_tsne['x'] = df_tsne.drop(
                columns=[self.target_col, self.batch_col]).iloc[:, 0]
            self.df_tsne['y'] = df_tsne.drop(
                columns=[self.target_col, self.batch_col]).iloc[:, 1]

        else:
            if df_tsne.shape[1] != 3:
                raise ValueError('Number of columns of TSNE dataframe is incorrect. \
                                 Expected 3 columns, but received ' +
                                str(df_tsne.shape[1]))
            self.df_tsne = pd.DataFrame()
            self.df_tsne[self.target_col] = df_tsne[self.target_col]
            try:
                self.df_tsne[self.target_col] = self.df_tsne[self.target_col].astype(str)
            except Exception as exc:
                raise TypeError(
                    "Values in target column of type " + str(self.df_tsne[self.target_col].dtypes) + " can't be cast to string."
                ) from exc

            self.df_tsne['x'] = df_tsne.drop(columns=[self.target_col]).iloc[:, 0]
            self.df_tsne['y'] = df_tsne.drop(columns=[self.target_col]).iloc[:, 1]


    def outlier_feature_bagging(self, R=20, threshold=0.5, outlier_metric='lof', n_neighbors=20, apply_threshold=True):
        """ Outlier detection using the feature bagging algorithm 
            as described in https://dl.acm.org/doi/pdf/10.1145/1081870.1081891.
            The algorithm is applied to each target class separately.

        Args:
            R : int
                The number of feature bagging rounds in the algorithm
            outlier_metric : str, default 'lof'
                The outlier detection metric to be used, one of ['lof', 'if', 'mixed']
            n_neighbors : int, default 20
                The number of neighbors to be used in the LOF algorithm
            threshold : float in (0,1], default 0.5
                The percentage of rounds a sample has to be classified as outlier 
                to be an outlier in the final score.
            apply_threshold : bool, default True
                Boolean value indicating if the final anomaly score should be converted
                to True/False values according to the outlier threshold. If False, the
                original anomaly scores are returned instead.
        
        Returns: A boolean numpy array of length n_samples, with True for every outlier 
                 and False for every non-outlier
        """

        if outlier_metric not in ['lof', 'if', 'mixed']:
            raise ValueError('Invalid value for outlier metric. Received "'
                             + str(outlier_metric)
                             + '", expected value from ["lof", "if", "mixed"]')

        df_reset = self.df.reset_index(drop=True)

        # Split dataframe per target
        df_per_target = [x for _, x in df_reset.groupby(self.target_col)]

        # Randomly calculate attributes to be used in each round of outlier detection
        active_attributes_list = []
        for t in range(R):
            N_t = random.randrange(math.floor(
                self.n_attributes / 2), self.n_attributes - 1)
            active_attributes = random.sample(range(self.n_attributes), N_t)
            active_attributes_list.append(active_attributes)

        for i,df_target in enumerate(df_per_target):

            anomaly_score = np.zeros(shape=(R, df_target.shape[0]))

            # Outlier detection
            for t in range(R):
                active_attributes = active_attributes_list[t]
                X = df_target.iloc[:, active_attributes]

                if X.shape[0] < 2:
                    anomaly_score[t] = [1]
                else:
                    # Local Outlier Factor
                    if outlier_metric == 'lof':
                        clf = LocalOutlierFactor(n_neighbors=n_neighbors)
                    # Isolation Forest
                    elif outlier_metric == 'if':
                        clf = IsolationForest()
                    # Mix between LOF and IF
                    else:
                        if t % 2:
                            clf = IsolationForest()
                        else:
                            clf = LocalOutlierFactor(n_neighbors=n_neighbors)

                    # Calculate anomaly score vector
                    lof_pred = clf.fit_predict(X)
                    lof_pred = (lof_pred - 1) / (-2)
                    anomaly_score[t] = lof_pred

            # Anomaly score combination
            outlier_vector = np.sum(anomaly_score, axis=0)

            df_target['outlier_score'] = outlier_vector
            df_per_target[i] = df_target

        df_with_outlier = pd.concat(df_per_target, axis=0).sort_index()

        if not apply_threshold:
            self.feature_bagging_result = np.array(df_with_outlier['outlier_score'] / R)
        else:
            self.feature_bagging_result = np.array((
                df_with_outlier['outlier_score'] / R) >= threshold)
        
        return self.feature_bagging_result

    def set_feature_bagging(self, feature_bagging_result):
        """ Set the outlier feature bagging result

        Args:
            feature_bagging_result : numpy.array
                A previously calculated result of the outlier_feature_bagging function
        """

        if feature_bagging_result.shape != (self.n_samples,):
            raise ValueError('Incorrect shape of feature_bagging_result. Expected shape ('
                             + str(self.n_samples)
                             + ',) but got ' + str(feature_bagging_result.shape))
        self.feature_bagging_result = feature_bagging_result


    def outlier_random_forest(self, threshold=0.5, n_estimators=100,
                              max_depth=None, apply_threshold=True):
        """ Outlier detection using Random Forest classification

        Args:
            threshold : float in (0, 1], default 0.5
                Value indicating the minimum prediction probability of a sample's true 
                label to not be considered an outlier
            n_estimators : int, default 100
                n_estimators parameter used in the scikit-learn implementation 
                of RandomForestClassifier
            max_depth : int, default None
                max_depth parameter used in the scikit-learn implementation 
                of RandomForestClassifier
            apply_threshold : bool, default True
                Boolean value indicating if the final prediction probabilities should be converted
                to True/False values according to the outlier threshold. If False,
                every data point's prediction probability of its true label is returned instead.

        Returns: A boolean numpy array of length n_samples, with True for every outlier 
                 and False for every non-outlier.
        """

        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth).fit(self.X, self.df[self.target_col])
        print(clf.classes_)
        strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)

        predictions = cross_val_predict(
           clf,
           self.X,
           self.df[self.target_col],
           cv=strat_k_fold,
           method="predict_proba"
        )

        outliers = []

        for i,y in enumerate(self.y):
            if not apply_threshold:
                outliers.append(predictions[i][y])
            else:
                if predictions[i][y] <= 1 - threshold:
                    outliers.append(True)
                else:
                    outliers.append(False)
        self.random_forest_result = np.array(outliers)

        return self.random_forest_result

    def set_random_forest(self, random_forest_result):
        """ Set the outlier random forest result
        
        Args:
            random_forest_result : numpy.array
                A previously calculated result of the outlier_random_forest function
        """

        if random_forest_result.shape != (self.n_samples,):
            raise ValueError('Incorrect shape of random_forest_result. Expected shape ('
                             + str(self.n_samples)
                             + ',) but got ' + str(random_forest_result.shape))
        self.random_forest_result = random_forest_result


    def get_neural_model(self, LR):
        """ Returns a simple neural model used in the neural training outlier algorithm

        Args:
            LR : float
                Learning rate of the model
        """

        model = Sequential()

        model.add(Dense(2048, activation='relu',
                        input_shape=(self.n_attributes,), kernel_initializer='he_uniform'))
        model.add(Dense(2048, activation='relu', kernel_initializer='he_uniform'))

        model.add(Dense(len(self.classes), activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate=LR),
                        metrics=['accuracy'])
        return model


    def outlier_neural_training(self, epochs=None, values_threshold=0.5, switches_threshold=0.5, apply_threshold=True):
        """ Outlier detection using training information of a simple neural network on the dataset.
            This function impelments both the neural training values algorithm and the
            neural training switches algorithm

        Args:
            epochs : int, default None
                The amount of epochs to use during the training. \
                If default None is used, this will be equal to \
                the number of unique classes in the data set.
            values_threshold : float in (0,1], default 0.5
                An outlier threshold indicating a percentage of the number of unique classes, 
                used in the neural training values algorithm. If during training, the number of different classes 
                a data point has been predicted as is equal to or greater than this percentage, 
                it will be considered an outlier.
            switches_threshold : float in (0,1], default 0.5
                An outlier threshold indicating a percentage of the number of training epochs, 
                used in the neural training switches algorithm. If during training, the amount of epochs where
                the prediction of a data point switches from the last epoch to the next exceeds this percentage,
                the data point will be considered an outlier.
            apply_threshold : bool, default True
                Boolean value indicating if the algorithms' results should be converted
                to True/False values according to the outlier threshold. If False,
                an array of every data point's number of prediction values 
                and an array of every data point's number of prediction switches will be returned instead.
        
        Returns: A tuple of two boolean numpy arrays of length n_samples, with True for every outlier 
                 and False for every non-outlier. The first array is the "values' algorithm result,
                 the second array is the "switches" algorithm result.

        """
        if epochs is None:
            epochs = max(len(self.classes),10)

        y_vector = keras.utils.to_categorical(self.y)
        X = self.X

        model_predictions = np.zeros(shape=(epochs,self.n_samples))

        class PredictionCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                y_pred = self.model.predict(X).argmax(axis=-1)
                model_predictions[epoch] = y_pred

        def next_power_of_2(x):  
            return 1 if x == 0 else 2**(x - 1).bit_length()

        batch_size = next_power_of_2(round(self.n_samples/16))
        learning_rate = batch_size/512000

        # Create your model
        model_1 = self.get_neural_model(LR = learning_rate)

        model_1.fit(self.X, y_vector,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    callbacks=[PredictionCallback()]
                    )

        model_predictions = np.array(model_predictions)

        switches = np.zeros(self.n_samples)
        unique_predictions = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            unique_predictions[i] = len(np.unique(model_predictions[:,i]))

            for n in range(0,len(model_predictions)-1):
                if model_predictions[n][i] != model_predictions[n+1][i]:
                    switches[i] += 1

        self.neural_training_values_result = unique_predictions/len(self.classes) >= values_threshold if apply_threshold else unique_predictions/len(self.classes)
        self.neural_training_switches_result = switches/(epochs-1) >= switches_threshold if apply_threshold else switches/(epochs-1)
        
        return (self.neural_training_values_result, self.neural_training_switches_result)

    def set_neural_training_values(self, neural_training_values_result):
        """ Set the neural training values result
        
        Args:
            neural_training_values_result : numpy.array
                A previously calculated result of the "values" algorithm of the outlier_neural_training function
        """

        if neural_training_values_result.shape != (self.n_samples,):
            raise ValueError('Incorrect shape of neural_training_values_result. Expected shape ('
                             + str(self.n_samples)
                             + ',) but got ' + str(neural_training_values_result.shape))
        self.neural_training_values_result = neural_training_values_result

    def set_neural_training_switches(self, neural_training_switches_result):
        """ Set the neural training switches result
        
        Args:
            neural_training_values_result : numpy.array
                A previously calculated result of the "switches" algorithm of the outlier_neural_training function
        """

        if neural_training_switches_result.shape != (self.n_samples,):
            raise ValueError('Incorrect shape of neural_training_values_result. Expected shape ('
                             + str(self.n_samples)
                             + ',) but got ' + str(neural_training_switches_result.shape))
        self.neural_training_switches_result = neural_training_switches_result


    def outlier_cleanlab(self):
        """ Outlier detection using the find_label_issues function of CleanLab (https://cleanlab.ai/)
        
        Returns: A boolean numpy array of length n_samples, with True for every outlier 
                 and False for every non-outlier
        """
        model_random_forest = RandomForestClassifier()
        strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
        pred_probs = cross_val_predict(
           model_random_forest,
           self.X,
           self.y,
           cv=strat_k_fold,
           method="predict_proba",
        )

        label_issues_info = find_label_issues(pred_probs=pred_probs,labels=self.y, 
                                              return_indices_ranked_by="self_confidence")
        self.cleanlab_result = np.array([True if x in label_issues_info 
                                         else False for x in range(self.n_samples)])
        
        return self.cleanlab_result

    def set_cleanlab(self, cleanlab_result):
        """ Set the cleanlab result
        
        Args:
            cleanlab_result : numpy.array
                A previously calculated result of the outlier_cleanlab function
        """

        if cleanlab_result.shape != (self.n_samples,):
            raise ValueError('Incorrect shape of cleanlab_result. Expected shape ('
                             + str(self.n_samples)
                             + ',) but got ' + str(cleanlab_result.shape))
        self.cleanlab_result = cleanlab_result


    def outlier_ensemble(self, threshold, include_cleanlab=False, granularity=0.01, excluded_methods=None):
        """ outlier detection using a combination of the different outlier functions in this class

        Args:
            threshold : float in (0,1)
                A threshold indicating a percentage of the total number of samples in the data set.
                This threshold is used to choose the thresholds of the individual outlier functions,
                in such a way that every outlier function will marl this percentage of data points as outliers.
            include_cleanlab : bool, default False
                If true, the outliers found by the cleanlab algorithm will be added to the final list of outliers.
            excluded_methods : list, default []
                Names of outlier methods to exclude from the ensemble algorithm. \
                Possible values are ['feature bagging', 'random forest', 'neural values', 'neural switches']

        Returns: A boolean numpy array of length n_samples, with True for every outlier 
                 and False for every non-outlier. This array is calculated by taking the intersection of
                 the results of every separate outlier function.

        """
        if excluded_methods is not None:
            if 'feature bagging' not in excluded_methods:
                if self.feature_bagging_result.size == 0:
                    self.outlier_feature_bagging(apply_threshold=False)
            if 'random forest' not in excluded_methods:
                if self.random_forest_result.size == 0:
                    self.outlier_random_forest(apply_threshold=False)
            if 'neural values' not in excluded_methods or 'neural switches' not in excluded_methods:
                if self.neural_training_values_result.size == 0 or self.neural_training_switches_result.size == 0:
                    self.outlier_neural_training(apply_threshold=False)
            if include_cleanlab:
                if self.cleanlab_result.size == 0 and include_cleanlab:
                    self.outlier_cleanlab()
                    
            outlier_selection = ['feature bagging' not in excluded_methods,
                                'random forest' not in excluded_methods,
                                'neural values' not in excluded_methods,
                                'neural switches' not in excluded_methods]
        else:
            outlier_selection = [True]*4

        outliers_array = np.array([self.feature_bagging_result,
                            1 - self.random_forest_result,
                            self.neural_training_values_result,
                            self.neural_training_switches_result])[outlier_selection]
        outliers_thresholds = np.arange(granularity,1+granularity,granularity)
        for i,outliers in enumerate(outliers_array):
            best_t = 0
            best_percentage = 0

            for t in outliers_thresholds:
                percentage = len(np.where(outliers >= t)[0])/len(outliers)
                if percentage > 0:
                    if (best_percentage == 0) or (abs(threshold - percentage) < abs(threshold - best_percentage)):
                        best_percentage = percentage
                        best_t = t

            if best_percentage == 0:
                outliers_array[i] = [True] * self.n_samples
            else:
                outliers_array[i] = outliers >= best_t

        outliers = np.logical_and.reduce(outliers_array)

        if include_cleanlab:
            outliers = np.logical_or(outliers, self.cleanlab_result)
        
        return outliers
    

    def nearest_neighbors(self, n_neighbors=10, branching=128, iterations=127, checks=64):
        """ Calculates the nearest neighbors of every data point in high-dimensional space,
            using the FLANN (Fast Library for Approximate Nearest Neighbors) implementation.

        Args:
            n_neighbors : int, default 10
                The number of nearest neighbors to calculate of every data point.
            branching = int, default 128
                The branching factor as used in the FLANN.nn function
            iterations = int, default 128
                The number of iterations as used in the FLANN.nn function
            checks = int, default 128
                The number of checks as used in the FLANN.nn function
        
        Returns:
            A dictionary, in which every key is the numerical index of a data point
            and every value is the list of numerical indices of its nearest neighbors
        """
        flann = pyflann.FLANN()
        result, _ = flann.nn(
            self.X.to_numpy().astype('float64'), self.X.to_numpy().astype('float64'), n_neighbors+1, algorithm="kmeans", branching=branching, 
            iterations=iterations, checks=checks)
        
        result = dict(np.array([[x[0], x[1:]] for x in result], dtype=object))
        self.nearest_neighbors_result = result
        
        return result
    
    def set_nearest_neighbors(self, nearest_neighbors):
        """ Set the nearest neighbors result
        
        Args:
            nearest_neighbors : dictionary
                A previously calculated result of the nearest_neighbors function
        """

        self.nearest_neighbors_result = nearest_neighbors


    def get_neighbor_labels_dict(self, index, include_batch=False):
        """ Calculate the labels of the nearest neighbors of a data point

        Args:
            index : int in [0 , n_samples]
                The numerical index of the data point to calculate the neighbor labels from.
            include_batch : bool, default False
                If True, only the nearest neighbors from the same batch of the data point 
                will be taken into account.
        
        Returns:
            A dictionary in which every key is a class label and every value is the number
            of nearest neighbors that have this label.
        """

        original_labels = np.array(self.df[self.target_col].copy())
        label = original_labels[index]

        if include_batch:
            original_batches = np.array(self.df[self.batch_col].copy())
            batch = original_batches[index]

        neighbors = self.nearest_neighbors_result[index]
        neighbor_labels = []

        if include_batch:
            for n in neighbors:
                if original_batches[n] == batch:
                    neighbor_labels.append(original_labels[n])
        else:
            neighbor_labels = [original_labels[x] for x in neighbors]

        neighbor_labels_dict = {}
        for neighbor_label in set(neighbor_labels):
            neighbor_labels_dict[neighbor_label] = neighbor_labels.count(neighbor_label)

        return (label, neighbor_labels_dict)


    def outlier_label_correction(self, outliers, threshold=0.8, include_batch=True):
        """ Label correction for a set of outliers, using their nearest neighbors.

        Args:
            outliers : numpy.array
                Boolean numpy array of length n_samples, indicating True for every outlier
                and False for every non-outlier
            threshold : float in (0.5, 1], default 0.8
                Percentage of nearest neighbors an outlier that need to the same class label
                that is not the outlier's class label, for the outlier to be considered a label error
            include_batch : bool, default = True
                If True, the threshold will only take into account nearest neighbors within the
                same batch as the outlier

        Returns:
            A tuple containing:
                - A boolean numpy array of length n_samples, containing the fully corrected class
                  labels of every data point.
                - A numpy array containing the numeric indices of the data point whose labels
                  were corrected.
        """

        if self.batch_col is None:
            include_batch = False

        corrected_labels = np.array(self.df[self.target_col].copy())
        correction_positions = []

        if not self.nearest_neighbors_result:
            self.nearest_neighbors()

        n_neighbors = len(self.nearest_neighbors_result[0])

        for i in np.where(outliers)[0]:
            (label, neighbor_labels_dict) = self.get_neighbor_labels_dict(i, include_batch)
            
            if neighbor_labels_dict:
                if label not in neighbor_labels_dict.keys():
                    max_label = max(neighbor_labels_dict, key=neighbor_labels_dict.get)
                    if neighbor_labels_dict[max_label] >= n_neighbors*threshold:
                        corrected_labels[i] = max_label
                        correction_positions.append(i)

        return (corrected_labels, np.array(correction_positions))