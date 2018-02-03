import numpy as np
from scipy.stats import spearmanr
import tensorflow as tf
import sklearn as skl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted
from ml_project.models.utils import parse_hooks
from ml_project.models.base import BaseTF
from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY)
from tensorflow.python.estimator.export.export_output import PredictOutput


class ExampleTF(BaseTF):
    """docstring for ExampleTF"""
    def __init__(
        self,
        input_fn_config={"shuffle": True},
        config={},
        params={}):  # noqa: E129

        super(ExampleTF, self).__init__(input_fn_config, config, params)

    def model_fn(self, features, labels, mode, params, config):

        input_tensor = tf.cast(features["X"], tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=-1)

        conv1 = tf.layers.conv1d(
                        input_tensor,
                        filters=16,
                        kernel_size=8,
                        strides=1,
                        activation=tf.nn.relu)

        max_pool1 = tf.layers.max_pooling1d(
                        conv1,
                        pool_size=4,
                        strides=1)

        flat = tf.layers.flatten(max_pool1)

        dense_layer_1 = tf.layers.dense(
                        flat,
                        units=512,
                        activation=tf.nn.relu)

        dense_layer_1_norm = tf.norm(dense_layer_1)

        logits = tf.layers.dense(
                        dense_layer_1,
                        units=4,
                        activation=None)

        probabs = tf.nn.softmax(logits)

        predictions = tf.argmax(probabs, axis=1)
        # ================================================================
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "predictions": predictions,
                    "probabs": probabs},
                export_outputs={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    PredictOutput({"predictions": predictions}),
                    "probabs":
                    PredictOutput({"probabs": probabs})})
        # ================================================================
        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, depth=4)

        loss = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels,
                        logits=logits)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.MomentumOptimizer(
                        learning_rate=params["learning_rate"],
                        momentum=0.9,
                        use_nesterov=True)

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        # ================================================================
        if "hooks" in params:
            training_hooks = parse_hooks(
                params["hooks"],
                locals(),
                self.save_path)
        else:
            training_hooks = []
        # ================================================================
        if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):  # noqa: E129
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=training_hooks)

    def score(self, X, y):
        y_pred = self.predict(X)
        return skl.metrics.f1_score(y, y_pred, average="micro")


class ExtraTreesClassifier(ExtraTreesClassifier):
    def predict(self, X, y=None):
        return super().predict(X).astype(int)


class SVC(SVC):
    def predict(self, X, y=None):
        return super().predict(X).astype(int)


class AdaBoostClassifier(AdaBoostClassifier):
    def predict(self, X, y=None):
        return super().predict(X).astype(int)


class GradientBoostingClassifier(GradientBoostingClassifier):
    def predict(self, X, y=None):
        return super().predict(X).astype(int)


class BaseClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, classifier):
        self.classifier = None

    def fit(self, X, y):
        """transform y"""
        print("Original shape of y is ", y.shape)
        n_samples, n_classes = y.shape
        y_new = []
        for i in range(n_samples):
            class_vec = np.zeros(n_classes, dtype=object)
            for j in range(1, n_classes+1):
                class_vec[np.argsort(y[i])[j-1]] = str(j)
            y_new.append(" ".join(class_vec))
        y_new = np.array(y_new)
        print("The shape of y_new is ", y_new.shape)

        self.classifier.fit(X, y_new)
        return self

    def predict_proba(self, X):
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.classifier.predict(X)[i].split())
        y_pred = np.array(y_pred)
        return y_pred.astype(int)

    def score(self, X, y, sample_weight=None):
        n_samples, _ = X.shape
        scores = []
        for i in range(n_samples):
            score, _ = spearmanr(self.predict_proba(X)[i], y[i])
            scores.append(score)
        return np.mean(scores)


class RandomForestPredictor(BaseClassifier):
    def __init__(self, classifier=None, n_estimators=10, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features='auto', oob_score=False, class_weight=None):
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.class_weight = class_weight

    def fit(self, X, y):
        self.classifier = RandomForestClassifier(
                                    n_estimators=self.n_estimators,
                                    max_depth=self.max_depth,
                                    min_samples_leaf=self.min_samples_leaf,
                                    min_samples_split=self.min_samples_split,
                                    max_features=self.max_features,
                                    oob_score=self.oob_score,
                                    class_weight=self.class_weight
                                    )
        super(RandomForestPredictor, self).fit(X, y)
        return self


class SVMPredictor(BaseClassifier):
    def __init__(self, classifier=None, C=1.0, kernel='rbf', gamma='auto'):
        self.classifier = classifier
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y):
        self.classifier = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)
        super(SVMPredictor, self).fit(X, y)
        return self


class ExtraTreesPredictor(BaseClassifier):
    def __init__(self, classifier=None, n_estimators=10, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features='auto', bootstrap=False,
                 oob_score=False, class_weight=None
                 ):
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.class_weight = class_weight

    def fit(self, X, y):
        self.classifier = ExtraTreesClassifier(
                                    n_estimators=self.n_estimators,
                                    max_depth=self.max_depth,
                                    min_samples_leaf=self.min_samples_leaf,
                                    min_samples_split=self.min_samples_split,
                                    max_features=self.max_features,
                                    bootstrap=self.bootstrap,
                                    oob_score=self.oob_score,
                                    class_weight=self.class_weight
                                    )
        super(ExtraTreesPredictor, self).fit(X, y)
        return self


class AdaBoostPredictor(BaseClassifier):
        def __init__(self, classifier=None, base_estimator=None,
                     n_estimators=50, learning_rate=1.0
                     ):
            self.classifier = classifier
            self.base_estimator = base_estimator
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate

        def fit(self, X, y):
            self.classifier = AdaBoostClassifier(
                                        base_estimator=self.base_estimator,
                                        n_estimators=self.n_estimators,
                                        learning_rate=self.learning_rate
                                        )
            super(AdaBoostPredictor, self).fit(X, y)
            return self


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""

    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))

    def score(self, X, y, sample_weight=None):
        n_samples, _ = X.shape
        scores = []
        for i in range(n_samples):
            score, _ = spearmanr(self.predict_proba(X)[i], y[i])
            scores.append(score)
        return np.mean(scores)
