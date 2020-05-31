import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
# def f1_score(real_labels, predicted_labels):
#     """
#     Information on F1 score - https://en.wikipedia.org/wiki/F1_score
#     :param real_labels: List[int]
#     :param predicted_labels: List[int]
#     :return: float
#     """
#     assert len(real_labels) == len(predicted_labels)
#     # raise NotImplementedError
#     true_pos = 0
#     false_pos = 0
#     true_neg = 0
#     false_neg = 0
#     for i in range(len(predicted_labels)):
#         if real_labels[i] == predicted_labels[i] == 1:
#             true_pos += 1
#         if real_labels[i] == predicted_labels[i] == 0:
#             true_neg += 1
#         if real_labels[i] == 0 and predicted_labels[i] != real_labels[i]:
#             false_pos += 1
#         if real_labels[i] == 1 and predicted_labels[i] != real_labels[i]:
#             false_neg += 1
#     print("TP, TN, FP, FN :", true_pos, true_neg, false_pos, false_neg)
#     recall = true_pos/float((true_pos+false_neg))
#     print("recall:", recall)
#     precision = true_pos/float((true_pos + false_pos))
#     print("precision:", precision)
#     f1 = 2*(precision*recall)/float((precision+recall))
#     return f1


def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    assert len(real_labels) == len(predicted_labels)
    for a, b in zip(real_labels, predicted_labels):
        if a == b == 1:
            true_pos += 1
        elif a == b == 0:
            true_neg += 1
        elif a == 1 and b == 0:
            false_neg += 1
        else:
            false_pos += 1
    recall = (true_pos / float((true_pos + false_neg))) if (true_pos + false_neg) else 0
    precision = (true_pos / float((true_pos + false_pos))) if (true_pos + false_pos) else 0
    f1 = (2 * (precision * recall) / float((precision + recall))) if (precision + recall) else 0
    return f1
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum_val = 0
        for i in range(len(point1)):
            sum_val += (abs(point1[i] - point2[i])) ** 3
        return float(sum_val ** (1 / 3))

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum_val = 0
        for i in range(len(point1)):
            sum_val += (point1[i] - point2[i]) ** 2
        return float(sum_val ** (1 / 2))

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum_val = 0
        for i in range(len(point1)):
            sum_val += point1[i] * point2[i]
        return float(sum_val)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        nr = 0
        for i in range(len(point1)):
            nr += point1[i] * point2[i]
        dr_1 = sum(map(lambda i: i * i, point1))
        dr_2 = sum(map(lambda j: j * j, point2))
        cos_sim = nr / float((dr_1 ** (1 / 2)) * (dr_2 ** (1 / 2)))
        return 1 - cos_sim

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        # x_minus_y = list(set(point1) - set(point2))
        # inn_prod = Distances.inner_product_distance(x_minus_y, x_minus_y)
        # dis = -np.exp(-(1/2)*inn_prod)
        # return dis
        euc_dis = Distances.euclidean_distance(point1, point2)
        dis = -np.exp(-(1 / 2) * (euc_dis ** 2))
        return dis


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        self.best_f1 = 0

        k_values = range(1, 30, 2)
        best_f1_for_a_k = []
        best_dist_for_a_k = []
        best_model_for_a_k = []

        for k in k_values:
            f1_k = []
            dis_k = []
            model_k = []
            # for each K, find f1 wrt all distances >> choose the distance with max f1
            # so for every K, you will max f1, best distance that led to that f1
            # to decide what K, sort by f1 first(Desc) >> if more than one K with same F1 >> use best distance of
            # those Ks to choose one >> if that still has more than 1 K >> choose the smallest K
            for dist in distance_funcs.items():
                model = KNN(k, dist[1])
                model.train(x_train, y_train)
                # print("model trained with k: {} and dist : {}".format(k, dist[0]))
                # print("x_val :", x_val)
                # print(model.get_k_neighbors(x_val[0]))
                pred_labels = model.predict(x_val)
                # print("predicted validation labels are:", pred_labels)
                # print("real validation labels, y_val:", y_val)
                # print("length of input x_val:", len(x_val))
                # print("length of predicted labels:", len(pred_labels))
                new_f1 = f1_score(y_val, pred_labels)
                # print("f1 score:", new_f1)
                f1_k.append(new_f1)
                dis_k.append(dist[0])
                model_k.append(model)
            max_f1 = max(f1_k)
            best_f1_for_a_k.append(max_f1)
            best_dist_for_a_k.append(dis_k[f1_k.index(max_f1)])
            best_model_for_a_k.append(model_k[f1_k.index(max_f1)])
        #     print("all F1 score for k : {} is {}".format(k, f1_k))
        #     print("all distance tried for k : {} is {}".format(k, dis_k))
        #     print("best f1 for k : {} for all ks so far is {}".format(k, best_f1_for_a_k))
        #     print("best dists for k : {} for all ks so far is {}".format(k, best_dist_for_a_k))
        #
        # print("best_f1_for_a_k:", best_f1_for_a_k)
        # print("length of f1s:", len(best_f1_for_a_k))
        # print("best_dist_for_a_k:", best_dist_for_a_k)
        # print("length of dists:", len(best_dist_for_a_k))
        max_f1 = max(best_f1_for_a_k)
        self.best_f1 = max_f1
        max_f1_indices = [i for i in range(len(best_f1_for_a_k)) if best_f1_for_a_k[i] == max_f1]
        # print("max f1 occurs at indices:", max_f1_indices)
        if len(max_f1_indices) == 1:
            # best_f1 = best_f1_for_a_k[max_f1_indices[0]]
            self.best_k = k_values[max_f1_indices[0]]
            self.best_distance_function = best_dist_for_a_k[max_f1_indices[0]]
            self.best_model = best_model_for_a_k[max_f1_indices[0]]
            return
        else:
            # break tie with distance priority
            # get distance names for indices and length of unique names if tie Ks have same best distance then use
            # lower k
            best_dist_names = [best_dist_for_a_k[i] for i in max_f1_indices]
            unique_dis_count = len(list(set(best_dist_names)))
            if unique_dis_count == 1:
                # choose the lower K as best K
                ks = [k_values[i] for i in max_f1_indices]
                k_min = min(ks)
                self.best_k = k_min
                self.best_distance_function = list(set(best_dist_names))[0]
                self.best_model = best_model_for_a_k[ks.index(k_min)]
                return
            else:
                # decide based on the priority of the distance measures
                if 'euclidean' in best_dist_names:
                    indices_with_same_dis = []
                    for i in max_f1_indices:
                        if best_dist_for_a_k[i] == 'euclidean':
                            indices_with_same_dis.append(i)
                    self.best_k = k_values[min(indices_with_same_dis)]
                    self.best_distance_function = 'euclidean'
                    self.best_model = best_model_for_a_k[min(indices_with_same_dis)]
                    return
                elif 'minkowski' in best_dist_names:
                    indices_with_same_dis = []
                    for i in max_f1_indices:
                        if best_dist_for_a_k[i] == 'minkowski':
                            indices_with_same_dis.append(i)
                    self.best_k = k_values[min(indices_with_same_dis)]
                    self.best_distance_function = 'minkowski'
                    self.best_model = best_model_for_a_k[min(indices_with_same_dis)]
                    return
                elif 'gaussian' in best_dist_names:
                    indices_with_same_dis = []
                    for i in max_f1_indices:
                        if best_dist_for_a_k[i] == 'gaussian':
                            indices_with_same_dis.append(i)
                    self.best_k = k_values[min(indices_with_same_dis)]
                    self.best_distance_function = 'gaussian'
                    self.best_model = best_model_for_a_k[min(indices_with_same_dis)]
                    return
                elif 'inner_prod' in best_dist_names:
                    indices_with_same_dis = []
                    for i in max_f1_indices:
                        if best_dist_for_a_k[i] == 'inner_prod':
                            indices_with_same_dis.append(i)
                    self.best_k = k_values[min(indices_with_same_dis)]
                    self.best_distance_function = 'inner_prod'
                    self.best_model = best_model_for_a_k[min(indices_with_same_dis)]
                    return
                elif 'cosine_dist' in best_dist_names:
                    indices_with_same_dis = []
                    for i in max_f1_indices:
                        if best_dist_for_a_k[i] == 'cosine_dist':
                            indices_with_same_dis.append(i)
                    self.best_k = k_values[min(indices_with_same_dis)]
                    self.best_distance_function = 'cosine_dist'
                    self.best_model = best_model_for_a_k[min(indices_with_same_dis)]
                    return
                else:
                    print("no decision reached :( ")
                    return
        # self.best_model = KNN(self.best_k, self.best_distance_function)
        raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

        best_k_for_scalers = []
        best_dist_for_scalers = []
        best_f1_for_scalers = []
        best_model_for_scalers = []
        scalers = []

        for scaler_type in scaling_classes.items():
            scalers.append(scaler_type[0])
            if scaler_type[0] == 'normalize':
                scaler1 = NormalizationScaler()
                x_train_scaled_norm = scaler1(x_train)
                x_val_scaled_norm = scaler1(x_val)
                self.tuning_without_scaling(distance_funcs, x_train_scaled_norm, y_train, x_val_scaled_norm, y_val)
                best_k_for_scalers.append(self.best_k)
                best_dist_for_scalers.append(self.best_distance_function)
                best_f1_for_scalers.append(self.best_f1)
                best_model_for_scalers.append(self.best_model)
            else:
                minmax_scaler = MinMaxScaler()
                x_train_scaled_minmax = minmax_scaler(x_train)
                x_val_scaled_minmax = minmax_scaler(x_val)
                self.tuning_without_scaling(distance_funcs, x_train_scaled_minmax, y_train, x_val_scaled_minmax, y_val)
                best_k_for_scalers.append(self.best_k)
                best_dist_for_scalers.append(self.best_distance_function)
                best_f1_for_scalers.append(self.best_f1)
                best_model_for_scalers.append(self.best_model)

        max_f1_scalers = max(best_f1_for_scalers)
        max_f1_scalers_indices = [i for i in range(len(best_f1_for_scalers)) if best_f1_for_scalers[i] == max_f1_scalers]
        if len(max_f1_scalers_indices) == 1:
            self.best_k = best_k_for_scalers[max_f1_scalers_indices[0]]
            self.best_distance_function = best_dist_for_scalers[max_f1_scalers_indices[0]]
            self.best_scaler = scalers[max_f1_scalers_indices[0]]
            self.best_model = best_model_for_scalers[max_f1_scalers_indices[0]]
        else:
            minmax_scaler_index = scalers.index('min_max_scale')
            self.best_k = best_k_for_scalers[minmax_scaler_index]
            self.best_distance_function = best_dist_for_scalers[minmax_scaler_index]
            self.best_scaler = scalers[minmax_scaler_index]
            self.best_model = best_model_for_scalers[minmax_scaler_index]
        return
        raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized_feat = []
        for feat in features:
            inn_prod_dist = Distances.inner_product_distance(feat, feat)
            inn_prod_dist_sqrt = inn_prod_dist ** (1 / 2)
            normalized_feat.append([i / float(inn_prod_dist_sqrt) if inn_prod_dist_sqrt else 0 for i in feat])
        return normalized_feat
        raise NotImplementedError


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.call_count = 0
        self.maxm_feat = None
        self.minm_feat = None
        pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        # find min and max to training data

        if self.call_count == 0:
            max_feat = []
            min_feat = []
            n_features = len(features[0])
            for i in range(n_features):
                max_i = max(features, key=lambda x: x[i])[i]
                max_feat.append(max_i)
                min_i = min(features, key=lambda x: x[i])[i]
                min_feat.append(min_i)
            self.maxm_feat = max_feat
            self.minm_feat = min_feat
            self.call_count += 1
            normalized_features = []
            for feat in features:
                new_list = [(feat[i] - self.minm_feat[i]) / (self.maxm_feat[i] - self.minm_feat[i]) if (
                        self.maxm_feat[i] - self.minm_feat[i]) else 0 for i in range(len(feat))]
                normalized_features.append(new_list)
            return normalized_features
        else:
            normalized_features = []
            for feat in features:
                new_list = [(feat[i] - self.minm_feat[i]) / (self.maxm_feat[i] - self.minm_feat[i]) if (
                            self.maxm_feat[i] - self.minm_feat[i]) else 0 for i in range(len(feat))]
                normalized_features.append(new_list)
            return normalized_features
        raise NotImplementedError

# if __name__ == "__main__":
#     a = [10, 0, 7, 6, 33, 11]
#     b = [19, 11, 21, 30, 11, 40]
#     print(Distances.minkowski_distance(a, b))
