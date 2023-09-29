from typing import DefaultDict, List, Tuple, Union
from .data_input import Data, Geometry, Variable
import numpy as np
import random
from enum import Enum
import math
from scipy.interpolate import interp1d


class AggregateMethod(Enum):
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    MEAN = "mean"


class CreateInputsML:
    """Utils class that creates features and targets for machine learning class"""

    def __init__(self) -> None:
        self._features = {}
        self._targets = {}
        self._input_dump = []
        self._features_train = {}
        self._targets_train = {}
        self._features_test = {}
        self._targets_test = {}
        self._features_validation = {}
        self._targets_validation = {}

    def split_train_test_data(
        self, train_percentage=0.7, validation_percentage_on_test=0.5
    ):
        """
        Method that splits training and testing data.

        :param train_percentage: Percentage of train samples
        :param validation_percentage_on_test: Percentage of validation samples taken from test samples
        """
        if self._features is {}:
            raise ValueError("No features were added for training")
        if self._targets is {}:
            raise ValueError("No features were added for training")
        # get random value of dictionary
        value = random.choice(list(self._features.values()))
        number_of_items = len(value)
        train_samples = int((train_percentage) * number_of_items)
        train_indexes = random.sample(range(0, number_of_items - 1), train_samples)
        test_and_validation_indexes = list(
            set(train_indexes) ^ set(range(0, number_of_items - 1))
        )
        validation_indexes = random.sample(
            test_and_validation_indexes,
            int(validation_percentage_on_test * len(test_and_validation_indexes)),
        )
        test_indexes = list(set(validation_indexes) ^ set(test_and_validation_indexes))

        # initialize empty dicts
        self._features_train = dict.fromkeys(self._features, [])
        self._features_test = dict.fromkeys(self._features, [])
        self._features_validation = dict.fromkeys(self._features, [])
        self._targets_train = dict.fromkeys(self._targets, [])
        self._targets_test = dict.fromkeys(self._targets, [])
        self._targets_validation = dict.fromkeys(self._targets, [])
        for item, value in self._features.items():
            self._features_train[item] = [value[index] for index in train_indexes]
            self._features_test[item] = [value[index] for index in test_indexes]
            self._features_validation[item] = [
                value[index] for index in validation_indexes
            ]
        for item, value in self._targets.items():
            self._targets_train[item] = [value[index] for index in train_indexes]
            self._targets_test[item] = [value[index] for index in test_indexes]
            self._targets_validation[item] = [
                value[index] for index in validation_indexes
            ]
        return

    def append_features(
        self,
        input: Data,
        variable_names: List[str],
        use_independent_variable: bool = True,
        use_location_as_input: Tuple[bool, bool, bool] = (False, False, False),
    ):
        """
        Function that appends features in private class properties.

        :param input: data class to be added as feature
        :param variable_names: list of strings that represents the features that should be extracted from the dataclass
        :param use_independent_variable: If true then the independent variable of the data class is used as a feature
        :param use_location_as_input: If true the location attribute of the data class is used as a feature

        """
        # create equivalent Data object to append to features
        equivalent_variables = []
        for variable in variable_names:
            equivalent_variables.append(
                Variable(label=variable, value=input.get_variable(variable).value)
            )
        input_with_only_requested_variables = Data(
            input.location, input.independent_variable, equivalent_variables
        )
        # append to features
        self._input_dump.append(
            {
                "input": input_with_only_requested_variables,
                "use_independent_variable": use_independent_variable,
                "use_location_as_input": use_location_as_input,
            }
        )

    def get_feature_names(self):
        """
        All the names of features are returned.
        """
        if self._features == {}:
            raise ValueError("There are no features assigned yet.")
        return list(self._features.keys())

    def get_all_features(self, flatten: bool):
        """
        Function that returns all features in a form of a numpy.array

        :param flatten: the returned array is flattened per feature
        """
        return self.get_features(self._features, flatten)

    def get_features_train(self, flatten: bool):
        """
        Function that returns features that are used for training in a form of a numpy.array

        :param flatten: the returned array is flattened per feature
        """
        return self.get_features(self._features_train, flatten)

    def get_features_test(self, flatten: bool):
        """
        Function that returns features that are used for testing in a form of a numpy.array

        :param flatten: the returned array is flattened per feature
        """
        return self.get_features(self._features_test, flatten)

    def get_features_validation(self, flatten: bool):
        """
        Function that returns features that are used for validation in a form of a numpy.array

        :param flatten: the returned array is flattened per feature
        """
        return self.get_features(self._features_validation, flatten)

    def get_features(self, features: dict, flatten: bool):
        """
        Function that returns features from dict

        :param features: a dictionary of features to be combined
        :param flatten: the returned array is flattened per feature
        """
        if flatten:
            # get first key of the dictionary
            first_key = list(features.keys())[0]
            number_of_items = len(features[first_key])
            result_features = []
            for item_index in range(number_of_items):
                record = [value[item_index] for key, value in features.items()]
                record = np.array(record).flatten()
                result_features.append(record)
            return result_features
        else:
            return np.array(
                [np.concatenate(value) for key, value in features.items()]
            ).T

    def get_all_targets(self, flatten: bool):
        return self.get_targets(self._targets, flatten)

    def get_targets_train(self, flatten: bool):
        return self.get_targets(self._targets_train, flatten)

    def get_targets_test(self, flatten: bool):
        return self.get_targets(self._targets_test, flatten)

    def get_targets_validation(self, flatten: bool):
        return self.get_targets(self._targets_validation, flatten)

    def get_targets(self, targets: dict, flatten: bool):
        if flatten:
            # get first key of the dictionary
            first_key = list(targets.keys())[0]
            number_of_items = len(targets[first_key])
            targets = []
            for item_index in range(number_of_items):
                record = [value[item_index] for key, value in targets.items()]
                record = np.array(record).flatten()
                targets.append(record)
            return targets
        else:
            return np.array([np.concatenate(value) for key, value in targets.items()]).T

    def add_features(
        self,
        input: Data,
        variable_names: List[str],
        use_independent_variable: bool = True,
        use_location_as_input: Tuple[bool, bool, bool] = (False, False, False),
    ):
        """
        Method that creates features based on the inputs given.
        """
        self.append_features(
            input, variable_names, use_independent_variable, use_location_as_input
        )
        for variable in variable_names:
            if variable not in list(self._features.keys()):
                self._features[variable] = [input.get_variable(variable).value]
            else:
                self._features[variable].append(input.get_variable(variable).value)
        if use_independent_variable:
            if input.independent_variable.label not in list(self._features.keys()):
                self._features[input.independent_variable.label] = [
                    input.independent_variable.value
                ]
            else:
                self._features[input.independent_variable.label].append(
                    input.independent_variable.value
                )
        if use_location_as_input[0]:
            self._features = self.add_location_to_features(
                "location_x", input.location.x, len(input.independent_variable.value)
            )
        if use_location_as_input[1]:
            self._features = self.add_location_to_features(
                "location_y", input.location.y, len(input.independent_variable.value)
            )
        if use_location_as_input[2]:
            self._features = self.add_location_to_features(
                "location_z", input.location.z, len(input.independent_variable.value)
            )

    def add_location_to_features(self, name_dict, value, length, dictionary=None):
        if dictionary is None:
            dictionary = self._features
        if name_dict not in list(dictionary.keys()):
            dictionary[name_dict] = [np.array([value] * length)]
        else:
            dictionary[name_dict].append(np.array([value] * length))
        return dictionary

    def add_targets(self, input: Data, variable_names: List[str]):
        """
        Static method that creates features based on the inputs given.
        """
        for variable in variable_names:
            if variable not in list(self._targets.keys()):
                self._targets[variable] = [input.get_variable(variable).value]
            else:
                self._targets[variable].append(input.get_variable(variable).value)

    def get_k_closest_features(
        self, point_compare: Geometry, combined_data: List[Data], number_of_points: int
    ):
        # compute distances in 3d space
        if number_of_points > len(combined_data):
            raise ValueError(
                f"The number of points requested ({number_of_points}) is smaller than the number of points provided ({len(combined_data)})."
            )
        distances, indexes = [], []
        for counter, combined_feature in enumerate(combined_data):
            distances.append(
                math.sqrt(
                    (point_compare.x - combined_feature.location.x) ** 2
                    + (point_compare.y - combined_feature.location.y) ** 2
                    + (point_compare.z - combined_feature.location.z) ** 2
                )
            )
            indexes.append(counter)
        closer_points_index = [
            index for dist, index in sorted(zip(distances, indexes))
        ][:number_of_points]
        closer_extracted_features = [
            combined_data[index] for index in closer_points_index
        ]
        return closer_extracted_features

    def aggregate_extracted_features(
        self,
        agrregate_method: AggregateMethod,
        aggregate_variable: str,
        closer_extracted_features: List[Data],
    ):
        if AggregateMethod.SUM == agrregate_method:
            aggregated_value = 0
            for closer_extracted_feature in closer_extracted_features:
                values_sum = sum(
                    closer_extracted_feature.get_variable(aggregate_variable).value
                )
                aggregated_value += values_sum
        elif AggregateMethod.MEAN == agrregate_method:
            aggregated_value = 0
            for closer_extracted_feature in closer_extracted_features:
                values_sum = sum(
                    closer_extracted_feature.get_variable(aggregate_variable).value
                )
                aggregated_value += values_sum
            aggregated_value = aggregated_value / len(closer_extracted_features)
        elif AggregateMethod.MIN == agrregate_method:
            aggregated_value = min(
                closer_extracted_features[0].get_variable(aggregate_variable).value
            )
            for closer_extracted_feature in closer_extracted_features:
                values_min = min(
                    closer_extracted_feature.get_variable(aggregate_variable).value
                )
                aggregated_value = min(values_min, aggregated_value)
        elif AggregateMethod.MAX == agrregate_method:
            aggregated_value = max(
                closer_extracted_features[0].get_variable(aggregate_variable).value
            )
            for closer_extracted_feature in closer_extracted_features:
                values_max = max(
                    closer_extracted_feature.get_variable(aggregate_variable).value
                )
                aggregated_value = max(values_max, aggregated_value)
        return aggregated_value

    def interpolate_on_independent_variable(
        self,
        closer_extracted_features: List[Data],
        main_features: Data,
        aggregate_method: AggregateMethod,
        aggregate_variable: str,
        bounds_error: bool = False,
        fill_value: Union[str, np.array, List] = 'extrapolate'
    ):
        new_values = []
        for closer_extracted_feature in closer_extracted_features:
            # interpolate on independent variable
            interpolator = interp1d(
                closer_extracted_feature.independent_variable.value,
                closer_extracted_feature.get_variable(aggregate_variable).value,
                bounds_error=bounds_error,
                fill_value=fill_value,
            )
            interpolated_results = interpolator(
                main_features.independent_variable.value
            )
            new_values.append(interpolated_results)
        if AggregateMethod.SUM == aggregate_method:
            aggregated_list = np.array([sum(i) for i in zip(*new_values)])
        elif AggregateMethod.MAX == aggregate_method:
            aggregated_list = np.array([max(i) for i in zip(*new_values)])
        elif AggregateMethod.MIN == aggregate_method:
            aggregated_list = np.array([min(i) for i in zip(*new_values)])
        elif AggregateMethod.MEAN == aggregate_method:
            aggregated_list = np.array([sum(i) / len(i) for i in zip(*new_values)])
        main_features.variables.append(
            Variable(
                label=aggregate_variable,
                value=aggregated_list,
            )
        )
        return main_features

    def find_closer_points(
        self,
        input_data: List[Data],
        combined_data: List[Data],
        aggregate_method: AggregateMethod,
        aggregate_variable: str,
        number_of_points: int = 1,
        interpolate_on_independent_variable: bool = False,
        bounds_error: bool = False,
        fill_value: Union[str, np.array, List] = 'extrapolate'
    ):
        """
        Function that finds the closest point and aggregates results and returns those aggregated results
        """
        # loop through all input data
        for main_features in input_data:
            closer_extracted_features = self.get_k_closest_features(
                main_features.location, combined_data, number_of_points
            )
            # aggregate method
            if interpolate_on_independent_variable:
                self.interpolate_on_independent_variable(
                    closer_extracted_features,
                    main_features,
                    aggregate_method,
                    aggregate_variable,
                    bounds_error=bounds_error,
                    fill_value=fill_value
                )
            else:
                aggregated_value = self.aggregate_extracted_features(
                    aggregate_method, aggregate_variable, closer_extracted_features
                )
                # add the aggregated value as variable in initial feature
                length_variable = len(main_features.variables[0].value)
                main_features.variables.append(
                    Variable(
                        label=aggregate_variable,
                        value=np.array([aggregated_value] * length_variable),
                    )
                )
        return input_data
