from dataclasses import dataclass
from typing import List, Union
import matplotlib.colors as clrs
import matplotlib.pylab as plt

from core.data_input import Data
from core.base_class import BaseClass


@dataclass
class OutputValues(BaseClass):
    """Class that contains output values

    :param label_names: List of labels of different variables
    :param label_values: List of variable values
    :param color: List of colormaps corresponding to each variable

    """

    label_names: List[str]
    label_values: List[List[float]]
    color: List[List[plt.cm.ScalarMappable]]


@dataclass
class OutputDatabase(BaseClass):
    """Class that contains all necessary outputs for visualization

    :param idx: List of indexes for each point
    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates
    :param dx: Size of visualization cube in the x direction
    :param dy: Size of visualization cube in the y direction
    :param dz: Size of visualization cube in the z direction
    :param values: List of output values

    """

    idx: List[float]
    x: List[float]
    y: List[float]
    z: List[float]
    dx: List[float]
    dy: List[float]
    dz: List[float]
    values: OutputValues

    @staticmethod
    def select_default_color_for_value(value: List[float]):
        """Function that selects the default colormap for each value

        :param value: list of values included in outputs
        """
        return plt.cm.ScalarMappable(
            norm=clrs.Normalize(vmin=min(value), vmax=max(value)), cmap="gist_rainbow"
        )

    def edit_color_values(
        self, min_values: List[float], max_values: List[float], cmaps: List[str]
    ):
        """Function that edit color values

        :param min_values: list of minimum values per variable
        :param max_values: list of maximum values per variable
        :param cmaps: list of cmap names per variable
        """
        if not (len(min_values) == len(self.values.label_names)):
            raise ValueError(
                f" Inputted value min_values with length {len(min_values)} does not have the same length as the number of inputs"
            )
        if not (len(max_values) == len(self.values.label_names)):
            raise ValueError(
                f" Inputted value max_values with length {len(max_values)} does not have the same length as the number of inputs"
            )
        if not (len(cmaps) == len(self.values.label_names)):
            raise ValueError(
                f" Inputted value cmaps with length {len(cmaps)} does not have the same length as the number of inputs"
            )
        self.values.color = []
        for counter, min_value in enumerate(min_values):
            self.values.color.append(
                plt.cm.ScalarMappable(
                    norm=clrs.Normalize(vmin=min_value, vmax=max_values[counter]),
                    cmap=cmaps[counter],
                )
            )

    @classmethod
    def from_input_data(
        cls,
        labels: List[str],
        values: List[List[float]],
        x: List[str],
        y: List[str],
        z: List[str],
        dx: Union[List[float], float] = 1,
        dy: Union[List[float], float] = 1,
        dz: Union[List[float], float] = 1,
    ):
        """Function that creates class from input data"""
        default_color = [
            OutputDatabase.select_default_color_for_value(value) for value in values
        ]
        output_values = OutputValues(
            label_names=labels, label_values=values, color=default_color
        )
        idx_list = list(range(0, len(x)))

        if type(dx) == float:
            dx = [dx] * len(x)
        if type(dy) == float:
            dy = [dy] * len(x)
        if type(dz) == float:
            dz = [dz] * len(x)

        return cls(
            idx=idx_list, x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, values=output_values
        )

    @classmethod
    def from_data_structure(
        cls,
        input_dataset: List[Data],
        dx: Union[List[float], float] = 1,
        dy: Union[List[float], float] = 1,
        dz: Union[List[float], float] = 1,
    ):
        """Function that creates class from dataclass"""
        values = {}
        x, y, z = [], [], []
        for data in input_dataset:
            x = x + [data.location.x] * len(data.independent_variable.value)
            y = y + [data.location.y] * len(data.independent_variable.value)
            z = z + list(data.independent_variable.value)
            for variable in data.variables:
                if variable.label in values.keys():
                    values[variable.label] = values[variable.label] + list(
                        variable.value
                    )
                else:
                    values[variable.label] = list(variable.value)
        names = list(values.keys())
        values_as_list = list(values.values())
        default_color = [
            OutputDatabase.select_default_color_for_value(value)
            for value in values_as_list
        ]
        output_values = OutputValues(
            label_names=names, label_values=values_as_list, color=default_color
        )
        idx_list = list(range(0, len(x)))
        if type(dx) == float:
            dx = [dx] * len(x)
        if type(dy) == float:
            dy = [dy] * len(x)
        if type(dz) == float:
            dz = [dz] * len(x)
        return cls(
            idx=idx_list, x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, values=output_values
        )
