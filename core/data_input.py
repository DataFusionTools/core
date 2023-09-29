from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np


@dataclass
class Geometry:
    x: float
    y: float
    z: Optional[float] = None
    label: Optional[float] = None


@dataclass
class Variable:
    label: str
    value: np.array


@dataclass
class Data:
    """
    Class that contains values of one location

    :param location: The location the data
    :param variables: Variables that are contain in the data
    :param independent_variable: Independent variable that is connected to all the data
    """

    location: Geometry
    independent_variable: Union[Variable, None] = None
    variables: List[Variable] = []

    def get_variable(self, name: str):
        """
        Function that returns variable based on its name
        """
        for variable in self.variables:
            if variable.label == name:
                return variable
        return None
    
    def update_variable(self, name:str, value:np.ndarray):
        """
        Function that updates variable
        """
        index = None
        for counter, variable in enumerate(self.variables):
            if name == variable.label:
                index = counter
                continue
        if index is None:
            raise ValueError(f"Name provided {name} is defined in this class instance.")
        self.variables[index].value = value

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, values: List[Variable]):
        if self.independent_variable is not None:
            for variable in values:
                if len(variable.value) != len(self.independent_variable.value):
                    raise ValueError(
                        f"Length of variable {variable.label} is not the same as that of the independent variable."
                    )
        self._variables = values
