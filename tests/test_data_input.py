from core.data_input import Geometry, Data, Variable

import pytest
import numpy as np


class TestDataFusionTools:
    @pytest.mark.unittest
    def test_no_independent_variable(self):
        input = {
            "variable_1": np.array(range(1, 10, 1)),
            "variable_2": np.array(range(1, 10, 1)),
        }
        variable_1 = Variable(label="variable_1", value=input["variable_1"])
        variable_2 = Variable(label="variable_2", value=input["variable_2"])
        location = Geometry(x=1, y=2)
        data = Data(
            location=location,
            variables=[variable_1, variable_2],
        )
        assert len(data.variables) == 2
        assert data.independent_variable is None

    @pytest.mark.unittest
    def test_data_input(self):
        # create dictionary of variables
        input = {
            "variable_1": np.array(range(1, 10, 1)),
            "variable_2": np.array(range(1, 10, 1)),
            "time": np.array(range(1, 10, 1)),
        }
        variable_1 = Variable(label="variable_1", value=input["variable_1"])
        variable_2 = Variable(label="variable_2", value=input["variable_2"])
        location = Geometry(x=1, y=2)
        data = Data(
            location=location,
            variables=[variable_1, variable_2],
            independent_variable=Variable(label="time", value=input["time"]),
        )
        assert len(data.variables) == 2
        # test get
        variable_from_getter = data.get_variable("variable_1")
        assert variable_from_getter == variable_1

    @pytest.mark.unittest
    def test_data_input_error_raised(self):
        # create dictionary of variables
        input = {
            "variable_1": np.array(range(1, 10, 1)),
            "variable_2": np.array(range(1, 10, 1)),
            "time": np.array(range(1, 10, 2)),
        }
        variable_1 = Variable(label="variable_1", value=input["variable_1"])
        variable_2 = Variable(label="variable_2", value=input["variable_2"])
        location = Geometry(x=1, y=2)
        with pytest.raises(ValueError) as excinfo:
            Data(
                location=location,
                variables=[variable_1, variable_2],
                independent_variable=Variable(label="time", value=input["time"]),
            )
            assert (
                "Length of variable variable_1 is not the same as that of the independent variable"
                in str(excinfo.value)
            )
