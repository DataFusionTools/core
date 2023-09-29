from core.data_input import Geometry, Data, Variable
from core.data_output import OutputDatabase
import pytest
import numpy as np


def create_list_data():
    input = {
        "variable_1": np.array(range(1, 10, 1)),
        "variable_2": np.array(range(1, 10, 1)),
        "variable_3": np.array(range(1, 15, 1)),
        "variable_4": np.array(range(1, 15, 1)),
    }
    variable_1 = Variable(label="variable_1", value=input["variable_1"])
    variable_2 = Variable(label="variable_2", value=input["variable_2"])
    variable_3 = Variable(label="variable_1", value=input["variable_3"])
    variable_4 = Variable(label="variable_2", value=input["variable_4"])
    independent_variable_1 = Variable(
        label="independent_variable_1", value=np.array(range(1, 10, 1))
    )
    independent_variable_2 = Variable(
        label="independent_variable_2", value=np.array(range(1, 15, 1))
    )
    location = Geometry(x=1, y=2)
    data_1 = Data(
        location=location,
        variables=[variable_1, variable_2],
        independent_variable=independent_variable_1,
    )
    data_2 = Data(
        location=location,
        variables=[variable_3, variable_4],
        independent_variable=independent_variable_2,
    )
    return [data_1, data_2]


class TestDataOutput:
    @pytest.mark.unittest
    def test_from_input_data(self):
        # set up input
        labels = ["one", "two"]
        values = [[1, 1, 1, 1], [2, 2, 2, 2]]
        x = [1, 1, 1, 1]
        y = [1, 2, 3, 4]
        z = [-1, -2, -3, -4]
        dx = 0.4
        dy = 0.4
        dz = 0.4
        # set up model
        model = OutputDatabase.from_input_data(
            labels, values, x, y, z, dx=dx, dy=dy, dz=dz
        )
        # check results
        assert type(model) == OutputDatabase
        assert model.x == x
        assert model.y == y
        assert model.z == z
        assert model.values.label_names == labels
        assert model.values.label_values == values
        assert len(model.values.color) == 2

    @pytest.mark.unittest
    def test_from_data_structure(self):
        # set up input
        list_data = create_list_data()
        # set up model
        model = OutputDatabase.from_data_structure(
            input_dataset=list_data, dx=0.5, dy=0.1, dz=0.2
        )
        # check results
        assert type(model) == OutputDatabase
        assert len(model.x) == 23
        assert len(model.y) == 23
        assert len(model.z) == 23
        assert len(model.values.color) == 2

    @pytest.mark.unittest
    def test_edit_color_values(self):
        # set up input
        list_data = create_list_data()
        min_values = [0, -1]
        max_values = [20, 30]
        cmaps = ["binary", "gist_yarg"]
        # set up model
        model = OutputDatabase.from_data_structure(
            input_dataset=list_data, dx=0.5, dy=0.1, dz=0.2
        )
        # check results
        assert type(model) == OutputDatabase
        assert len(model.x) == 23
        assert len(model.y) == 23
        assert len(model.z) == 23
        assert len(model.values.color) == 2
        # run test
        model.edit_color_values(min_values, max_values, cmaps)
        assert model.values.color[0].cmap.name == "binary"
        assert model.values.color[1].cmap.name == "gist_yarg"
        assert model.values.color[0].norm.vmax == 20
        assert model.values.color[1].norm.vmax == 30
        assert model.values.color[0].norm.vmin == 0
        assert model.values.color[1].norm.vmin == -1
