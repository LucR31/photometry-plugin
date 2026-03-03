from aiida.engine import WorkChain
from aiida.plugins import DataFactory
from aiida import orm
from aiida_photometry.calcfunctions import (
    global_background_cf,
    background_2d_cf
    )

FitsData = DataFactory("fits.data")


class BackgroundWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("image", valid_type=FitsData)
        spec.input("method", valid_type=orm.Str)
        spec.input("parameters", valid_type=orm.Dict)
        spec.input_namespace("extra", dynamic=True, required=False)

        spec.outline(
            cls.run_background,
        )

        spec.output("result")

    def run_background(self):

        method = self.inputs.method.value

        if method == "global":
            result = global_background_cf(
                self.inputs.image,
                self.inputs.parameters
            )

        elif method == "background_2d":
            result = background_2d_cf(
                self.inputs.image,
                self.inputs.parameters
            )

        else:
            raise ValueError(f"Unknown background method: {method}")

        self.out("result", result)