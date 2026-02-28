from aiida.engine import WorkChain
from aiida.plugins import DataFactory
from aiida.orm import List, Dict
from aiida_photometry.calcfunctions import create_master_bias, calibrate_science

FitsData = DataFactory("fits.data")


class SimpleCalibrationWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("raw_science", valid_type=FitsData)
        spec.input_namespace("bias_frames", valid_type=FitsData, dynamic=True)
        spec.input("parameters", valid_type=Dict)
        spec.outline(
            cls.create_master_bias_step,
            # cls.calibrate_science_step
        )
        spec.output("master_bias", valid_type=FitsData)
        spec.output("calibrated_science", valid_type=FitsData)

    def create_master_bias_step(self):
        bias_nodes = self.inputs.bias_frames
        master = create_master_bias(**bias_nodes, parameters=self.inputs.parameters)
        self.ctx.master_bias = master
        self.out("master_bias", master)

    def calibrate_science_step(self):
        calibrated = calibrate_science(
            self.inputs.raw_science,
            self.ctx.master_bias,
            None,  # master dark, omitted for test
            None,  # master flat, omitted for test
            self.inputs.parameters,
        )
        self.ctx.calibrated_science = calibrated
        self.out("calibrated_science", calibrated)
