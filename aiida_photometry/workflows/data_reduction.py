from aiida.engine import WorkChain
from aiida.plugins import DataFactory
from aiida.orm import Dict
from aiida_photometry.calcfunctions import (
    create_master_bias,
    create_master_dark,
    create_master_flat,
    calibrate_science,
)

FitsData = DataFactory("fits.data")


class SimpleCalibrationWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("raw_science", valid_type=FitsData)
        spec.input_namespace("bias_frames", valid_type=FitsData, dynamic=True)
        spec.input_namespace(
            "dark_frames", valid_type=FitsData, dynamic=True, required=False
        )
        spec.input_namespace(
            "flat_frames", valid_type=FitsData, dynamic=True, required=False
        )
        spec.input("parameters", valid_type=Dict)
        spec.outline(
            cls.create_master_bias_step,
            cls.create_master_dark_step,
            cls.create_master_flat_step
        )
        spec.output("master_bias", valid_type=FitsData)
        spec.output("master_dark", valid_type=FitsData)
        spec.output("master_flat", valid_type=FitsData)
        spec.output("calibrated_science", valid_type=FitsData)

    def create_master_bias_step(self):
        bias_nodes = self.inputs.bias_frames
        master = create_master_bias(**bias_nodes, parameters=self.inputs.parameters)
        self.ctx.master_bias = master
        self.out("master_bias", master)

    def create_master_dark_step(self):
        if "dark_frames" not in self.inputs:
            self.report("No dark frames provided — skipping master dark creation")
            return

        dark_nodes = self.inputs.dark_frames
        master = create_master_dark(
            master_bias = self.ctx.master_bias, parameters=self.inputs.parameters, **dark_nodes
        )
        self.ctx.master_dark = master
        self.out("master_dark", master)

    def create_master_flat_step(self):
        if "flat_frames" not in self.inputs:
            self.report("No flat frames provided — skipping master flat creation")
            return

        flat_nodes = self.inputs.flat_frames
        master = create_master_flat(
            master_bias = self.ctx.master_bias,
            master_dark = self.ctx.master_dark,
            parameters = self.inputs.parameters, 
            **flat_nodes
        )
        self.ctx.master_flat = master
        self.out("master_flat", master)

    def calibrate_science_step(self):
        calibrated = calibrate_science(
            self.inputs.raw_science,
            self.ctx.master_bias,
            self.ctx.master_dark,
            None,  # master flat, omitted for test
            self.inputs.parameters,
        )
        self.ctx.calibrated_science = calibrated
        self.out("calibrated_science", calibrated)
