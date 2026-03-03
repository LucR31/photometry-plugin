from aiida.engine import WorkChain
from aiida import orm
from aiida.plugins import WorkflowFactory,DataFactory

FitsData = DataFactory("fits.data")
# Load child workflows via entry points
SourceDetectionWC = WorkflowFactory("centroid.detection")
AperturePhotometryWC = WorkflowFactory("aperture.photometry")
BackgroundWC = WorkflowFactory("background.estimation")


class PhotometryPipelineWorkChain(WorkChain):
    """
    End-to-end photometry pipeline
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            BackgroundWC,
            namespace="background",
            exclude=("image"),
        )

        spec.expose_inputs(
            SourceDetectionWC,
            namespace="detection",
            exclude=("image", "background"),
        )

        spec.expose_inputs(
            AperturePhotometryWC,
            namespace="aperture",
            exclude=("image", "positions"),
        )

        spec.input(
            "image",
            valid_type=FitsData,
            help="Science image",
        )

        spec.input(
            "background",
            valid_type=orm.ArrayData,
            required=False,
            help="Optional precomputed background image.",
        )

        spec.expose_outputs(
            BackgroundWC,
            namespace="background",
        )

        spec.expose_outputs(
            SourceDetectionWC,
            namespace="detection",
        )

        spec.expose_outputs(
            AperturePhotometryWC,
            namespace="aperture",
        )

        spec.outline(
            cls.run_background,
            cls.run_source_detection,
            cls.run_aperture_photometry,
            cls.finalize,
        )

    def run_background(self):
        if "background" in self.inputs:
            pass
        else:
            self.ctx.bkg = self.submit(
                BackgroundWC,
                image=self.inputs.image,
                method=self.inputs.bkg_method,
                parameters=self.inputs.bkg_parameters,
            )   

    def run_source_detection(self):
        """Run the source detection workflow."""
        inputs = self.exposed_inputs(SourceDetectionWC, namespace="detection")

        inputs["image"] = self.inputs.image

        future = self.submit(SourceDetectionWC, **inputs)
        return self.to_context(source_detection=future)

    def run_aperture_photometry(self):
        """Run aperture photometry using detected source positions."""
        inputs = self.exposed_inputs(AperturePhotometryWC, namespace="aperture")

        inputs["image"] = self.inputs.image
        inputs["positions"] = self.ctx.source_detection.outputs.sources

        future = self.submit(AperturePhotometryWC, **inputs)
        return self.to_context(photometry=future)

    def finalize(self):
        self.out_many(
            self.exposed_outputs(
                self.ctx.photometry,
                AperturePhotometryWC,
                namespace="aperture",
            )
        )
