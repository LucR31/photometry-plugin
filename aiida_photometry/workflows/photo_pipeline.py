from aiida.engine import WorkChain, submit
from aiida import orm
from aiida.plugins import WorkflowFactory


# Load child workflows via entry points
SourceDetectionWC = WorkflowFactory("centroid.detection")
AperturePhotometryWC = WorkflowFactory("aperture.photometry")


class PhotometryPipelineWorkChain(WorkChain):
    """
    End-to-end photometry pipeline:
      1. Detect sources
      2. Refine centroids
      3. Perform aperture photometry
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ------------------------------------------------------------------
        # Expose child workflow inputs
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Shared / top-level inputs
        # ------------------------------------------------------------------

        spec.input(
            "image",
            valid_type=orm.ArrayData,
            help="Science image (2D array named 'image')"
        )

        spec.input(
            "background",
            valid_type=orm.ArrayData,
            required=False,
            help="Optional background image"
        )

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------

        spec.expose_outputs(
            SourceDetectionWC,
            namespace="detection",
        )

        spec.expose_outputs(
            AperturePhotometryWC,
            namespace="aperture",
        )

        # ------------------------------------------------------------------
        # Workflow outline
        # ------------------------------------------------------------------

        spec.outline(
            cls.run_source_detection,
            cls.run_aperture_photometry,
            cls.finalize,
        )

    # ----------------------------------------------------------------------
    # Steps
    # ----------------------------------------------------------------------

    def run_source_detection(self):
        """Run the source detection workflow."""
        inputs = self.exposed_inputs(
            SourceDetectionWC,
            namespace="detection"
        )

        inputs["image"] = self.inputs.image

        if "background" in self.inputs:
            inputs["background"] = self.inputs.background

        future = self.submit(SourceDetectionWC, **inputs)
        return self.to_context(source_detection=future)

    def run_aperture_photometry(self):
        """Run aperture photometry using detected source positions."""
        inputs = self.exposed_inputs(
            AperturePhotometryWC,
            namespace="aperture"
        )

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
