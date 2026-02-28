from aiida.engine import WorkChain
from aiida import orm
from aiida.orm import ArrayData, Dict
from aiida_photometry.calcfunctions import centroid_sources_cf, detect_sources_cf


class SourceDetectionWorkChain(WorkChain):
    """
    Detect sources in an image using a two-step process:
      1. detect sources with DAOStarFinder
      2. refine positions using centroid_sources
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # --- Inputs ---
        spec.input(
            "image", valid_type=ArrayData, help="Science image (2D array named 'image')"
        )

        spec.input(
            "background",
            valid_type=ArrayData,
            required=False,
            help="Optional background image to subtract",
        )

        spec.input(
            "detection_params",
            valid_type=Dict,
            default=lambda: Dict(dict={"threshold": 3.0, "fwhm": 3.0}),
            help="Parameters for source detection (DAOStarFinder)",
        )

        spec.input(
            "refine_params",
            valid_type=Dict,
            default=lambda: Dict(dict={}),
            help="Parameters for centroid_sources refinement",
        )

        # --- Outputs ---
        spec.output("sources", valid_type=ArrayData, help="Refined source positions")

        # --- Outline ---
        spec.outline(
            cls.prepare_image,
            cls.detect_sources,
            cls.refine_sources,
            cls.finalize,
        )

        # Exit codes
        spec.exit_code(300, "ERROR_INVALID_IMAGE", "Input image must be 2D")
        spec.exit_code(301, "ERROR_NO_SOURCES", "No sources were detected")

    def prepare_image(self):
        """Subtract background if provided and store in context."""
        image = self.inputs.image.get_array("image")
        if image.ndim != 2:
            return self.exit_codes.ERROR_INVALID_IMAGE

        if "background" in self.inputs:
            bck = self.inputs.background.get_array("image")
            image = image - bck

        img = ArrayData()
        img.set_array("image", image)
        self.ctx.image = img

    def detect_sources(self):
        """Run DAOStarFinder calcfunction to detect sources."""
        self.ctx.positions = detect_sources_cf(
            image=self.ctx.image, options=self.inputs.detection_params
        )

        # Check if sources were found
        x = self.ctx.positions.get_array("x")
        y = self.ctx.positions.get_array("y")
        if len(x) == 0:
            return self.exit_codes.ERROR_NO_SOURCES

    def refine_sources(self):
        """Refine the detected source positions using centroid_sources calcfunction."""
        self.ctx.refined_positions = centroid_sources_cf(
            image=self.ctx.image,
            positions=self.ctx.positions,
            options=self.inputs.refine_params,
        )

    def finalize(self):
        """Expose refined sources as output."""
        self.out("sources", self.ctx.refined_positions)
