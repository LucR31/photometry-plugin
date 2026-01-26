from aiida.engine import WorkChain
from aiida import orm

from aiida_photometry.calcfunctions import (
    circular_aperture_photometry_cf,
    circular_annulus_photometry_cf,
    elliptical_aperture_photometry_cf,
    elliptical_annulus_photometry_cf,
)

# -----------------------------------------------------------------------------
# Dispatch table: user-facing method -> calcfunction
# -----------------------------------------------------------------------------

APERTURE_DISPATCH = {
    "circular": circular_aperture_photometry_cf,
    "circular_annulus": circular_annulus_photometry_cf,
    "elliptical": elliptical_aperture_photometry_cf,
    "elliptical_annulus": elliptical_annulus_photometry_cf,
}


class AperturePhotometryWorkChain(WorkChain):
    """
    Production-grade aperture photometry workflow.

    Supports:
      - circular
      - circular annulus
      - elliptical
      - elliptical annulus
    """

    # -------------------------------------------------------------------------
    # Definition
    # -------------------------------------------------------------------------

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # --- Inputs ---
        spec.input(
            "image",
            valid_type=orm.ArrayData,
            help="Science image with array named 'image'",
        )

        spec.input(
            "background",
            valid_type=orm.ArrayData,
            required=False,
            help="Optional background image to subtract",
        )

        spec.input(
            "positions",
            valid_type=orm.ArrayData,
            help="Source positions with arrays 'x' and 'y'",
        )

        spec.input(
            "aperture",
            valid_type=orm.Dict,
            default=lambda: orm.Dict(dict={"r": 5.0}),
            help="Aperture geometry parameters",
        )

        spec.input(
            "photometry_options",
            valid_type=orm.Dict,
            default=lambda: orm.Dict(dict={}),
            help="Options forwarded to photutils.aperture_photometry",
        )

        spec.input(
            "method",
            valid_type=orm.Str,
            default=lambda: orm.Str("circular"),
            help="Photometry method: circular | circular_annulus | elliptical | elliptical_annulus",
        )

        # --- Outputs ---
        spec.output(
            "photometry",
            valid_type=orm.Dict,
            help="Aperture photometry results",
        )

        # --- Outline ---
        spec.outline(
            cls.validate_inputs,
            cls.prepare_image,
            cls.run_photometry,
            cls.finalize,
        )

        # --- Exit codes ---
        spec.exit_code(300, "ERROR_INVALID_IMAGE", "Input image must be 2D")
        spec.exit_code(301, "ERROR_INVALID_POSITIONS", "Positions must contain 'x' and 'y'")
        spec.exit_code(302, "ERROR_INVALID_APERTURE", "Invalid aperture parameters")
        spec.exit_code(310, "ERROR_UNKNOWN_METHOD", "Unknown photometry method")

    # -------------------------------------------------------------------------
    # Steps
    # -------------------------------------------------------------------------

    def validate_inputs(self):
        # Validate image
        image = self.inputs.image.get_array("image")
        if image.ndim != 2:
            return self.exit_codes.ERROR_INVALID_IMAGE

        # Validate positions
        positions = self.inputs.positions
        for key in ("x", "y"):
            if key not in positions.get_arraynames():
                return self.exit_codes.ERROR_INVALID_POSITIONS

        # Validate method
        method = self.inputs.method.value
        if method not in APERTURE_DISPATCH:
            return self.exit_codes.ERROR_UNKNOWN_METHOD

        # Minimal aperture validation
        aperture = self.inputs.aperture.get_dict()
        if not aperture:
            return self.exit_codes.ERROR_INVALID_APERTURE

    def prepare_image(self):
        image = self.inputs.image.get_array("image")

        if "background" in self.inputs:
            background = self.inputs.background.get_array("image")
            image = image - background

        img = orm.ArrayData()
        img.set_array("image", image)

        self.ctx.image = img

    def run_photometry(self):
        method = self.inputs.method.value
        photometry_cf = APERTURE_DISPATCH[method]

        self.ctx.photometry = photometry_cf(
            self.ctx.image,
            self.inputs.positions,
            self.inputs.aperture,
            self.inputs.photometry_options,
        )

    def finalize(self):
        self.out("photometry", self.ctx.photometry)

