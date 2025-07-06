class VariableScaler:
    """Utility that converts physical variables to dimensionless form and back.

    For now we use very simple constant scales that already give good
    conditioning for typical reservoir problems.  Later the scales can be
    computed adaptively from statistics of the model (height, capillary
    pressure, etc.).
    """

    # --- default constants ---
    DEFAULT_P_SCALE = 1e6        # 1 MPa
    DEFAULT_Q_SCALE = 1.0        # unit flow rate (m3/s) – not used yet

    def __init__(self, reservoir, fluid, p_scale: float | None = None):
        self.p_scale = p_scale if p_scale else self.DEFAULT_P_SCALE

    # ------------------------------------------------------------------
    # Pressure helpers
    # ------------------------------------------------------------------
    def p_to_hat(self, p):
        """Pa → dimensionless"""
        return p / self.p_scale

    def p_from_hat(self, p_hat):
        """dimensionless → Pa"""
        return p_hat * self.p_scale

    # Saturation is already dimensionless → identity
    @staticmethod
    def s_to_hat(s):
        return s

    @staticmethod
    def s_from_hat(s_hat):
        return s_hat 