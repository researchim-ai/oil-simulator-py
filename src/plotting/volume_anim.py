import os
from typing import Optional

import numpy as np

# Third-party heavy dependencies; we import lazily so that normal CPU runs without
# PyVista/Vtk installed do not crash.  If the user calls `save_volume_gif`, we will
# raise a clear error message if the necessary packages are missing.

def _lazy_import_pyvista():
    try:
        import pyvista as pv  # noqa: F401
        return pv
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyVista is required for 3-D GIF export. Install with `pip install pyvista vtk` "
        ) from e


def save_volume_gif(
    vtk_file: str,
    output_gif: str,
    scalar: str = "Water_Saturation",
    n_frames: int = 36,
    cmap: str = "viridis",
    opacity: str | float | tuple = "sigmoid",
    clim: Optional[tuple[float, float]] = None,
):
    """Create a rotating GIF from a VTK structured-grid file.

    Parameters
    ----------
    vtk_file : str
        Path to the *.vts file produced by ``output.vtk_writer.save_to_vtk``.
    output_gif : str
        Destination GIF filename (``.gif`` extension will be added if missing).
    scalar : str, default "Water_Saturation"
        Which cell-data array to visualise.
    n_frames : int, default 36
        Number of camera positions (frames) for the 360° turn-table animation.
    cmap : str, default "viridis"
        Colormap for scalar rendering.
    opacity : Union[str, float, tuple]
        Opacity mapping accepted by ``pyvista.add_volume``.
    clim : Optional[tuple], default None
        Fixed color limits; if ``None`` PyVista chooses automatically.
    """

    pv = _lazy_import_pyvista()

    if not os.path.isfile(vtk_file):
        raise FileNotFoundError(f"VTK file '{vtk_file}' not found.")

    if not output_gif.lower().endswith(".gif"):
        output_gif += ".gif"

    grid = pv.read(vtk_file)

    if scalar not in grid.cell_data.keys():
        raise KeyError(
            f"Scalar '{scalar}' not found in VTK cell-data. Available: {list(grid.cell_data.keys())}"
        )

    # Build an off-screen plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_volume(
        grid,
        scalars=scalar,
        cmap=cmap,
        opacity=opacity,
        clim=clim,
        scalar_bar_args={"title": scalar, "vertical": True},
    )

    # Camera initial positioning: look from +X, elevation 30°
    plotter.camera_position = "xy"
    plotter.camera.elevation = 30  # type: ignore[attr-defined]

    # Initialise GIF writer
    plotter.open_gif(output_gif)

    # Rotate 360°
    for _ in range(n_frames):
        plotter.camera.azimuth(360 / n_frames)  # type: ignore[attr-defined]
        plotter.render()
        plotter.write_frame()

    plotter.close()
    print(f"3-D GIF saved to {output_gif}") 