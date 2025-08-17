from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod
from typing import Union, Tuple
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class BasePlot(ABC):
    def __init__(
        self,
        output_path: Union[str, Path],
        dpi: int = 300,
        thumbnail_dpi: int = 100,
    ):
        self.output_path = Path(output_path).expanduser()
        self.dpi = dpi
        self.thumbnail_dpi = thumbnail_dpi

    @abstractmethod
    def _draw(self) -> Figure:
        """
        Implement plot drawing logic here using matplotlib and return the
        figure.
        """
        pass

    def save(self) -> Path:
        """Saves full plot and thumbnail, returns full image path."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = self._draw()
        fig.savefig(self.output_path, dpi=self.dpi)
        plt.close(fig)

        self._make_thumbnail()
        return self.output_path

    def _make_thumbnail(self) -> None:
        """Create a thumbnail image from the full-sized plot."""
        thumbnail_path = self.output_path.with_name(
            self.output_path.stem + "_thumb.png"
        )

        with Image.open(self.output_path) as img:
            scale = self.thumbnail_dpi / self.dpi
            width, height = img.size
            new_size: Tuple[int, int] = (
                int(scale * width),
                int(scale * height),
            )
            img.thumbnail(new_size, Image.Resampling.LANCZOS)
            img.save(thumbnail_path)

    def get_thumbnail_path(self) -> Path:
        return self.output_path.with_name(self.output_path.stem + "_thumb.png")
