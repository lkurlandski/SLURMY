"""
Assist with I/O operations.
"""

from pathlib import Path
from typing import ClassVar


class OutputHelper:
    _model_path: ClassVar[str] = "model/"
    _trainer_log_file: ClassVar[str] = "trainer_log.json"
    _test_results_file: ClassVar[str] = "test_results.json"
    _learning_curve_file: ClassVar[str] = "learning_curve.png"

    def __init__(self, output_root: Path, dataset_name: str, model_name: str) -> None:
        self.output_root = output_root
        self.dataset = dataset_name
        self.model_name = model_name

    @property
    def path(self) -> Path:
        return self.output_root / self.dataset / self.model_name

    @property
    def trainer_log_file(self) -> Path:
        return self.path / self._trainer_log_file

    @property
    def test_results_file(self) -> Path:
        return self.path / self._test_results_file

    @property
    def learning_curve_file(self) -> Path:
        return self.path / self._learning_curve_file

    @property
    def model_path(self) -> Path:
        return self.path / self._model_path

    def mkdir(self, exist_ok: bool = False) -> None:
        self.output_root.mkdir(parents=True, exist_ok=exist_ok)
        self.path.mkdir(parents=True, exist_ok=exist_ok)
        self.model_path.mkdir(parents=True, exist_ok=exist_ok)
