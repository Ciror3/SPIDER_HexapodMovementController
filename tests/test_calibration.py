import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider_rl.environments.previous_action import (
    ACTION_METADATA,
    INITIAL_PREVIOUS_MOTION,
    SpiderEnv,
)


class CalibrationTests(unittest.TestCase):
    def test_external_calibration_is_loaded(self):
        sources = [INITIAL_PREVIOUS_MOTION] + [name for _, name in ACTION_METADATA.values()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Source_Motion",
                        "Target_Motion",
                        "Target_ID",
                        "Avg_Delta_Fwd(m)",
                        "Avg_Delta_Lat(m)",
                        "Avg_Delta_Theta(deg)",
                    ]
                )
                for source in sources:
                    for command_id, target in ACTION_METADATA.values():
                        writer.writerow([source, target, command_id, 0.01, 0.02, 10.0])

            env = SpiderEnv(calibration_path=path)
            try:
                command_id, movement = env.commands[INITIAL_PREVIOUS_MOTION][0]
                self.assertEqual(command_id, ACTION_METADATA[0][0])
                np.testing.assert_allclose(movement, [0.02, -0.01, np.deg2rad(10.0)])
            finally:
                env.close()

    def test_incomplete_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.csv"
            path.write_text(
                "Source_Motion,Target_Motion,Target_ID,Avg_Delta_Fwd(m),"
                "Avg_Delta_Lat(m),Avg_Delta_Theta(deg)\n"
                "initial,Pivot_Left,227,0.0,0.0,0.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no cubre todas las transiciones"):
                SpiderEnv(calibration_path=path)


if __name__ == "__main__":
    unittest.main()
