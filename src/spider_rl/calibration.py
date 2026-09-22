"""Carga mediciones de OptiTrack para el modelo con movimiento previo."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


INITIAL_PREVIOUS_MOTION = "__initial__"
INITIAL_ALIASES = {"", "none", "initial", "inicio", INITIAL_PREVIOUS_MOTION}

ACTIONS = {
    0: (227, "Pivot_Left"),
    1: (228, "Pivot_Right"),
    2: (251, "FwdSteer_Left"),
    3: (252, "Fwd"),
    4: (253, "FwdSteer_Right"),
    5: (256, "BwdSteer_Left"),
    6: (258, "BwdSteer_Right"),
    7: (262, "FastFwd"),
    8: (266, "Bwd"),
    9: (267, "FastBwd"),
    10: (261, "FastFwdSteer_Left"),
    11: (263, "FastFwdSteer_Right"),
}

REQUIRED_COLUMNS = {
    "Source_Motion",
    "Target_Motion",
    "Target_ID",
    "Avg_Delta_Fwd(m)",
    "Avg_Delta_Lat(m)",
    "Avg_Delta_Theta(deg)",
}


def build_previous_commands(calibration_path):
    """Convierte el CSV de calibración a la tabla usada por ``SpiderEnv``.

    El archivo debe contener una fila por transición (movimiento anterior,
    movimiento actual). Las traslaciones se convierten desde la convención de
    OptiTrack usada durante la calibración a ``[x_forward, y_lateral]``.
    """

    path = Path(calibration_path)
    motion_to_action = {motion_name: action for action, (_, motion_name) in ACTIONS.items()}
    previous_commands = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Faltan columnas en {path}: {missing}")

        for line_number, row in enumerate(reader, start=2):
            source_motion = row["Source_Motion"].strip()
            if source_motion.lower() in INITIAL_ALIASES:
                source_motion = INITIAL_PREVIOUS_MOTION
            target_motion = row["Target_Motion"].strip()
            if target_motion not in motion_to_action:
                raise ValueError(
                    f"Movimiento objetivo desconocido en {path}:{line_number}: {target_motion}"
                )

            target_id = int(row["Target_ID"])
            action = motion_to_action[target_motion]
            expected_id = ACTIONS[action][0]
            if target_id != expected_id:
                raise ValueError(
                    f"Target_ID {target_id} no coincide con {target_motion} "
                    f"({expected_id}) en {path}:{line_number}"
                )

            delta_fwd = float(row["Avg_Delta_Fwd(m)"])
            delta_lat = float(row["Avg_Delta_Lat(m)"])
            delta_theta = np.deg2rad(float(row["Avg_Delta_Theta(deg)"]))

            # Convención validada durante la calibración original.
            movement = [delta_lat, -delta_fwd, delta_theta]
            previous_commands.setdefault(source_motion, {})[action] = (target_id, movement)

    return previous_commands


def main():
    parser = argparse.ArgumentParser(description="Valida y convierte una calibración de OptiTrack.")
    parser.add_argument("calibration_path")
    args = parser.parse_args()
    print(json.dumps(build_previous_commands(args.calibration_path), indent=2))


if __name__ == "__main__":
    main()
