# Protocolo de reproducibilidad

## Niveles de reproducción

| Nivel | Se puede hacer desde este repositorio | Requiere |
|---|---|---|
| Contrato del entorno | Sí | Python y dependencias |
| Evaluación de modelos publicados | Sí | Checkpoints incluidos |
| Reentrenamiento | Sí, estadísticamente | Tiempo de cómputo; CPU o GPU |
| Recalibración | Parcial | Robot, OptiTrack, ROS 2 y bridge |
| Sim-to-real | Parcial | Todo el hardware y su configuración |

“Reproducible” no significa igualdad bit a bit entre CPU, GPU y versiones distintas de PyTorch. El objetivo es repetir el protocolo con las mismas semillas y obtener conclusiones comparables.

## Entorno de referencia

Los modelos finales registran Python 3.10, NumPy 2.2.6, Gymnasium 1.2.x y Stable-Baselines3 2.7/2.8. `requirements.txt` fija las dependencias directas compatibles con el modelo principal. PyTorch se instala según la plataforma para no imponer un build CUDA incorrecto.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest
```

Guardá el entorno exacto de una corrida:

```bash
python -m pip freeze > artifacts/requirements-frozen.txt
python - <<'PY'
import platform
import torch
print(platform.platform())
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
```

## Evaluación PPO vs. búsqueda discreta

```bash
spider-compare \
  --env-mode previous_steps \
  --ppo-model models/previous_action/ppo_previous_action.zip \
  --episodes 100 \
  --max-steps 200 \
  --seed 0 \
  --dwa-horizons 4,9,15,20 \
  --dwa-beam-widths 32,512,1024 \
  --output-dir artifacts/comparison
```

El script genera un CSV y gráficos. PPO y el baseline reciben los mismos targets, límite de pasos y semillas. Las primeras inferencias PPO se descartan del cronometraje mediante `--warmup-decisions`.

Para la ablación de observación:

```bash
spider-compare \
  --env-mode previous_steps \
  --zero-previous-action \
  --ppo-model models/ablations/ppo_no_previous_action.zip \
  --no-dwa \
  --episodes 100 \
  --max-steps 200 \
  --seed 0 \
  --output-dir artifacts/ablation-no-history
```

## Reentrenamiento

Usá un nombre que codifique experimento y semilla. Repetí con varias semillas para reportar media e intervalo, no una única corrida.

```bash
for seed in 0 1 2 3 4; do
  spider-train-previous \
    --total-timesteps 500000 \
    --n-envs 4 \
    --seed "$seed" \
    --run-name "previous-steps-seed-${seed}" \
    --output-dir artifacts
done
```

La generación de mapas de política durante entrenamiento puede ser costosa. Ajustá `--policy-map-freq 0` para desactivarla o aumentá su frecuencia; los checkpoints se controlan con `--checkpoint-freq`.

## Qué registrar al publicar un resultado

- commit de Git;
- comando completo;
- semilla(s), episodios y máximo de pasos;
- variante de entorno y si se usó acción previa;
- checkpoint y checksum SHA-256;
- versiones de Python, NumPy, Gymnasium, Stable-Baselines3 y PyTorch;
- CPU/GPU y sistema operativo para cualquier métrica temporal;
- archivo de calibración o versión de la tabla embebida.

Ejemplo de checksum:

```bash
sha256sum models/previous_action/ppo_previous_action.zip
```

Checksums de los artefactos publicados en esta revisión:

| Modelo | SHA-256 |
|---|---|
| PPO estándar | `da0d1d92baaa300654b8b8ea9d4f2a7fc3cbd065ef3aebc351c96ac97e61130b` |
| PPO con movimiento anterior | `26930ff62f1da46340f37d9f61eeb656e023c042b1f3d3d43fc51177fc2ffe40` |
| Ablación sin historial | `242ddbfa73b6457c23e8c52c684c79e4f958415d740b8ddbc8f153fc009a430d` |

## Amenazas a la validez

Los targets y el ruido son pseudoaleatorios y reproducibles, pero kernels de PyTorch/GPU pueden introducir no determinismo. El tiempo de decisión no mide la duración física del movimiento y no debe compararse entre máquinas sin normalización. Las métricas generadas por estas herramientas describen simulación; la transferencia física requiere un protocolo separado.
