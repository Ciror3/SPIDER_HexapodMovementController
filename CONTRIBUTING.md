# Contribuir a S.P.I.D.E.R.

Gracias por querer mejorar el proyecto. Las contribuciones más útiles son las que mantienen trazabilidad entre calibración, simulación, evaluación y hardware.

## Preparar el entorno

```bash
git clone https://github.com/Ciror3/tpFInalRLSpider.git
cd tpFInalRLSpider
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[render,dev]'
python -m pytest
```

## Flujo de cambios

1. Abrí un issue para cambios de comportamiento o nuevos experimentos.
2. Creá una rama desde `main`.
3. Agregá o actualizá pruebas cuando cambie la dinámica, observación o recompensa.
4. Ejecutá `python -m pytest` y `python -m ruff check .`.
5. En el pull request explicá motivación, comando exacto de reproducción, semilla, versión del entorno y métricas antes/después.

No subas checkpoints periódicos, logs de TensorBoard ni cientos de mapas de política. Conservá como máximo un modelo final y las métricas agregadas de una corrida relevante; para artefactos grandes usá una GitHub Release y documentá su checksum.

## Convenciones experimentales

- Todas las coordenadas y traslaciones están en metros; los ángulos internos, en radianes.
- Informá siempre `seed`, cantidad de episodios, límite de pasos y variante del entorno.
- No compares tiempos de pared obtenidos en equipos distintos sin documentar CPU, GPU, sistema operativo y versiones.
- Un cambio a la tabla de movimientos calibrados altera el MDP y debe quedar explícito.

Al contribuir aceptás que tu aporte se distribuya bajo la licencia MIT del repositorio.
