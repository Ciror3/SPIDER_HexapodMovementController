# Calibración y despliegue físico

Esta sección documenta el prototipo usado en el trabajo. No es necesaria para ejecutar la simulación.

## Componentes

- robot hexápodo con controlador CM-550;
- ESP32/bridge que recibe `Int32` y emite el paquete UART correspondiente;
- sistema OptiTrack con rigid bodies para robot y target;
- ROS 2 y `mocap4r2_msgs`;
- computadora capaz de ejecutar Stable-Baselines3.

El firmware del ESP32, la definición física del robot y la configuración de Motive/OptiTrack no están incluidos. Por eso el despliegue no es reproducible únicamente con este repositorio.

## Convenciones de coordenadas

En la instalación experimental, OptiTrack usa `Y` vertical, `Z` forward y `X` lateral. La política consume `[forward, lateral]` en el marco local del robot. Verificá la orientación de los rigid bodies con movimientos controlados antes de habilitar una política autónoma; una inversión de ejes puede producir comandos opuestos.

## Nodos de referencia

- `src/spider_rl/ros/motion_publisher_node.py`: envío manual de IDs, útil para verificar el bridge.
- `src/spider_rl/ros/calibration_node.py`: mide desplazamientos medios con OptiTrack.
- `src/spider_rl/ros/ppo_publisher_node.py`: transforma pose, ejecuta PPO y publica el comando.

Estos archivos son nodos de referencia y deben integrarse a un workspace ROS 2 que ya contenga `mocap4r2_msgs` y el bridge del ESP32. Antes de correrlos, sourceá ROS 2 y el workspace correspondiente.

## Calibración condicionada

La variante con movimiento previo espera un CSV con una fila para cada par `(Source_Motion, Target_Motion)` y estas columnas:

```csv
Source_Motion,Target_Motion,Target_ID,Avg_Delta_Fwd(m),Avg_Delta_Lat(m),Avg_Delta_Theta(deg)
```

La fuente inicial puede escribirse como `__initial__`, `initial`, `inicio` o `none`. Validá el archivo antes de entrenar:

```bash
spider-calibration calibration_results_combinations_full.csv > /tmp/calibration.json
```

Luego pasalo explícitamente:

```bash
spider-train-previous \
  --calibration-path calibration_results_combinations_full.csv \
  --seed 0
```

El cargador rechaza columnas faltantes, IDs inconsistentes y matrices que no cubren las 13 × 12 transiciones requeridas.

## Secuencia segura de puesta en marcha

1. Levantá y verificá OptiTrack sin publicar comandos.
2. Confirmá nombres de rigid body, unidades y ejes.
3. Probá `motion_publisher_node.py` con el robot elevado o en un área contenida.
4. Compará un único movimiento con la tabla calibrada.
5. Cargá una política compatible con la dimensión de observación.
6. Probá objetivos lejanos con velocidad limitada y parada accesible.
7. Recién entonces habilitá una secuencia autónoma.

Nunca ejecutes un checkpoint desconocido ni una política de 14 entradas con un nodo que solo construye 2 entradas. El nodo de referencia incluido usa el modelo estándar de 2 entradas; desplegar el modelo con historial requiere mantener y enviar el one-hot de la acción anterior.
