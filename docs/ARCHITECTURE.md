# Arquitectura y decisiones de diseño

## Problema

El agente debe llevar un hexápodo hacia un objetivo usando únicamente información local. En vez de simular articulaciones y contactos, el entorno abstrae cada motion real como una transformación rígida medida experimentalmente.

## MDP

En la variante estándar, la observación es la posición relativa del objetivo `o_t = [x_t, y_t]`. El robot se mantiene en el origen de su marco. Las 12 acciones corresponden directamente a IDs de movimiento del controlador físico.

La transición aplica el desplazamiento calibrado `(Δx, Δy, Δθ)` al marco del robot:

```text
p[t+1] = R(-Δθ) · (p[t] - [Δx, Δy])
```

Se agrega ruido gaussiano proporcional de 5 % a cada componente. Una semilla pasada a `reset` gobierna tanto el estado inicial como este ruido.

## Dinámica condicionada por el movimiento anterior

La postura mecánica y la fase de marcha hacen que un comando no tenga siempre el mismo efecto. La variante `previous_steps` selecciona el desplazamiento a partir de una matriz de transición:

```text
(Δx_t, Δy_t, Δθ_t) = M(movimiento_anterior, acción_actual)
```

La observación añade un one-hot de 12 posiciones. El estado inicial usa una fila especial `__initial__`. La opción `--zero-previous-action` conserva la arquitectura de 14 entradas pero anula esa información, permitiendo la ablación con el mismo tamaño de red.

## Recompensa

La recompensa combina:

```text
R_t = α Δdistancia - costo_paso
    + mejora_orientación_si_hay_progreso
    - penalización_si_hay_retroceso
    + bonus_de_éxito
```

La señal diferencial evita premiar simplemente estados cercanos y favorece progreso efectivo. Los valores implementados están en cada clase de entorno (`distance_scale`, `step_cost`, `orientation_weight`, `backtrack_penalty`, `success_bonus`).

## PPO

Los entrenamientos usan Stable-Baselines3, una MLP con dos capas de 64 unidades para actor y crítico, activación `Tanh`, `n_steps=256`, `batch_size=8`, `n_epochs=20` y `gamma≈0.9791`. `src/spider_rl/training/` contiene la configuración autoritativa.

## Línea base discreta

`DiscreteTreeSearchController` expande secuencias sobre las mismas 12 acciones y puntúa cada transición con una recompensa equivalente a la del entorno. `horizon` controla profundidad y `beam_width` limita las ramas retenidas. El costo crece aproximadamente con `O(B · H · |A|)` y, a diferencia de PPO, requiere acceso al modelo dinámico durante la decisión.

El nombre histórico de algunos scripts usa “DWA”, pero la implementación actual es una búsqueda discreta en árbol inspirada como baseline de planificación local; no es una implementación canónica del Dynamic Window Approach continuo.

## Frontera software/hardware

La simulación y evaluación terminan en un ID de comando. En hardware, `src/spider_rl/ros/ppo_publisher_node.py`:

1. recibe poses de OptiTrack a través de `mocap4r2_msgs/RigidBodies`;
2. transforma el target del marco global al marco local del robot;
3. ejecuta inferencia PPO;
4. publica un `std_msgs/Int32` en `cm550_command`;
5. el bridge ESP32 convierte ese ID al paquete UART del CM-550.

El firmware del ESP32 y la configuración de OptiTrack no forman parte de este repositorio.
