# Hoja de ruta

Esta hoja de ruta separa deuda de reproducibilidad de nuevas preguntas de investigación. No implica fechas ni compromisos de release.

## 1. Cerrar la reproducción del experimento actual

- Publicar el PDF final y los recursos necesarios para compilar `informe.tex`.
- Publicar en una GitHub Release el dataset crudo de calibración y los checkpoints elegidos, con checksums.
- Registrar hardware y versiones de cada benchmark temporal.
- Empaquetar `src/spider_rl/ros` como paquete `ament_python` y agregar un launch file con parámetros.
- Incorporar el firmware/bridge ESP32 o enlazar su repositorio y versión exacta.

## 2. Medición sim-to-real cuantitativa

- Guardar trayectorias sincronizadas de simulación y OptiTrack en un formato abierto.
- Definir métricas de error de trayectoria, éxito, tiempo físico y consumo de acciones.
- Repetir por target y semilla, reportando distribución e intervalos de confianza.
- Medir latencia pose → inferencia → comando y duración real de cada motion.

## 3. Robustez del modelo

- Estimar incertidumbre por transición, no solo media y ruido fijo del 5 %.
- Comparar memoria de una acción con ventanas más largas o modelos recurrentes.
- Evaluar randomización de dinámica, fricción y latencia.
- Separar conjuntos de calibración y validación física.

## 4. Navegación más compleja

- Validar físicamente las variantes con obstáculos.
- Integrar percepción compatible con hardware y estudiar fallos del LiDAR simulado.
- Evaluar múltiples objetivos, restricciones de seguridad y escenarios multi-robot.
- Comparar con baselines clásicos adicionales y una implementación DWA canónica.

## Criterio para marcar un hito como completo

Cada hito debe incluir código, datos o enlace persistente, comando reproducible, versiones, semillas, métricas y una descripción de limitaciones. Una demostración aislada en video es evidencia complementaria, no sustituto de esos artefactos.
