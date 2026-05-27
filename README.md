<div align="center">

# 🚶 SmartFlow
### Sistema de Análisis de Flujo Peatonal en Campus Universitario

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=flat)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv)](https://opencv.org)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-F9AB00?style=flat&logo=google-colab)](https://colab.research.google.com)
[![Universidad Autónoma de Occidente](https://img.shields.io/badge/UAO-Ingeniería%20Multimedia-CC0000?style=flat)](https://www.uao.edu.co)

**Procesamiento Digital de Imágenes · Semestre I 2026**  
Victor Manuel Salazar Chaparro · Juan Camilo Casierra Riascos · Brandon Montilla Hernandez

</div>

---

## 📋 Descripción

SmartFlow es un sistema basado en **visión por computador** que analiza el flujo peatonal en el campus de la Universidad Autónoma de Occidente (UAO). Procesa imágenes estáticas de tres zonas clave (cafetería, sótanos y biblioteca), detecta personas con **YOLOv8**, genera **mapas de calor de densidad gaussiana** y extrae métricas cuantitativas de movilidad para apoyar decisiones de gestión de espacios.

### ¿Qué hace el sistema?

```
Imágenes (Drive) → Preprocesamiento CLAHE → Detección YOLOv8
                                                    ↓
                         Mapa de calor ← Tracking de posiciones
                                ↓
                   Métricas de movilidad + Exportación CSV/JSON
```

### Resultados obtenidos (325 imágenes reales del campus UAO)

| Zona | Personas promedio | Confianza promedio | Nivel de congestión |
|------|:-----------------:|:------------------:|:-------------------:|
| Cafetería | 2.1 | 0.464 | Baja – Media |
| Sótano | 1.3 | 0.359 | Baja |
| Biblioteca | 0.5 | 0.212 | Sin actividad – Baja |
| **Global** | **1.3** | **0.345** | **Baja (42.8%)** |

> ⏱️ Tiempo promedio de inferencia: **0.97 s/imagen** en Google Colab con GPU T4

---

## 📁 Estructura del repositorio

```
SmartFlow/
├── ProyectoFinalPDI_CasierraMontillaSalazar.ipynb   # Notebook principal
├── README.md                                         # Este archivo
└── resultados/                                       # (generado al ejecutar)
    ├── smartflow_metricas.csv
    └── smartflow_resumen.json
```

---

## 🗄️ Dataset

Las imágenes del campus UAO están alojadas en Google Drive:

📂 **[Acceder al dataset en Google Drive](https://drive.google.com/drive/u/0/folders/1r4eyGC5GG8QqUsz_atkIPzaCYSrSQ0WH)**

El dataset contiene **325 imágenes reales** organizadas en carpetas por zona:

```
ProyectoFinal_PDI/
└── imagenes/
    ├── Cafeteria/     # Imágenes de la cafetería del campus
    ├── Sotanos/       # Imágenes de los sótanos
    └── Biblioteca/    # Imágenes de la biblioteca
```

> Las imágenes fueron capturadas en distintos momentos del día y desde diferentes ángulos, cubriendo condiciones variadas de iluminación y densidad de personas.

---

## ⚙️ Instrucciones de ejecución

### Requisito previo

El notebook está diseñado para ejecutarse en **Google Colab** (recomendado con GPU T4 para mejor rendimiento). No requiere instalación local.

### Paso 1 — Abrir el notebook en Colab

1. Ve a [Google Colab](https://colab.research.google.com)
2. Selecciona **Archivo → Abrir cuaderno → GitHub** (o sube el `.ipynb` directamente)
3. Activa la GPU: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**

### Paso 2 — Instalar dependencias

Ejecuta la **Celda 1**. Esto instala automáticamente:

```python
!pip install ultralytics --quiet
!pip install opencv-python-headless --quiet
!pip install seaborn --quiet
```

### Paso 3 — Montar Google Drive y configurar rutas

Ejecuta la **Celda 2**. Cuando se te solicite, autoriza el acceso a tu Google Drive.

Asegúrate de que la estructura de carpetas en tu Drive sea:

```
MyDrive/
└── ProyectoFinal_PDI/
    ├── imagenes/
    │   ├── Cafeteria/
    │   ├── Sotanos/
    │   └── Biblioteca/
    └── resultados/        # Se crea automáticamente
```

> Si usas rutas diferentes, edita el diccionario `ZONE_FOLDERS` en la Celda 2.

### Paso 4 — Ejecutar el pipeline completo

Ejecuta las celdas en orden (Celda 3 → Celda 10). El sistema:

1. **Celda 3** — Importaciones y parámetros globales
2. **Celda 4** — Carga del modelo YOLOv8n
3. **Celda 5** — Definición de funciones del sistema
4. **Celda 6** — Pipeline principal (procesa todas las imágenes)
5. **Celda 7** — Métricas globales y análisis comparativo
6. **Celda 8** — Gráficas de análisis por zona
7. **Celda 9** — Mapa de calor acumulado
8. **Celda 10** — Exportación a CSV y JSON

### Paso 5 (opcional) — Inspeccionar una imagen específica

En la **Celda 11**, cambia el valor de `INSPECT_INDEX` para analizar cualquier imagen individual:

```python
INSPECT_INDEX = 0   # 0 = primera imagen, 1 = segunda, etc.
```

---

## 🛠️ Parámetros configurables

Puedes ajustar el comportamiento del sistema modificando estas variables en la **Celda 3**:

| Parámetro | Valor por defecto | Descripción |
|-----------|:-----------------:|-------------|
| `CONF_THRESHOLD` | `0.40` | Confianza mínima para aceptar una detección |
| `IOU_THRESHOLD` | `0.45` | Umbral IoU para Non-Maximum Suppression |
| `HEATMAP_SIGMA` | `40` | Radio de influencia gaussiana (píxeles) |
| `HEATMAP_ALPHA` | `0.55` | Transparencia del overlay del mapa de calor |

Para mayor precisión (a costa de velocidad), cambia el modelo en la Celda 4:

```python
model = YOLO('yolov8s.pt')   # Small  — más preciso
model = YOLO('yolov8m.pt')   # Medium — aún más preciso
```

---

## 📊 Salidas del sistema

Al finalizar la ejecución encontrarás en `resultados/`:

| Archivo | Descripción |
|---------|-------------|
| `smartflow_metricas.csv` | Tabla con métricas por imagen (zona, personas, confianza, nivel de congestión, tiempo) |
| `smartflow_resumen.json` | Resumen ejecutivo del análisis completo |

Las visualizaciones se generan directamente en el notebook:
- Imágenes anotadas con bounding boxes y nivel de congestión
- Mapas de calor por imagen y acumulados por zona
- Gráficas comparativas entre zonas

---

## 🏗️ Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                        SmartFlow                            │
├────────────┬──────────────┬──────────────┬──────────────────┤
│ Módulo 1   │   Módulo 2   │   Módulo 3   │    Módulo 4      │
│Preproces.  │  Detección   │  Mapa calor  │  Análisis zonal  │
│  CLAHE     │   YOLOv8n    │  Gaussiano   │  Cuadrícula 3×4  │
└────────────┴──────────────┴──────────────┴──────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Módulo 5         │
                    │  Visualización y      │
                    │  exportación CSV/JSON │
                    └───────────────────────┘
```

### Niveles de congestión

| Nivel | Rango | Acción sugerida |
|-------|:-----:|-----------------|
| 🔘 Sin actividad | 0 personas | Sin intervención |
| 🟢 Baja | 1 – 3 | Monitoreo pasivo |
| 🟡 Media | 4 – 8 | Alerta temprana |
| 🟠 Alta | 9 – 15 | Redistribución de flujo |
| 🔴 Crítica | > 15 | Intervención inmediata |

---

## 📦 Dependencias

| Librería | Versión | Uso |
|----------|---------|-----|
| `ultralytics` | ≥ 8.0 | Modelo YOLOv8 para detección de personas |
| `opencv-python-headless` | ≥ 4.x | Preprocesamiento CLAHE y manipulación de imágenes |
| `numpy` | ≥ 1.24 | Operaciones matriciales y generación de mapas de calor |
| `pandas` | ≥ 2.0 | Gestión de métricas y exportación CSV |
| `matplotlib` | ≥ 3.7 | Visualizaciones y gráficas comparativas |
| `seaborn` | ≥ 0.12 | Gráficas estadísticas |

> Todas las dependencias se instalan automáticamente al ejecutar la Celda 1 del notebook.

---

## 👥 Autores

| Nombre | Código |
|--------|--------|
| Victor Manuel Salazar Chaparro | 2221679 |
| Juan Camilo Casierra Riascos | 2226055 |
| Brandon Montilla Hernandez | 2220324 |

**Profesor:** Nicolas Llanos Neuta  
**Materia:** Procesamiento Digital de Imágenes  
**Facultad:** Ingeniería y Ciencias Básicas — Ingeniería Multimedia  
**Universidad Autónoma de Occidente, Cali, Colombia — 2026**

---

## 📄 Licencia

Proyecto académico desarrollado para la materia Procesamiento Digital de Imágenes de la Universidad Autónoma de Occidente. Uso educativo.
