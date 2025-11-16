============================================================
// Documentación
============================================================
//
// Descripción del modelo
// ------------------------------------------------------------
// Este proyecto implementa un sistema de detección y conteo de
// objetos presentes utilizando un modelo de detección de objetos
// ligero basado en YOLOv8.El modelo está entrenado para identificar
// objetos típicos del salón (por ejemplo: cpu, mesa, 
// mouse, nada/fondo, pantalla, silla y teclado).
-------------------------------------------------------------
   Construcción del conjunto de datos y entrenamiento
-------------------------------------------------------------
// Para el entrenamiento del modelo se construyó primero un
// conjunto de datos de detección a partir del repositorio
// "objetos_salon". Estas imágenes se utilizaron como en ejemplos
// de detección de un solo objeto y se formo un primer conjunto de datos.  
// Para complementar el conjunto de entrenamiento con escenas
// reales que contienen varios objetos a la vez, se tomaron
// videos del salón de cómputo y se extrajeron frames
// cada cierto número de fotogramas. Sobre estos frames se aplicó
// un modelo YOLOv8x preentrenado en COCO, que sirvió
// para generar etiquetas: para cada frame se obtuvieron
// las detecciones del modelo, se filtraron las clases
// relevantes y se mapearon las etiquetas COCO a las clases del
// proyecto (cpu, pantalla, mouse, silla, etc.). Cada detección
// se transformó al formato YOLO (clase, centro, ancho y alto
// normalizados) y se guardó en archivos de texto, generando así
// un segundo conjunto de datos.
//
// Los dos conjuntos de datos se unificaron en un
// único dataset YOLO con particiones de entrenamiento y
// validación. A partir de este dataset unificado se entrenó un
// modelo YOLOv8n ligero usando la librería Ultralytics en Google
// Colab, con un tamaño de entrada fijo y un número de épocas adecuado.
// Tras el entrenamiento se seleccionó el modelo con mejor rendimiento y se
// verificó su comportamiento aplicándolo a imágenes del salón
// comprobando que producía varias detecciones
// coherentes por imagen y un conteo razonable de objetos por
// clase.

----------------------------------------------------------------
 Exportación a TensorFlow Lite y modelo final
--- ------------------------------------------------------------
// Una vez validado el desempeño del detector, el modelo YOLOv8n
// entrenado se exportó a formato TensorFlow Lite (archivo
// ".tflite").El archivo TFLite contiene el grafo del modelo
// optimizado para inferencia, de forma que pueda ejecutarse de
// manera eficiente en el navegador sin necesidad de un servidor
// de backend.


---------------------------------------------------------------
 Estructura de la aplicación web y uso
 --------------------------------------------------------------
// La aplicación web se compone principalmente de tres archivos:
//
// - "index.html": define la interfaz gráfica básica (formulario
//   para cargar imágenes, botón para ejecutar la detección, área
//   de estado y un lienzo donde se dibuja la imagen con las
//   cajas de detección).
//
// - "app.js": carga el modelo TFLite en el navegador mediante
//   TensorFlow.js y tfjs-tflite, prepara la imagen
//   con la resolución de entrada del modelo, ejecuta la
//   inferencia, interpreta la salida y dibuja las cajas de
//   detección en color azul claro sobre la imagen. También
//   calcula y muestra una tabla con el conteo de objetos
//   detectados por clase.
//
// - "objetos_salon.tflite": archivo del modelo entrenado y
//   exportado a TensorFlow Lite, que contiene los pesos y la
//   arquitectura optimizada para inferencia.
//
// Para ejecutar el sistema localmente, los archivos "index.html",
// "app.js" y "objetos_salon.tflite" deben estar en la misma
// carpeta.
--------------------------------------------------
Para ejecutar
--------------------------------------------------
// En el buscador dentro de la carpeta se escribirá 
// "cmd" y posteriormente se ejecutará el comando
//
//   python -m http.server 8000
//
// Después, se accede desde el navegador a la dirección:
//
//   http://localhost:8000/
//
// En la página se selecciona una imagen general del salón de
// cómputo utilizando el selector de archivo y se pulsa el botón
// "Detectar objetos". El navegador ejecuta el modelo TFLite de
// forma local, dibuja las cajas de detección sobre la imagen y
// muestra el conteo de objetos detectados por clase, logrando
// así detectar y contar los elementos del salón sin depender de
// procesamiento en servidor.

