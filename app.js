// =========================
// Configuración del modelo
// =========================

// Nombre de tu modelo .tflite (mismo folder que index.html)
const MODEL_URL = "objetos_salon.tflite"; // cámbialo si tu archivo se llama distinto

// Tamaño de entrada del modelo (el que usaste al exportar YOLO)
const IMAGE_SIZE = 640;

// Nombres de clases en el mismo orden que en el YAML de entrenamiento
// AJUSTA ESTA LISTA A TUS CLASES REALES
const CLASS_NAMES = [
  "cpu",
  "mesa",
  "mouse",
  "nada",
  "pantalla",
  "silla",
  "teclado"
];

const MAX_DETECTIONS = 50;
const IOU_THRESHOLD = 0.5;
const SCORE_THRESHOLD = 0.25;

let tfliteModel = null;
let imageLoaded = false;

// =========================
// Utilidades de dibujo
// =========================

function drawImageToCanvas(img, canvas) {
  const ctx = canvas.getContext("2d");
  const cw = canvas.width;
  const ch = canvas.height;

  ctx.clearRect(0, 0, cw, ch);

  const iw = img.width;
  const ih = img.height;
  const scale = Math.min(cw / iw, ch / ih);
  const drawW = iw * scale;
  const drawH = ih * scale;
  const offsetX = (cw - drawW) / 2;
  const offsetY = (ch - drawH) / 2;

  ctx.drawImage(img, offsetX, offsetY, drawW, drawH);
}

function drawDetections(canvas, boxesData, scoresData, classesData) {
  const ctx = canvas.getContext("2d");
  const cw = canvas.width;
  const ch = canvas.height;

  ctx.lineWidth = 2;
  ctx.textBaseline = "top";
  ctx.font = "14px system-ui";

  for (let i = 0; i < scoresData.length; i++) {
    const score = scoresData[i];
    const classId = classesData[i];

    const label = CLASS_NAMES[classId] || `cls_${classId}`;
    const caption = `${label} ${(score * 100).toFixed(1)}%`;

    const y1 = boxesData[i * 4 + 0] * ch;
    const x1 = boxesData[i * 4 + 1] * cw;
    const y2 = boxesData[i * 4 + 2] * ch;
    const x2 = boxesData[i * 4 + 3] * cw;

    const w = x2 - x1;
    const h = y2 - y1;

    // ======= Azul claro =======
    ctx.strokeStyle = "#60a5fa";                 // borde azul claro
    ctx.fillStyle = "rgba(96,165,250,0.18)";     // relleno azul claro semitransparente
    ctx.fillRect(x1, y1, w, h);
    ctx.strokeRect(x1, y1, w, h);

    const textWidth = ctx.measureText(caption).width;
    const textHeight = 14;
    const yText = y1 - (textHeight + 4);

    // Fondo del texto en azul claro
    ctx.fillStyle = "#60a5fa";
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + 8,
      textHeight + 6
    );

    // Texto en azul muy oscuro
    ctx.fillStyle = "#0f172a";
    ctx.fillText(caption, x1 + 2, yText < 0 ? 0 : yText + 1);
  }
}

function updateCounts(classesData) {
  const container = document.getElementById("counts");
  if (!classesData || classesData.length === 0) {
    container.innerHTML = "<p>No se detectaron objetos por encima del umbral.</p>";
    return;
  }

  const counts = {};
  for (const cid of classesData) {
    const name = CLASS_NAMES[cid] || `cls_${cid}`;
    counts[name] = (counts[name] || 0) + 1;
  }

  let html = "<p>Conteo de objetos detectados:</p>";
  html += "<table><thead><tr><th>Clase</th><th>Cantidad</th></tr></thead><tbody>";
  for (const [name, n] of Object.entries(counts)) {
    html += `<tr><td>${name}</td><td>${n}</td></tr>`;
  }
  html += "</tbody></table>";
  container.innerHTML = html;
}

// =========================
// Carga de imagen
// =========================

function setupImageInput() {
  const input = document.getElementById("image-input");
  const canvas = document.getElementById("canvas");
  const runButton = document.getElementById("run-button");
  const statusEl = document.getElementById("status");

  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      drawImageToCanvas(img, canvas);
      URL.revokeObjectURL(url);
      imageLoaded = true;
      runButton.disabled = false;
      statusEl.textContent = "Imagen cargada. Pulsa \"Detectar objetos\".";
      document.getElementById("counts").innerHTML = "";
    };
    img.onerror = () => {
      statusEl.textContent = "Error al cargar la imagen.";
      URL.revokeObjectURL(url);
    };
    img.src = url;
  });
}

// =========================
// Inferencia y postproceso
// =========================

async function runDetection() {
  const statusEl = document.getElementById("status");
  const canvas = document.getElementById("canvas");

  if (!tfliteModel || !imageLoaded) return;

  statusEl.textContent = "Ejecutando detección...";
  await tf.nextFrame();

  // 1) Entrada [1,H,W,3] normalizada a [0,1]
  const input = tf.tidy(() => {
    const tensor = tf.browser.fromPixels(canvas);        // [H,W,3]
    const expanded = tf.expandDims(tensor, 0);           // [1,H,W,3]
    const normalized = tf.div(expanded, tf.scalar(255)); // [0,1]
    return normalized;
  });

  // 2) Ejecutar modelo TFLite
  const rawOutput = tfliteModel.predict(input);
  input.dispose();

  // 2.1) Asegurarnos de tener un tf.Tensor
  let outputTensor;
  if (rawOutput instanceof tf.Tensor) {
    outputTensor = rawOutput;
  } else if (Array.isArray(rawOutput)) {
    outputTensor = rawOutput[0];
  } else {
    const firstKey = Object.keys(rawOutput)[0];
    outputTensor = rawOutput[firstKey];
  }

  console.log("Salida del modelo, shape:", outputTensor.shape);

  // 3) Interpretar salida YOLOv8: [1, C, N] -> [1, N, C]
  const transposed = tf.transpose(outputTensor, [0, 2, 1]); // [1,N,C]
  outputTensor.dispose();

  // 3.1) Cajas [y1,x1,y2,x2] normalizadas
  const boxes = tf.tidy(() => {
    const w = tf.slice(transposed, [0, 0, 2], [-1, -1, 1]);
    const h = tf.slice(transposed, [0, 0, 3], [-1, -1, 1]);
    const xCenter = tf.slice(transposed, [0, 0, 0], [-1, -1, 1]);
    const yCenter = tf.slice(transposed, [0, 0, 1], [-1, -1, 1]);

    const x1 = tf.sub(xCenter, tf.div(w, 2));
    const y1 = tf.sub(yCenter, tf.div(h, 2));
    const y2 = tf.add(y1, h);
    const x2 = tf.add(x1, w);

    const concat = tf.concat([y1, x1, y2, x2], 2); // [1,N,4]
    return tf.squeeze(concat);                      // [N,4] en [y1,x1,y2,x2]
  });

  // 3.2) Scores y clases
  const numClasses = CLASS_NAMES.length;
  const scoresAndClasses = tf.tidy(() => {
    const rawScores = tf.squeeze(
      tf.slice(transposed, [0, 0, 4], [-1, -1, numClasses]),
      0
    ); // [N,numClasses]

    const maxScores = tf.max(rawScores, 1);    // [N]
    const classIds = tf.argMax(rawScores, 1);  // [N]

    return { maxScores, classIds };
  });

  const scores = scoresAndClasses.maxScores;
  const classes = scoresAndClasses.classIds;

  // 4) Non-Max Suppression
  const nmsIdx = await tf.image.nonMaxSuppressionAsync(
    boxes,
    scores,
    MAX_DETECTIONS,
    IOU_THRESHOLD,
    SCORE_THRESHOLD
  );

  // 5) Recoger resultados
  const selectedBoxes = tf.gather(boxes, nmsIdx);
  const selectedScores = tf.gather(scores, nmsIdx);
  const selectedClasses = tf.gather(classes, nmsIdx);

  const boxesData = selectedBoxes.dataSync();
  const scoresData = selectedScores.dataSync();
  const classesData = Array.from(selectedClasses.dataSync());

  // 6) Dibujar y contar
  drawDetections(canvas, boxesData, scoresData, classesData);
  updateCounts(classesData);

  // 7) Limpieza
  tf.dispose([
    transposed,
    boxes,
    scores,
    classes,
    nmsIdx,
    selectedBoxes,
    selectedScores,
    selectedClasses
  ]);

  statusEl.textContent = "Detección finalizada.";
}

// =========================
// Inicialización
// =========================

async function init() {
  const statusEl = document.getElementById("status");
  const runButton = document.getElementById("run-button");

  try {
    statusEl.textContent = "Configurando backend CPU…";
    await tf.setBackend("cpu");
    await tf.ready();

    tflite.setWasmPath(
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/"
    );

    statusEl.textContent = "Cargando modelo TFLite…";
    tfliteModel = await tflite.loadTFLiteModel(MODEL_URL);

    statusEl.textContent = "Modelo cargado. Sube una imagen.";
    setupImageInput();
    runButton.disabled = true;

    runButton.addEventListener("click", () => {
      if (!runButton.disabled) {
        runDetection().catch((err) => {
          console.error(err);
          statusEl.textContent = "Error en la detección.";
        });
      }
    });
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error al inicializar TensorFlow/TFLite.";
  }
}

window.addEventListener("load", init);
