import express from "express";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Soportar subida de imágenes en base64 de pizarras o apuntes
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ extended: true, limit: "25mb" }));

// Archivo persistente para que los puntos y tokens NO se reseteen al recargar el HTML o reiniciar
const USAGE_FILE = path.join(__dirname, "usage-stats.json");
const DAILY_TOKEN_LIMIT = 1_000_000;

interface UsageData {
  date: string;
  totalTokensUsed: number;
  promptTokensUsed: number;
  candidatesTokensUsed: number;
  callsCount: number;
}

function getTodayUtcString(): string {
  const now = new Date();
  return now.toISOString().split("T")[0]; // YYYY-MM-DD en UTC
}

function getMsUntilUtcMidnight(): number {
  const now = new Date();
  const tomorrow = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1, 0, 0, 0, 0));
  return Math.max(0, tomorrow.getTime() - now.getTime());
}

function loadUsageData(): UsageData {
  const today = getTodayUtcString();
  try {
    if (fs.existsSync(USAGE_FILE)) {
      const raw = fs.readFileSync(USAGE_FILE, "utf-8");
      const data: UsageData = JSON.parse(raw);
      // Si es de un día anterior, reiniciamos el contador diario automáticamente
      if (data.date === today) {
        return data;
      }
    }
  } catch (err) {
    console.error("Error al leer archivo de uso persistente:", err);
  }

  const initial: UsageData = {
    date: today,
    totalTokensUsed: 0,
    promptTokensUsed: 0,
    candidatesTokensUsed: 0,
    callsCount: 0,
  };
  saveUsageData(initial);
  return initial;
}

function saveUsageData(data: UsageData) {
  try {
    fs.writeFileSync(USAGE_FILE, JSON.stringify(data, null, 2), "utf-8");
  } catch (err) {
    console.error("Error al guardar archivo de uso persistente:", err);
  }
}

let sessionUsage: UsageData = loadUsageData();

function trackUsage(usageMetadata?: any) {
  const today = getTodayUtcString();
  if (sessionUsage.date !== today) {
    sessionUsage = {
      date: today,
      totalTokensUsed: 0,
      promptTokensUsed: 0,
      candidatesTokensUsed: 0,
      callsCount: 0,
    };
  }

  const p = usageMetadata?.promptTokenCount || 0;
  const c = usageMetadata?.candidatesTokenCount || 0;
  const t = usageMetadata?.totalTokenCount || p + c;

  sessionUsage.promptTokensUsed += p;
  sessionUsage.candidatesTokensUsed += c;
  sessionUsage.totalTokensUsed += t;
  sessionUsage.callsCount += 1;

  saveUsageData(sessionUsage);

  const remainingTokens = Math.max(0, DAILY_TOKEN_LIMIT - sessionUsage.totalTokensUsed);
  const remainingSlides = Math.floor(remainingTokens / 1200);
  const remainingKeywords = Math.floor(remainingTokens / 450);
  const remainingPhotoAnalysis = Math.floor(remainingTokens / 1100);
  const remainingQuickQuestions = Math.floor(remainingTokens / 280);

  return {
    callTokens: { prompt: p, candidates: c, total: t },
    session: {
      date: sessionUsage.date,
      totalUsed: sessionUsage.totalTokensUsed,
      dailyLimit: DAILY_TOKEN_LIMIT,
      remainingTokens,
      remainingSlides,
      remainingKeywords,
      remainingPhotoAnalysis,
      remainingQuickQuestions,
      msUntilReset: getMsUntilUtcMidnight(),
      isLow: remainingTokens < DAILY_TOKEN_LIMIT * 0.15,
    },
  };
}

// Lazy/safe Gemini Client initialization
function getGenAI(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "La clave GEMINI_API_KEY no está configurada en las variables de entorno. Puedes configurarla en el panel de Secrets de Google AI Studio."
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

// Helper to call Gemini with retry on transient 503/429 errors
async function generateWithRetry(ai: GoogleGenAI, config: any) {
  const modelsToTry = [config.model || "gemini-3.8-flash", "gemini-3.1-flash-lite"];
  let lastErr: any = null;

  for (const model of modelsToTry) {
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        return await ai.models.generateContent({
          ...config,
          model,
        });
      } catch (err: any) {
        lastErr = err;
        const msg = String(err?.message || "");
        const isTransient =
          msg.includes("503") ||
          msg.includes("UNAVAILABLE") ||
          msg.includes("high demand") ||
          msg.includes("429") ||
          msg.includes("RESOURCE_EXHAUSTED");

        if (isTransient && attempt < 3) {
          await new Promise((r) => setTimeout(r, attempt * 1000));
          continue;
        }
        break;
      }
    }
  }
  throw lastErr;
}

// Health check endpoint
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    hasGeminiKey: Boolean(process.env.GEMINI_API_KEY),
    sessionTokens: sessionUsage,
  });
});

// Endpoint: Estadísticas de tokens con persistencia real y tiempo hasta reinicio
app.get("/api/tokens", (_req, res) => {
  const today = getTodayUtcString();
  if (sessionUsage.date !== today) {
    sessionUsage = loadUsageData();
  }

  const remainingTokens = Math.max(0, DAILY_TOKEN_LIMIT - sessionUsage.totalTokensUsed);
  res.json({
    date: sessionUsage.date,
    totalUsed: sessionUsage.totalTokensUsed,
    dailyLimit: DAILY_TOKEN_LIMIT,
    remainingTokens,
    remainingSlides: Math.floor(remainingTokens / 1200),
    remainingKeywords: Math.floor(remainingTokens / 450),
    remainingPhotoAnalysis: Math.floor(remainingTokens / 1100),
    remainingQuickQuestions: Math.floor(remainingTokens / 280),
    msUntilReset: getMsUntilUtcMidnight(),
    isLow: remainingTokens < DAILY_TOKEN_LIMIT * 0.15,
  });
});

// Endpoint: Chat rápido para resolver dudas inmediatas
app.post("/api/chat", async (req, res) => {
  try {
    const { message = "", history = [] } = req.body;

    if (!message || typeof message !== "string" || !message.trim()) {
      res.status(400).json({ error: "Por favor escribe una pregunta rápida." });
      return;
    }

    const ai = getGenAI();

    const systemPrompt = `Eres un asistente escolar y tutor experto, rápido y directo.
Tu misión es resolver dudas inmediatas de estudiantes sobre su exposición, tema, dudas conceptuales o preguntas rápidas.
Directivas:
- Responde de forma clara, concisa y sin rodeos (máximo 2-3 párrafos o puntos directos).
- Si te piden un ejemplo o cómo decirlo en voz alta, dalo de inmediato con lenguaje natural.
- Sé amable, motivador y pedagógico.`;

    const contents: any[] = [];
    if (Array.isArray(history) && history.length > 0) {
      for (const h of history.slice(-6)) {
        if (h.role && h.text) {
          contents.push({
            role: h.role === "assistant" ? "model" : "user",
            parts: [{ text: h.text }],
          });
        }
      }
    }

    contents.push({
      role: "user",
      parts: [{ text: message }],
    });

    const response = await generateWithRetry(ai, {
      model: "gemini-3.8-flash",
      contents,
      config: {
        systemInstruction: systemPrompt,
        temperature: 0.3,
      },
    });

    const reply = response.text?.trim() || "";
    const usage = trackUsage(response.usageMetadata);

    res.json({
      reply,
      tokenUsage: usage,
    });
  } catch (error: any) {
    console.error("Error en chat rápido:", error);
    res.status(500).json({
      error: error?.message || "Ocurrió un error al procesar tu pregunta rápida.",
    });
  }
});

// Endpoint: Organizar presentación (listas para copiar y pegar)
app.post("/api/slides", async (req, res) => {
  try {
    const { text = "", image, schoolLevel = "3º ESO", summaryLevel = "medium", writingStyle = "Claro y directo" } = req.body;

    if (!text.trim() && !image) {
      res.status(400).json({ error: "Por favor proporciona un texto o adjunta una foto para estructurar la presentación." });
      return;
    }

    const ai = getGenAI();

    const summaryMap: Record<string, string> = {
      verylong: "muy completa, con 6 a 8 diapositivas con desarrollo detallado",
      long: "completa, con 5 a 6 diapositivas bien desarrolladas",
      medium: "equilibrada, con 4 a 5 diapositivas bien enfocadas",
      short: "corta y directa, con 3 a 4 diapositivas concisas",
      veryshort: "muy sintética, de 2 a 3 diapositivas esenciales",
    };

    const systemPrompt = `Eres un diseñador y redactor experto de presentaciones académicas y profesionales.
Tu misión es estructurar una PRESENTACIÓN DE DIAPOSITIVAS lista para copiar y pegar directamente en diapositivas (PowerPoint, Google Slides o Canva).

DIRECTIVAS ESTRICTAS:
1. NO añadas "keywords" ni etiquetas dentro de las diapositivas.
2. NO uses frases como "Para explicar en voz alta", "Decir al público", ni muletillas orales.
3. Cada diapositiva debe contener:
   - "title": Título claro, directo y representativo del tema de la diapositiva.
   - "content": El contenido real de la diapositiva redactado en puntos limpios o párrafos concisos y bien formateados, listo para copiar y pegar en la diapositiva.
4. Incluye:
   - "title": Título general de la presentación.
   - "intro": Introducción limpia de apertura de la presentación.
   - "slides": Lista ordenada de diapositivas con "title" y "content".
   - "conclusion": Conclusión o cierre sintetizado.
5. Adapta la complejidad al nivel: ${schoolLevel}.
6. Longitud y profundidad: ${summaryMap[summaryLevel] || "equilibrada"}.
7. Tono: ${writingStyle}.
8. Si se adjunta una foto (de libro, pizarra o apuntes), extrae meticulosamente toda la información visible relevante y organízala.`;

    const contents: any[] = [];
    if (image && image.data && image.mimeType) {
      contents.push({
        inlineData: {
          data: image.data,
          mimeType: image.mimeType,
        },
      });
    }

    let promptText = "Por favor crea una presentación formal de diapositivas estructurada y lista para copiar y pegar.";
    if (text.trim()) {
      promptText += `\n\nApuntes/texto de referencia:\n${text}`;
    }
    contents.push({ text: promptText });

    const response = await generateWithRetry(ai, {
      model: "gemini-3.8-flash",
      contents,
      config: {
        systemInstruction: systemPrompt,
        temperature: 0.2,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: {
              type: Type.STRING,
              description: "Título general de la presentación",
            },
            intro: {
              type: Type.STRING,
              description: "Texto limpio de la introducción para la primera diapositiva",
            },
            slides: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: {
                    type: Type.STRING,
                    description: "Título directo de la diapositiva",
                  },
                  content: {
                    type: Type.STRING,
                    description: "Contenido limpio de la diapositiva, listo para copiar y pegar (sin 'en voz alta' ni keywords)",
                  },
                },
                required: ["title", "content"],
              },
              description: "Conjunto de diapositivas que conforman la presentación",
            },
            conclusion: {
              type: Type.STRING,
              description: "Texto limpio de la conclusión o cierre",
            },
          },
          required: ["title", "intro", "slides", "conclusion"],
        },
      },
    });

    const outputText = response.text?.trim() || "";
    if (!outputText) {
      throw new Error("El modelo no devolvió ningún contenido.");
    }

    const data = JSON.parse(outputText);
    const usage = trackUsage(response.usageMetadata);

    res.json({
      ...data,
      tokenUsage: usage,
    });
  } catch (error: any) {
    console.error("Error al generar diapositivas:", error);
    res.status(500).json({
      error: error?.message || "Ocurrió un error inesperado al estructurar la presentación.",
    });
  }
});

// Endpoint: Explicar foto O TEXTO de pizarra, libro o apuntes
app.post("/api/explain-photo", async (req, res) => {
  try {
    const { image, text = "", notes = "", level = "normal" } = req.body;

    if ((!image || !image.data || !image.mimeType) && !text.trim()) {
      res.status(400).json({ error: "Por favor adjunta una foto o pega el texto de la pizarra, libro o apuntes." });
      return;
    }

    const ai = getGenAI();

    const levelDescriptions: Record<string, string> = {
      "muy fácil": "explicación extremadamente sencilla, apta para que cualquier persona o un niño lo entienda a la primera, con analogías cotidianas y sin tecnicismos complejos",
      fácil: "explicación fácil, muy clara, con vocabulario accesible y oraciones sencillas",
      normal: "explicación equilibrada, clara, directa y completa, conservando los términos necesarios pero explicándolos de forma perfectamente entendible",
    };

    const targetLevel = levelDescriptions[level] || levelDescriptions["normal"];

    const systemPrompt = `Eres un docente experto con un don especial para hacer que cualquier tema complejo sea súper fácil de comprender.
El usuario te proporciona contenido que proviene de una pizarra escolar, un libro o apuntes (ya sea en fotografía, en texto copiado o ambos).

Tu trabajo es:
1. Analizar cuidadosamente todas las palabras, ideas, diagramas, fórmulas, listas o esquemas.
2. Identificar el tema principal y crear un TÍTULO formulado, atractivo y descriptivo.
3. Redactar una SÍNTESIS ORGANIZADA de lo que decía la pizarra o libro de forma limpia y comprensible.
4. Redactar una EXPLICACIÓN FORMULADA adaptada exactamente al nivel pedido: ${targetLevel}. Explicarlo de otra forma para que el estudiante realmente lo entienda y se le quede grabado.
5. Formular los PUNTOS CLAVE esenciales ordenados paso a paso.
6. Proporcionar un RESUMEN EN UNA FRASE para recordar siempre el concepto central.`;

    const contents: any[] = [];
    if (image && image.data && image.mimeType) {
      contents.push({
        inlineData: {
          data: image.data,
          mimeType: image.mimeType,
        },
      });
    }

    let promptText = `Por favor analiza este contenido de pizarra/libro y explícalo a nivel "${level}".`;
    if (text.trim()) {
      promptText += `\n\nTexto de la pizarra/libro transcrito:\n${text}`;
    }
    if (notes.trim()) {
      promptText += `\n\nNotas adicionales del usuario:\n${notes}`;
    }
    contents.push({ text: promptText });

    const response = await generateWithRetry(ai, {
      model: "gemini-3.8-flash",
      contents,
      config: {
        systemInstruction: systemPrompt,
        temperature: 0.2,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: {
              type: Type.STRING,
              description: "Título formulado del tema",
            },
            photoContentSummary: {
              type: Type.STRING,
              description: "Lo que aparece en la pizarra o libro, sintetizado y bien organizado",
            },
            explanation: {
              type: Type.STRING,
              description: "Explicación clara y didáctica formulada para que se entienda fácilmente según el nivel elegido",
            },
            keyPoints: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Puntos clave explicados paso a paso",
            },
            oneSentenceTakeaway: {
              type: Type.STRING,
              description: "Una frase resumen definitiva para acordarse de la idea principal",
            },
          },
          required: ["title", "photoContentSummary", "explanation", "keyPoints", "oneSentenceTakeaway"],
        },
      },
    });

    const outputText = response.text?.trim() || "";
    if (!outputText) {
      throw new Error("El modelo no devolvió ningún contenido.");
    }

    const data = JSON.parse(outputText);
    const usage = trackUsage(response.usageMetadata);

    res.json({
      ...data,
      tokenUsage: usage,
    });
  } catch (error: any) {
    console.error("Error al analizar y explicar foto/texto:", error);
    res.status(500).json({
      error: error?.message || "Ocurrió un error al analizar el contenido con Gemini.",
    });
  }
});

// Endpoint: Sacar keywords ultracortas punto y aparte
app.post("/api/keywords", async (req, res) => {
  try {
    const { text = "", image, count = 8 } = req.body;

    if (!text.trim() && !image) {
      res.status(400).json({ error: "Por favor proporciona un texto o una foto para extraer las palabras clave." });
      return;
    }

    const ai = getGenAI();
    const targetCount = Math.max(3, Math.min(25, Number(count) || 8));

    const instructions = `Eres un preparador experto de exposiciones orales.
El usuario necesita KEYWORDS nemotécnicas para guiarse al hablar en público sin leer.

DEFINICIÓN EXACTA DE KEYWORD:
- Son ULTRA CORTAS: de 1 a 4 palabras por línea (máximo 5 palabras).
- NUNCA generes un párrafo continuo ni texto en bloque.
- Deben ser punto y aparte: cada keyword es una línea independiente.
- Ejemplo del usuario:
  Listen carefully
  understand ideas
  respect
  trust
- Si una idea o frase del texto es muy interesante o llamativa pero larga, pon solo el inicio esencial seguido de puntos suspensivos (...) para activar el recuerdo al exponer. Ejemplo: "Comenzó la batalla en...", "El 99% de la masa...".
- No metas explicaciones largas ni frases completas con sujeto, verbo y predicado. Solo conceptos clave, cifras o verbos de acción.`;

    const contents: any[] = [];
    if (image && image.data && image.mimeType) {
      contents.push({
        inlineData: {
          data: image.data,
          mimeType: image.mimeType,
        },
      });
    }

    let promptText = `Extrae entre ${Math.max(3, targetCount - 2)} y ${targetCount + 2} keywords ultracortas (1 a 4 palabras cada una), en líneas separadas punto y aparte.`;
    if (text.trim()) {
      promptText += `\n\nTexto original:\n${text}`;
    }
    contents.push({ text: promptText });

    const response = await generateWithRetry(ai, {
      model: "gemini-3.8-flash",
      contents,
      config: {
        systemInstruction: instructions,
        temperature: 0.2,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            keywordsList: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Lista ordenada de keywords ultracortas (1 a 4 palabras por elemento)",
            },
            formattedText: {
              type: Type.STRING,
              description: "Las keywords unidas por saltos de línea (una por línea, punto y aparte)",
            },
          },
          required: ["keywordsList", "formattedText"],
        },
      },
    });

    const outputText = response.text?.trim() || "";
    if (!outputText) {
      throw new Error("El modelo no devolvió ningún contenido.");
    }

    const data = JSON.parse(outputText);
    const usage = trackUsage(response.usageMetadata);

    res.json({
      ...data,
      tokenUsage: usage,
    });
  } catch (error: any) {
    console.error("Error al extraer keywords:", error);
    res.status(500).json({
      error: error?.message || "Ocurrió un error inesperado al extraer las keywords con Gemini.",
    });
  }
});

// Vite middleware / Production static serving
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Servidor de Apoyo para la Presentación ejecutándose en http://0.0.0.0:${PORT}`);
  });
}

startServer();
