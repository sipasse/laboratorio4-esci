"""Modelos preentrenados que se ejecutan en el kernel de Colab."""

from dataclasses import fields
from pathlib import Path
from time import perf_counter
import gc
import json
import re

import numpy as np

from asistente import ACCIONES, Plan, es_confirmacion, presupuesto_explicito, reglas
from esci import hash_archivo, normalizar


E5_ID = "intfloat/multilingual-e5-small"
AGENTE_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ASR_ID = "Systran/faster-whisper-small"
DISENO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLON_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REF_TEXTO = (
    "Esta es una voz sintética creada para una práctica de procesamiento del habla. "
    "En esta grabación leo un fragmento sobre un viaje por la costa, con un ritmo pausado y una pronunciación clara."
)


def liberar_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dispositivo():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def dtype_modelo(device):
    import torch
    if device == "cpu":
        return torch.float32
    # T4 usa float16. La emulación de bfloat16 no garantiza soporte de SDPA.
    usa_bfloat16 = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if usa_bfloat16 else torch.float16


class E5:
    dimension = 384

    def __init__(self, device=None, lote=24, max_length=256):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or dispositivo()
        self.lote = lote
        t = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(E5_ID)
        dtype = dtype_modelo(self.device)
        self.model = AutoModel.from_pretrained(E5_ID, dtype=dtype).to(self.device).eval()
        self.config = {"modelo": E5_ID, "revision": getattr(self.model.config, "_commit_hash", None),
                       "max_length": max_length, "lote": lote, "dtype": str(dtype), "pooling": "promedio_con_mascara"}
        self.carga_s = perf_counter() - t

    def encode(self, textos, prefijo):
        import torch
        from torch.nn.functional import normalize
        vectors = []
        with torch.inference_mode():
            for i in range(0, len(textos), self.lote):
                lote = [f"{prefijo}: {t}" for t in textos[i:i + self.lote]]
                enc = self.tokenizer(lote, padding=True, truncation=True,
                                     max_length=self.config["max_length"], return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state.float()
                mask = enc["attention_mask"].unsqueeze(-1)
                pooling = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.append(normalize(pooling, p=2, dim=-1).cpu().numpy())
        return np.concatenate(vectors) if vectors else np.empty((0, self.dimension), dtype=np.float32)


class PlanificadorQwen:
    def __init__(self, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or dispositivo()
        t = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(AGENTE_ID)
        self.model = AutoModelForCausalLM.from_pretrained(AGENTE_ID, dtype=dtype_modelo(self.device))
        self.model = self.model.to(self.device).eval()
        self.carga_s = perf_counter() - t
        self.recuperaciones = 0
        self.llamadas = 0
        self.origen_plan = "reglas"

    def _generar(self, sistema, turno, tokens):
        import torch
        mensajes = [{"role": "system", "content": sistema}, {"role": "user", "content": turno}]
        entrada = self.tokenizer.apply_chat_template(
            mensajes, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        self.llamadas += 1
        with torch.inference_mode():
            out = self.model.generate(
                **entrada, do_sample=False, max_new_tokens=tokens, pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(out[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True)

    def recomendar(self, consulta, preferencias, opciones):
        prompt = (
            "Selecciona el número de una de las opciones de compra aportadas para la consulta. "
            "Considera el título recuperado, la preferencia, el precio en céntimos y el stock. "
            "Devuelve solo el número de opción. Usa 0 si faltan datos para elegir."
        )
        contexto = json.dumps({"consulta": consulta, "preferencias": preferencias, "opciones": opciones}, ensure_ascii=False)
        salida = self._generar(prompt, contexto, 8).strip()
        return int(salida) if re.fullmatch(r"[1-4]", salida) else None

    def __call__(self, texto, contexto):
        directas = reglas(texto)
        self.origen_plan = "reglas"
        if es_confirmacion(texto):
            return directas
        if directas.accion == "aclarar" and directas.atributo == "presupuesto":
            return directas
        prompt = (
            "Interpreta un turno de compra en español. Devuelve exclusivamente un objeto JSON. "
            f"El campo accion debe ser uno de {', '.join(sorted(ACCIONES))}. "
            "Campos opcionales: consulta (nombre del producto para buscar), opcion (entero desde 1), "
            "product_id (solo un ID mostrado en contexto), cantidad (entero de unidades), "
            "presupuesto_soles (número que el usuario acaba de indicar), marca, color, "
            "menor_precio (booleano), atributo (marca, color, precio o stock). "
            "Omite los datos que no fueron mencionados en este turno. Un presupuesto no es cantidad. "
            "Usa cambiar para fijar una cantidad y sumar para añadir unidades a una línea existente. "
            "Usa agregar para elegir una opción nueva y comparar si se piden diferencias entre opciones. "
            "Si se piden alternativas para un producto nuevo, usa buscar y especifica consulta. "
            "Usa aclarar si falta identificar el producto. No inventes productos ni preferencias. "
            'Ejemplo: "Me quedo con la segunda" -> {"accion":"agregar","opcion":2}. '
            'Ejemplo: "Busca un ratón por hasta 80 soles" -> {"accion":"buscar","consulta":"ratón","cantidad":1,"presupuesto_soles":80}. '
            'Ejemplo: "Mejor dos unidades" -> {"accion":"cambiar","cantidad":2}.'
        )
        contexto = json.dumps(contexto, ensure_ascii=False)
        bruto = self._generar(prompt, f"Contexto de la tienda: {contexto}\nTurno del usuario: {texto}", 110)
        try:
            match = re.search(r"\{[^{}]*\}", bruto, re.S)
            if match is None:
                raise ValueError("El modelo no produjo un objeto JSON.")
            datos = json.loads(match[0])
            permitidos = {f.name for f in fields(Plan)}
            datos = {k: v for k, v in datos.items() if k in permitidos and v is not None}
            plan = Plan(**datos)
            enteros = (plan.opcion, plan.cantidad)
            strings = (plan.consulta, plan.product_id, plan.marca, plan.color, plan.atributo)
            if (
                not isinstance(plan.accion, str) or plan.accion not in ACCIONES
                or plan.accion in {"aclarar", "confirmar"}
                or any(v is not None and type(v) is not int for v in enteros)
                or any(not isinstance(v, str) for v in strings) or type(plan.menor_precio) is not bool
            ):
                raise ValueError("El plan no cumple el esquema.")
            self.origen_plan = "llm"
            if directas.accion != "aclarar":
                inicial = Plan(**vars(plan))
                plan.accion = directas.accion
                for campo in fields(Plan):
                    if campo.name != "accion" and getattr(directas, campo.name) != campo.default:
                        setattr(plan, campo.name, getattr(directas, campo.name))
                if directas.accion in {"buscar", "agregar", "sumar", "cambiar", "retirar"}:
                    plan.cantidad = directas.cantidad
                if directas.opcion is not None or directas.menor_precio:
                    plan.product_id = directas.product_id
                if directas.accion in {"cotizar", "cancelar", "carrito", "comparar", "preferencias"}:
                    self.origen_plan = "reglas"
                    return directas
                if plan != inicial:
                    self.origen_plan = "llm_con_campos_verificados"
            plan.presupuesto_soles = presupuesto_explicito(texto)
            for attr in ("marca", "color"):
                value = getattr(plan, attr)
                if value and normalizar(value) not in normalizar(texto) and getattr(directas, attr) != value:
                    setattr(plan, attr, "")
            return plan
        except (ValueError, TypeError):
            self.recuperaciones += 1
            self.origen_plan = "reglas"
            return directas


def compatibilidad_pyav():
    """Adapta el decodificador de Faster-Whisper a la API de PyAV 19."""
    import av
    from faster_whisper import audio

    if int(av.__version__.split(".", 1)[0]) < 19 or audio.av is not av:
        return

    class BackendAudio:
        def __getattr__(self, nombre):
            return getattr(av, nombre)

        def open(self, *args, **opciones):
            opciones.pop("metadata_errors", None)
            return av.open(*args, **opciones)

    audio.av = BackendAudio()


class Whisper:
    def __init__(self):
        from faster_whisper import WhisperModel
        compatibilidad_pyav()
        t = perf_counter()
        self.model = WhisperModel(ASR_ID, device="cpu", compute_type="int8", cpu_threads=4)
        self.carga_s = perf_counter() - t

    def transcribir(self, ruta, palabras=False):
        segmentos, info = self.model.transcribe(
            str(ruta), language="es", task="transcribe", beam_size=5,
            condition_on_previous_text=False, vad_filter=True, word_timestamps=palabras,
        )
        segmentos = list(segmentos)
        retorno = {"texto": " ".join(s.text.strip() for s in segmentos).strip(), "duracion": info.duration}
        if palabras:
            retorno["palabras"] = [
                {"texto": w.word.strip(), "inicio": w.start, "fin": w.end}
                for segmento in segmentos for w in (segmento.words or [])
            ]
        return retorno


def _audio_model(modelo, device):
    import torch
    from qwen_tts import Qwen3TTSModel

    precision = dtype_modelo(device)
    if modelo == CLON_ID and precision == torch.float16:
        # La predicción de códigos del clonador puede desbordarse en FP16 en Turing.
        precision = torch.float32
    return Qwen3TTSModel.from_pretrained(
        modelo, device_map=device, dtype=precision, attn_implementation="sdpa"
    )


def generar_audios_sinteticos(carpeta, entradas=(), semilla=42, device=None, referencia_autorizada=None):
    """Prepara la referencia de clonación y sintetiza las entradas de prueba."""
    import soundfile as sf
    import torch

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    device = device or dispositivo()
    ref = carpeta / "referencia_asistente.wav"
    registro = carpeta / "referencia_asistente.json"
    if referencia_autorizada is not None:
        archivo, texto = referencia_autorizada
        audio, frecuencia = sf.read(archivo)
        duracion = len(audio) / frecuencia
        if duracion < 3 or not texto.strip():
            raise ValueError("La referencia autorizada requiere al menos tres segundos y su transcripción.")
        sf.write(ref, audio, frecuencia, subtype="PCM_16")
        info = {"origen": "grabación autorizada indicada en la configuración", "texto": texto.strip(),
                "frecuencia_hz": frecuencia, "duracion_s": duracion, "sha256": hash_archivo(ref)}
        registro.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    entradas = [(Path(path), texto) for path, texto in entradas]
    crear_ref = not (ref.is_file() and registro.is_file())
    entradas = [(path, texto) for path, texto in entradas if not path.is_file()]
    if not crear_ref and not entradas:
        return ref, json.loads(registro.read_text(encoding="utf-8"))
    torch.manual_seed(semilla)
    t = perf_counter()
    modelo = _audio_model(DISENO_ID, device)
    carga_s = perf_counter() - t
    try:
        if crear_ref:
            t = perf_counter()
            audios, sr = modelo.generate_voice_design(
                text=REF_TEXTO, language="Spanish",
                instruct="An original synthetic adult female voice. Neutral Latin American Spanish. Warm and clear. Steady pace.",
                max_new_tokens=1100,
            )
            sf.write(ref, audios[0], sr, subtype="PCM_16")
            info = {"origen": "sintética diseñada para el laboratorio", "modelo": DISENO_ID, "texto": REF_TEXTO,
                    "semilla": semilla, "carga_modelo_s": carga_s, "sintesis_s": perf_counter() - t,
                    "frecuencia_hz": sr, "duracion_s": len(audios[0]) / sr, "sha256": hash_archivo(ref)}
            registro.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        for i in range(0, len(entradas), 4):
            lote = entradas[i:i + 4]
            wavs, sr = modelo.generate_voice_design(
                text=[texto for _, texto in lote], language=["Spanish"] * len(lote),
                instruct=["An original synthetic adult male voice. Neutral Latin American Spanish. Clear pronunciation."] * len(lote),
                max_new_tokens=420,
            )
            for (path, _), wav in zip(lote, wavs):
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, wav, sr, subtype="PCM_16")
        if referencia_autorizada is not None:
            info["carga_modelo_s"] = carga_s
            registro.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        del modelo
        liberar_cuda()
    return ref, json.loads(registro.read_text(encoding="utf-8"))


def para_hablar(texto):
    def leer_importe(match):
        soles, centimos = int(match[1]), int(match[2])
        return f"{soles} soles" + (f" y {centimos} céntimos" if centimos else "")
    return re.sub(r"S/\s*(\d+)\.(\d{2})", leer_importe, texto)


class VozClonada:
    def __init__(self, referencia, texto_referencia, device=None):
        self.referencia = Path(referencia)
        self.texto_referencia = texto_referencia
        t = perf_counter()
        self.modelo = _audio_model(CLON_ID, device or dispositivo())
        self.prompt = self.modelo.create_voice_clone_prompt(
            ref_audio=str(self.referencia), ref_text=texto_referencia, x_vector_only_mode=False
        )
        self.carga_s = perf_counter() - t

    def sintetizar(self, texto, salida):
        import soundfile as sf
        normal = " ".join(para_hablar(texto).split())
        if normalizar(normal) == normalizar(self.texto_referencia) or not normal:
            raise ValueError("La respuesta hablada debe contener texto nuevo.")
        # El presupuesto crece con el texto y acota repeticiones sin token de fin.
        max_tokens = max(96, 36 + 7 * len(normal.split()))
        wavs, sr = self.modelo.generate_voice_clone(
            text=normal, language="Spanish", voice_clone_prompt=self.prompt,
            max_new_tokens=max_tokens, do_sample=True, non_streaming_mode=True,
        )
        salida = Path(salida)
        salida.parent.mkdir(parents=True, exist_ok=True)
        sf.write(salida, wavs[0], sr, subtype="PCM_16")
        return normal

    def similitud(self, salida):
        from torch.nn.functional import cosine_similarity
        resultado = self.modelo.create_voice_clone_prompt(ref_audio=str(salida), x_vector_only_mode=True)
        ref = self.prompt[0].ref_spk_embedding.float().flatten()
        generado = resultado[0].ref_spk_embedding.float().flatten()
        return float(cosine_similarity(ref, generado, dim=0).item())
