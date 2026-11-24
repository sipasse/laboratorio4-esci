"""Escenarios reproducibles y evidencia de recuperación, compra y voz."""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
import json

import pandas as pd

from asistente import Asistente, dinero, es_confirmacion, presupuesto_explicito, reglas
from esci import METODOS, tokens
from tienda import Restriccion, Tienda


@dataclass
class Caso:
    nombre: str
    turnos: tuple
    estados: dict
    compra: dict = field(default_factory=dict)
    stock: dict = field(default_factory=dict)
    eventos: dict = field(default_factory=dict)
    voz: bool = False
    presupuesto: int | None = None
    idempotencia: bool = False
    corregir_asr: bool = False
    entidades_audio: tuple = ()


def _guardar(path, datos):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def anclas_del_catalogo(indice, desarrollo):
    """Comprueba las fichas de los guiones usando juicios de desarrollo."""
    exactas = desarrollo.loc[desarrollo.esci_label.eq("E")]
    familias = ("manta eléctrica", "termómetro Beurer")
    grupos = []
    for familia, cantidad in zip(familias, (2, 1)):
        claves = set(tokens(familia))
        ids = {
            p for q, p in zip(exactas["query"], exactas.product_id)
            if set(tokens(q)) == claves and claves.issubset(tokens(indice.fichas[p]["titulo"]))
        }
        if len(ids) < cantidad:
            raise ValueError(f"El guion requiere {cantidad} fichas exactas de desarrollo para {familia}.")
        grupos.extend(sorted(ids)[:cantidad])
    return (*grupos, *familias)


def definir_casos(a, b, c, familia, otra_familia):
    buscar = f"Busca {familia}"
    buscar_dos = f"Busca dos unidades de {familia} con presupuesto de 100 soles"
    barata = "Elijo la opción más barata"
    confirmar = "Confirmo la compra"
    mencion = {"producto": " ".join(tokens(familia))}
    solicitud = {**mencion, "cantidad": 2, "importe": 10000}
    eleccion = {"producto": "opcion:menor_precio"}
    return [
        Caso("01_compra_voz", (buscar_dos, barata, confirmar),
             {0: "opciones", 1: "cotizacion", 2: "pedido"}, {a: 2}, voz=True, presupuesto=10000,
             entidades_audio=(solicitud, eleccion, {})),
        Caso("02_aclaracion_voz", ("Quiero uno", buscar),
             {0: "aclaracion", 1: "opciones"}, voz=True, entidades_audio=({}, mencion)),
        Caso("03_cambio_voz", (buscar_dos, barata, "Mejor una unidad", confirmar),
             {1: "cotizacion", 2: "cotizacion", 3: "pedido"}, {a: 1}, voz=True, presupuesto=10000,
             entidades_audio=(solicitud, eleccion, {"cantidad": 1}, {})),
        Caso("04_agotado_voz", (buscar,), {0: "sin_opciones"}, stock={a: 0, b: 0, c: 0}, voz=True,
             entidades_audio=(mencion,)),
        Caso("05_correccion_asr", (buscar_dos, barata, "Corrijo, dos unidades", confirmar),
             {2: "cotizacion", 3: "pedido"}, {a: 2}, voz=True, corregir_asr=True, presupuesto=10000,
             entidades_audio=(solicitud, eleccion, {"cantidad": 2}, {})),
        Caso("06_stock_y_repeticion_voz", (buscar_dos, barata, confirmar, barata, confirmar, confirmar),
             {1: "cotizacion", 2: "alternativa", 3: "cotizacion", 4: "pedido", 5: "pedido"}, {b: 2},
             eventos={2: {a: 0, c: 0}}, voz=True, idempotencia=True, presupuesto=10000,
             entidades_audio=(solicitud, eleccion, {}, eleccion, {}, {})),
        Caso("07_confirmacion_sin_cotizacion", (confirmar,), {0: "aclaracion"}),
        Caso("08_presupuesto_insuficiente", (f"Busca {familia} hasta 20 soles",), {0: "sin_opciones"}),
        Caso("09_presupuesto_exacto", (f"Busca dos unidades de {familia} hasta 60 soles", barata, confirmar),
             {2: "pedido"}, {a: 2}, presupuesto=6000),
        Caso("10_cantidad_invalida", (f"Busca cero unidades de {familia}",), {0: "restriccion"}),
        Caso("11_cantidad_mayor_que_stock", (f"Busca 20 unidades de {familia}",), {0: "sin_opciones"}),
        Caso("12_presupuesto_cambia", (buscar_dos, barata, "Presupuesto de 20 soles", confirmar),
             {1: "cotizacion", 3: "aclaracion"}, presupuesto=2000),
        Caso("13_cancelacion", (buscar, barata, "Cancela la compra", confirmar), {2: "cancelado", 3: "aclaracion"}),
        Caso("14_retirar_del_carrito", (buscar, barata, "Quita el producto", "Ver carrito"), {2: "carrito", 3: "carrito"}),
        Caso("15_consultar_color", (buscar, barata, "Qué color tiene"), {2: "informacion"}),
        Caso("16_opcion_inexistente", (buscar, "Elijo la opción nueve"), {1: "aclaracion"}),
        Caso("17_carrito_multilinea", (buscar, barata, f"Busca {otra_familia}", "Elijo la opción uno", confirmar),
             {1: "cotizacion", 3: "cotizacion", 4: "pedido"}, {a: 1, c: 1}),
        Caso("18_comparacion", (buscar, "Compara las opciones"), {0: "opciones", 1: "opciones"}),
        Caso("19_confirmacion_negada", (buscar, barata, "Todavía no confirmo la compra", "Ver carrito"), {3: "carrito"}),
        Caso("20_stock_sin_sustituto", (buscar_dos, barata, confirmar),
             {1: "cotizacion", 2: "sin_opciones"}, eventos={2: {a: 0, b: 0, c: 0}}),
    ]


def _pares_stock(anchors, overrides):
    a, b, c = anchors
    precios = {a: 3000, b: 4400, c: 5900}
    cantidades = {a: 5, b: 6, c: 4, **overrides}
    return {p: (precios[p], cantidades[p]) for p in precios}


def _entidades(texto):
    p = reglas(texto)
    importe = presupuesto_explicito(texto)
    try:
        presupuesto = None if importe is None else dinero(importe)
    except Restriccion:
        presupuesto = "importe inválido"
    producto = " ".join(tokens(p.consulta)) or p.product_id or None
    if p.menor_precio:
        producto = "opcion:menor_precio"
    elif p.opcion is not None:
        producto = f"opcion:{p.opcion}"
    return {"cantidad": p.cantidad, "importe": presupuesto, "producto": producto}


def comparar_entidades(referencia, reconocido):
    comparacion = {}
    for campo, actual in _entidades(reconocido).items():
        esperado = referencia.get(campo)
        tipo = None
        if esperado != actual:
            tipo = "adicion" if esperado is None else "omision" if actual is None else "sustitucion"
        comparacion[campo] = {"esperado": esperado, "observado": actual, "tipo": tipo,
                              "ocurrencia": esperado is not None or actual is not None, "error": tipo is not None}
    return comparacion


def word_error_rate(ref, pred):
    ref, pred = tokens(ref), tokens(pred)
    fila = list(range(len(pred) + 1))
    for i, palabra in enumerate(ref, 1):
        nueva = [i]
        for j, otra in enumerate(pred, 1):
            nueva.append(min(fila[j] + 1, nueva[-1] + 1, fila[j - 1] + (palabra != otra)))
        fila = nueva
    return fila[-1] / max(1, len(ref))


def preparar_error_controlado(whisper, entrada, salida):
    """Enmascara el tramo de la cantidad en una entrada sintética de diagnóstico."""
    import soundfile as sf
    limpia = whisper.transcribir(entrada, palabras=True)
    palabras = limpia.get("palabras", [])
    objetivo = next((i for i, p in enumerate(palabras) if tokens(p["texto"])[:1] in (["dos"], ["2"])), None)
    if objetivo is None:
        return Path(entrada), {"tipo": "sin_enmascarar", "transcripcion_limpia": limpia["texto"]}
    audio, sr = sf.read(entrada)
    tramo = palabras[objetivo:objetivo + 3]
    if tokens(" ".join(p["texto"] for p in tramo[1:])) not in (["unidades", "de"], ["unidad", "de"]):
        tramo = tramo[:1]
    inicio = max(0, round((tramo[0]["inicio"] - 0.03) * sr))
    fin = min(len(audio), round((tramo[-1]["fin"] + 0.03) * sr))
    audio[inicio:fin] = 0
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    sf.write(salida, audio, sr, subtype="PCM_16")
    return salida, {"tipo": "cantidad_enmascarada", "transcripcion_limpia": limpia["texto"],
                    "intervalo_s": [inicio / sr, fin / sr], "archivo_original": str(entrada),
                    "tramo": " ".join(p["texto"] for p in tramo)}


def entradas_sinteticas(casos, carpeta):
    carpeta = Path(carpeta)
    return [(carpeta / caso.nombre / f"entrada_{i + 1:02d}.wav", texto)
            for caso in casos if caso.voz for i, texto in enumerate(caso.turnos)]


def ejecutar_caso(caso, indice, anclas, planificador=None, carpeta=None,
                  whisper=None, voz=None, carpeta_entradas=None, semilla=42, fuente_audio="sintetica"):
    """Restaura la misma tienda para una ejecución por audio o por texto."""
    modo = "voz" if whisper is not None and voz is not None else "texto"
    if modo == "voz" and (carpeta is None or carpeta_entradas is None or len(caso.entidades_audio) != len(caso.turnos)):
        raise ValueError("La prueba por voz requiere archivos y anotaciones de referencia para todos los turnos.")
    tienda = Tienda(indice.fichas, semilla)
    tienda.restaurar(_pares_stock(anclas, caso.stock), stock_otros=0)
    sesi = caso.nombre
    asistente = Asistente(indice, tienda, planificador, sesion=sesi)
    inicial = tienda.estado(sesi)
    pasos, fallos, infracciones = [], [], []
    carpeta = None if carpeta is None else Path(carpeta) / caso.nombre / modo
    try:
        for i, texto_correcto in enumerate(caso.turnos):
            cambios = caso.eventos.get(i, {})
            for pid, n in cambios.items():
                tienda.fijar_stock(pid, n)
            inv_previo = tienda.inventario()
            pedidos_previos = tienda.pedidos(sesi)
            texto, transcrito, audio_in, audio_out, hablado, degradacion = texto_correcto, None, None, None, None, None
            reloj = perf_counter()
            asr_s, tts_s = 0.0, 0.0
            if modo == "voz":
                audio_in = Path(carpeta_entradas) / caso.nombre / f"entrada_{i + 1:02d}.wav"
                if caso.corregir_asr and i == 0 and fuente_audio == "sintetica":
                    audio_in, degradacion = preparar_error_controlado(whisper, audio_in, carpeta / "entrada_alterada.wav")
                reloj = perf_counter()
                transcrito = whisper.transcribir(audio_in)
                texto = transcrito["texto"]
                asr_s = perf_counter() - reloj
            turno = asistente.procesar(texto)
            if modo == "voz":
                t = perf_counter()
                audio_out = carpeta / f"respuesta_{i + 1:02d}.wav"
                hablado = voz.sintetizar(turno["respuesta"], audio_out)
                tts_s = perf_counter() - t
            total = perf_counter() - reloj
            despues = tienda.estado(sesi)
            pedidos_nuevos = despues["pedidos"][len(pedidos_previos):]
            deltas_esperados = {}
            for pedido in pedidos_nuevos:
                if not es_confirmacion(texto_correcto):
                    infracciones.append({"turno": i + 1, "motivo": "pedido sin autorización en el guion"})
                suma = sum(x["precio"] * x["cantidad"] for x in pedido["lineas"])
                limite = pedido["presupuesto"]
                if suma != pedido["total"] or (limite is not None and suma > limite):
                    infracciones.append({"turno": i + 1, "motivo": "total o presupuesto inconsistente"})
                if caso.presupuesto is not None and limite != caso.presupuesto:
                    infracciones.append({"turno": i + 1, "motivo": "presupuesto del pedido distinto al del guion"})
                for linea in pedido["lineas"]:
                    pid, cantidad = linea["producto"], linea["cantidad"]
                    deltas_esperados[pid] = deltas_esperados.get(pid, 0) - cantidad
                    if cantidad <= 0 or cantidad > inv_previo[pid]["stock"]:
                        infracciones.append({"turno": i + 1, "motivo": "cantidad supera stock o rango permitido"})
            deltas = {p: dato["stock"] - inv_previo[p]["stock"] for p, dato in despues["inventario"].items()
                      if dato["stock"] != inv_previo[p]["stock"]}
            if deltas != deltas_esperados:
                infracciones.append({"turno": i + 1, "motivo": "descuento de inventario distinto a lo registrado"})
            esperado = caso.estados.get(i)
            if esperado and turno["estado"] != esperado:
                fallos.append(f"Turno {i + 1}: se esperaba {esperado}, se obtuvo {turno['estado']}.")
            prueba_audio = {}
            if modo == "voz":
                entendido = whisper.transcribir(audio_out)["texto"]
                prueba_audio = {"referencia_entrada": texto_correcto, "asr": transcrito, "degradacion": degradacion,
                                "entidades": comparar_entidades(caso.entidades_audio[i], texto),
                                "wer_entrada": word_error_rate(texto_correcto, texto),
                                "audio_entrada": str(audio_in), "audio_respuesta": str(audio_out),
                                "texto_sintetizado": hablado, "asr_respuesta": entendido,
                                "wer_respuesta": word_error_rate(hablado, entendido),
                                "similitud_vocal": voz.similitud(audio_out)}
            pasos.append({
                **turno, "eventos_inventario": cambios, "deltas_inventario": deltas, "audio": prueba_audio,
                "tiempos_s": {**turno["tiempos_s"], "asr": asr_s, "tts_y_wav": tts_s, "archivo_a_audio_listo": total},
            })
        final = tienda.estado(sesi)
        pedidos = final["pedidos"]
        compra = {x["producto"]: x["cantidad"] for p in pedidos for x in p["lineas"]}
        if compra != caso.compra or len(pedidos) != int(bool(caso.compra)):
            fallos.append(f"Compra esperada {caso.compra}. Se registraron {len(pedidos)} pedidos con {compra}.")
        if caso.presupuesto is not None and asistente.memoria.presupuesto != caso.presupuesto:
            fallos.append("El presupuesto final difiere del previsto para la conversación.")
        if caso.nombre in {"13_cancelacion", "14_retirar_del_carrito"} and final["carrito"]:
            fallos.append("El carrito debía quedar vacío.")
        if caso.idempotencia:
            prev, ult = [p["evidencia"].get("pedido") for p in pasos[-2:]]
            if not prev or not ult or not ult["repetido"] or prev["id"] != ult["id"] or pasos[-1]["deltas_inventario"]:
                fallos.append("La última confirmación no probó el mismo identificador y descuento único.")
        correccion = None
        if caso.corregir_asr and modo == "voz":
            antes = pasos[1]["evidencia"].get("cotizacion", {}).get("lineas", [])
            despues = pasos[2]["evidencia"].get("cotizacion", {}).get("lineas", [])
            previas = {x["producto"]: x["cantidad"] for x in antes}
            corregidas = {x["producto"]: x["cantidad"] for x in despues}
            correccion = {"error_asr": pasos[0]["audio"]["entidades"]["cantidad"],
                          "antes": previas, "despues": corregidas}
            if not correccion["error_asr"]["error"] or not antes or previas == corregidas or corregidas != caso.compra:
                fallos.append("La traza debe mostrar una cantidad transcrita errónea y una cotización corregida antes del pedido.")
        salida = {
            "escenario": caso.nombre, "modo": modo,
            "fuente_entrada": fuente_audio if modo == "voz" else "texto de referencia",
            "correcto": not fallos and not infracciones, "fallos": fallos, "incumplimientos": infracciones,
            "turnos": pasos, "estado_inicial": inicial, "estado_final": final, "correccion_asr": correccion,
        }
        if carpeta is not None:
            _guardar(carpeta / "traza.json", salida)
        return salida
    finally:
        tienda.cerrar()


def ejecutar_escenarios(casos, indice, anclas, planificador=None, carpeta=None,
                        whisper=None, voz=None, carpeta_entradas=None, fuente_audio="sintetica"):
    if (whisper is None) != (voz is None):
        raise ValueError("La evaluación de voz requiere el reconocedor y el sintetizador.")
    completa = whisper is not None and voz is not None
    filas, audios = [], []
    for caso in casos:
        texto = ejecutar_caso(caso, indice, anclas, planificador, carpeta)
        primario = texto
        if completa and caso.voz:
            primario = ejecutar_caso(caso, indice, anclas, planificador, carpeta,
                                     whisper, voz, carpeta_entradas, fuente_audio=fuente_audio)
            for i, (p, control) in enumerate(zip(primario["turnos"], texto["turnos"]), 1):
                audios.append({
                    "escenario": caso.nombre, "turno": i, "estado_audio": p["estado"],
                    "estado_transcripcion_correcta": control["estado"], "voz_correcta": primario["correcto"],
                    "referencia_correcta": texto["correcto"], **p["tiempos_s"],
                    "wer_entrada": p["audio"]["wer_entrada"], "wer_respuesta": p["audio"]["wer_respuesta"],
                    "origen_plan": p["origen_plan"], "similitud_vocal": p["audio"]["similitud_vocal"],
                    "entrada_enmascarada": (p["audio"]["degradacion"] or {}).get("tipo") == "cantidad_enmascarada",
                    **{f"error_{k}": int(d["error"]) for k, d in p["audio"]["entidades"].items()},
                    **{f"entidad_{k}": int(d["ocurrencia"]) for k, d in p["audio"]["entidades"].items()},
                    **{f"tipo_error_{k}": d["tipo"] for k, d in p["audio"]["entidades"].items()},
                })
        filas.append({"escenario": caso.nombre, "modalidad_evaluada": primario["modo"],
                      "correcto": primario["correcto"], "control_texto_correcto": texto["correcto"],
                      "incumplimientos": len(primario["incumplimientos"]), "detalle": " ".join(primario["fallos"])})
    resumen, voz_df = pd.DataFrame(filas), pd.DataFrame(audios)
    if carpeta is not None:
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        resumen.to_csv(carpeta / "escenarios.csv", index=False, encoding="utf-8-sig")
        if not voz_df.empty:
            voz_df.to_csv(carpeta / "turnos_de_voz.csv", index=False, encoding="utf-8-sig")
        def errores_entidades(tabla):
            return {campo: {"errores": int(tabla[f"error_{campo}"].sum()),
                            "ocurrencias": int(tabla[f"entidad_{campo}"].sum()),
                            "por_tipo": tabla[f"tipo_error_{campo}"].value_counts().to_dict()}
                    for campo in ("cantidad", "importe", "producto")}

        _guardar(carpeta / "resumen_operativo.json", {
            "modo": "evaluación integrada" if completa else "ejecución por texto",
            "escenarios": len(resumen), "correctos": int(resumen.correcto.sum()),
            "porcentaje_correcto": 100 * float(resumen.correcto.mean()),
            "incumplimientos": int(resumen.incumplimientos.sum()),
            "casos_voz_ejecutados": int(resumen.modalidad_evaluada.eq("voz").sum()),
            "errores_asr": errores_entidades(voz_df) if not voz_df.empty else {},
            "errores_asr_sin_enmascarar": errores_entidades(voz_df.loc[~voz_df.entrada_enmascarada])
            if not voz_df.empty else {},
        })
    return resumen, voz_df


def analizar_ranking(resultados, ranking, pares, fichas, carpeta):
    """Compara el primer resultado con coincidencias exactas para casos observados."""
    metodo = METODOS[2] if METODOS[2] in set(resultados.metodo) else METODOS[0]
    subset = resultados.loc[resultados.metodo.eq(metodo)]
    primeros = ranking.loc[ranking.metodo.eq(metodo) & ranking.posicion.eq(1)]
    exactos = set(primeros.loc[primeros.etiqueta.eq("E"), "query_id"])
    mejores_exactos = subset.loc[subset.query_id.isin(exactos)]
    peores = subset.loc[~subset.query_id.isin(exactos) & subset.ndcg10.lt(1 - 1e-9)].nsmallest(5, "ndcg10")
    if len(peores) < 5:
        otros = subset.loc[subset.ndcg10.lt(1 - 1e-9) & ~subset.query_id.isin(peores.query_id)]
        peores = pd.concat([peores, otros.nsmallest(5 - len(peores), "ndcg10")])
    cuadros = [
        ("acierto", mejores_exactos.nlargest(5, "ndcg10")),
        ("fallo", peores),
    ]
    ejemplos = []
    for tipo, bloque in cuadros:
        for fila in bloque.itertuples(index=False):
            mejores = ranking.loc[ranking.metodo.eq(metodo) & ranking.query_id.eq(fila.query_id)].sort_values("posicion")
            if mejores.empty or (tipo == "fallo" and fila.ndcg10 >= 1 - 1e-9):
                continue
            primero = mejores.iloc[0]
            consulta_tokens = set(tokens(fila.consulta))
            coincidencias = consulta_tokens.intersection(tokens(primero.titulo))
            candidatos = pares.loc[pares.query_id.eq(fila.query_id) & pares.esci_label.eq("E"), "product_id"]
            titulo_esperado = ""
            if not candidatos.empty:
                referencia = max(candidatos, key=lambda p: (len(consulta_tokens.intersection(tokens(fichas[p]["titulo"]))), p))
                titulo_esperado = fichas[referencia]["titulo"]
            posicion_e = mejores.loc[mejores.etiqueta.eq("E"), "posicion"]
            primera_exacta = "" if posicion_e.empty else int(posicion_e.iloc[0])
            faltantes = consulta_tokens.intersection(tokens(titulo_esperado)) - coincidencias
            if fila.ndcg10 >= 1 - 1e-9:
                lectura = "El orden de relevancia de los primeros diez puestos alcanza el valor ideal."
            elif primero.etiqueta == "E":
                lectura = "El primer resultado es exacto. Los candidatos posteriores explican la distancia al orden ideal."
            elif faltantes:
                lectura = "El primer título omite términos de la consulta que sí aparecen en la ficha exacta de referencia."
            else:
                lectura = f"El primer resultado tiene etiqueta {primero.etiqueta} pese a su coincidencia textual parcial."
            ejemplos.append({"tipo": tipo, "metodo": metodo, "consulta": fila.consulta, "ndcg10": fila.ndcg10,
                             "primer_titulo": primero.titulo, "primera_etiqueta": primero.etiqueta,
                             "primer_exacto_en_top10": primera_exacta, "titulo_exacto_referencia": titulo_esperado,
                             "terminos_compartidos": ", ".join(sorted(coincidencias)),
                             "terminos_omitidos_en_primer_titulo": ", ".join(sorted(faltantes)),
                             "diagnostico": lectura})
    tabla = pd.DataFrame(ejemplos)
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / "aciertos_y_fallos.csv", index=False, encoding="utf-8-sig")
    return tabla
