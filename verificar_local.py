"""Verificación de desarrollo ejecutable en la terminal de VS Code."""

import argparse
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from asistente import Asistente, dinero, presupuesto_explicito, reglas
from esci import Indice, evaluar_recuperacion, ndcg10, preparar_datos
from evaluacion import analizar_ranking, anclas_del_catalogo, comparar_entidades, definir_casos, ejecutar_escenarios
from modelos import PlanificadorQwen
from tienda import Restriccion, Tienda


def catalogo_prueba():
    filas = [
        ("A", "Cable USB rojo para carga rápida", "Laboratorio", "rojo"),
        ("B", "Cable USB azul para datos y carga", "Laboratorio", "azul"),
        ("C", "Botella térmica de agua", "Prueba", ""),
    ]
    return pd.DataFrame([
        {"product_id": pid, "locale": "es", "titulo": titulo, "descripcion": "", "vinetas": "",
         "marca": marca, "color": color}
        for pid, titulo, marca, color in filas
    ])


def comprobar_regresiones(indice, casos):
    tienda = Tienda(indice.fichas)
    inventario = {"A": (3000, 5), "B": (4400, 6), "C": (5900, 4)}
    presupuesto_cien = "Busca dos unidades de cable USB con presupuesto de cien soles"
    barata = "Elijo la opción más barata"
    confirmar = "Confirmo la compra"
    try:
        tienda.restaurar(inventario)
        planificador = PlanificadorQwen.__new__(PlanificadorQwen)
        planificador.recuperaciones = 0
        planificador._generar = lambda *args: '{"accion":"preferencias","presupuesto_soles":50}'
        asistente = Asistente(indice, tienda, planificador)
        asistente.procesar(presupuesto_cien)
        assert asistente.procesar(barata)["evidencia"]["cotizacion"]["total"] == 6000
        asistente.procesar("Puedo gastar 50 soles")
        assert asistente.memoria.presupuesto == 5000
        assert asistente.procesar(confirmar)["estado"] == "aclaracion"
        assert tienda.carrito(asistente.sesion) == {"A": 2} and not tienda.pedidos(asistente.sesion)

        tienda.restaurar(inventario)
        asistente = Asistente(indice, tienda)
        asistente.procesar(presupuesto_cien)
        asistente.procesar(barata)
        assert asistente.procesar("Quiero cambiar mi presupuesto")["estado"] == "aclaracion"
        assert asistente.procesar(confirmar)["estado"] == "aclaracion"
        asistente.procesar("Ochenta soles con cincuenta céntimos")
        assert asistente.memoria.presupuesto == 8050
        assert asistente.procesar("Cotiza el carrito")["estado"] == "cotizacion"
        assert asistente.procesar(confirmar)["evidencia"]["pedido"]["presupuesto"] == 8050

        tienda.restaurar(inventario)
        asistente = Asistente(indice, tienda)
        asistente.procesar(presupuesto_cien)
        asistente.procesar(barata)
        assert asistente.procesar("Agrega una unidad más de la opción uno")["estado"] == "cotizacion"
        assert tienda.carrito(asistente.sesion) == {"A": 3}
        asistente.procesar("Mejor una unidad")
        assert tienda.carrito(asistente.sesion) == {"A": 1} and asistente.memoria.presupuesto == 10000

        tienda.restaurar(inventario)
        asistente = Asistente(indice, tienda)
        for turno in ("Busca dos unidades de cable USB hasta 200 soles", barata,
                      "Busca botella de agua", "Elijo la opción uno"):
            asistente.procesar(turno)
        tienda.fijar_stock("A", 0)
        opciones = asistente.procesar(confirmar)["evidencia"]["opciones"]
        assert [x["product_id"] for x in opciones] == ["B"]
        asistente.procesar("Elijo la opción uno")
        assert tienda.carrito(asistente.sesion) == {"B": 2, "C": 1}
        tienda.fijar_stock("B", 0)
        assert asistente.procesar(confirmar)["estado"] == "sin_opciones"
        assert tienda.carrito(asistente.sesion) == {"B": 2, "C": 1} and not tienda.pedidos(asistente.sesion)

        tienda.restaurar(inventario)
        asistente = Asistente(indice, tienda)
        asistente.procesar("Busca dos unidades de cable USB hasta 200 soles")
        asistente.procesar(barata)
        asistente.procesar("Elijo la opción dos con una unidad")
        assert tienda.carrito(asistente.sesion) == {"A": 2, "B": 1}
        tienda.fijar_stock("A", 0)
        assert asistente.procesar(confirmar)["estado"] == "alternativa"
        asistente.procesar("Elijo la opción uno")
        assert tienda.carrito(asistente.sesion) == {"B": 3}
    finally:
        tienda.cerrar()

    assert dinero(presupuesto_explicito("Mi límite es de S/ 1.500,25")) == 150025
    discrepancias = comparar_entidades({"importe": 8050}, "Mi límite es de ochenta soles con quince céntimos")
    assert discrepancias["importe"]["tipo"] == "sustitucion" and discrepancias["importe"]["observado"] == 8015
    for caso in casos:
        if caso.voz:
            assert len(caso.turnos) == len(caso.entidades_audio)
            for referencia, turno in zip(caso.entidades_audio, caso.turnos):
                assert not any(v["error"] for v in comparar_entidades(referencia, turno).values()), turno


def comprobar_comparacion_metodos():
    class EmbeddingsDePrueba:
        dimension = 2
        config = {"modelo": "vectores_fijos_de_prueba"}

        def encode(self, textos, prefijo):
            if prefijo == "query":
                return np.tile(np.array([1, 0], dtype=np.float32), (len(textos), 1))
            puntos = ([0, 1] if "rojo" in t else [1, 0] if "azul" in t else [-1, 0] for t in textos)
            return np.asarray(list(puntos), dtype=np.float32)

    indice = Indice(catalogo_prueba(), EmbeddingsDePrueba())
    pares = pd.DataFrame({"query_id": [1, 1, 1], "query": ["cable rojo"] * 3,
                          "product_id": ["A", "B", "C"], "esci_label": ["E", "S", "I"]})
    metricas, posiciones = evaluar_recuperacion(indice, pares)
    assert metricas.groupby("metodo").query_id.apply(list).tolist() == [[1], [1], [1]]
    primero = posiciones.loc[posiciones.posicion.eq(1)].set_index("metodo").product_id
    assert primero["BM25"] == "A" and primero["E5 sin BM25"] == "B" and primero["Híbrido BM25 + E5"] == "A"


def comprobaciones():
    indice = Indice(catalogo_prueba())
    casos = definir_casos("A", "B", "C", "cable usb", "botella de agua")
    resumen, audio = ejecutar_escenarios(casos, indice, ("A", "B", "C"))
    assert len(resumen) == 20 and audio.empty
    if not resumen.correcto.all():
        raise AssertionError(resumen.loc[~resumen.correcto, ["escenario", "detalle"]].to_string(index=False))
    assert not resumen.incumplimientos.any()
    assert abs(ndcg10([0.1, 1, 0]) - 0.6875501678) < 1e-8
    assert ndcg10([1, 0.1, 0]) == 1.0
    comprobar_regresiones(indice, casos)
    comprobar_comparacion_metodos()

    tienda = Tienda(indice.fichas)
    try:
        tienda.restaurar({"A": (3000, 5), "B": (4400, 6)}, stock_otros=0)
        tienda.fijar_carrito("fallo", {"A": 2, "B": 1}, 15000)
        oferta = tienda.cotizar("fallo", 15000)
        previo = tienda.estado("fallo")
        tienda.db.executescript("""
            CREATE TRIGGER fallo_inducido BEFORE UPDATE OF stock ON inventario
            WHEN OLD.producto = 'B' AND NEW.stock < OLD.stock
            BEGIN SELECT RAISE(ABORT, 'fallo inducido'); END;
        """)
        try:
            tienda.confirmar("fallo", oferta["id"], 15000)
        except sqlite3.IntegrityError:
            assert tienda.estado("fallo") == previo
        else:
            raise AssertionError("El fallo inducido debía revertir la compra.")
        tienda.db.execute("DROP TRIGGER fallo_inducido")
        pedido = tienda.confirmar("fallo", oferta["id"], 15000)
        posterior = tienda.estado("fallo")
        repetido = tienda.confirmar("fallo", oferta["id"], 15000)
        assert repetido["repetido"] and repetido["id"] == pedido["id"]
        assert tienda.estado("fallo") == posterior
    finally:
        tienda.cerrar()

    with TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "competencia.sqlite"
        uno, dos = Tienda(indice.fichas, ruta=ruta), Tienda(indice.fichas, ruta=ruta)
        try:
            uno.restaurar({"A": (3000, 1)}, stock_otros=0)
            uno.fijar_carrito("uno", {"A": 1})
            dos.fijar_carrito("dos", {"A": 1})
            cot1 = uno.cotizar("uno")
            cot2 = dos.cotizar("dos")
            uno.confirmar("uno", cot1["id"])
            try:
                dos.confirmar("dos", cot2["id"])
            except Restriccion as error:
                assert error.motivo == "stock" and not dos.pedidos("dos")
            else:
                raise AssertionError("La segunda sesión debía detectar el cambio de stock.")
        finally:
            uno.cerrar()
            dos.cerrar()

    tienda = Tienda(indice.fichas)
    try:
        tienda.restaurar({"A": (3000, 5), "B": (4400, 6)}, stock_otros=0)
        agente = Asistente(indice, tienda)
        agente.procesar("Busca dos unidades de cable USB con presupuesto de ciento veinte soles")
        agente.procesar("Elijo la opción más barata")
        assert agente.memoria.presupuesto == 12000
        assert agente.procesar("¿Confirmo la compra?")["estado"] == "aclaracion"
        agente.procesar("Qué color tiene")
        assert agente.procesar("Sí")["estado"] == "aclaracion"
        assert not tienda.pedidos(agente.sesion)
        cambio = agente.procesar("Mejor una unidad")
        assert cambio["estado"] == "cotizacion" and agente.memoria.cantidad_deseada == 1
        assert agente.memoria.presupuesto == 12000
        preferencia = agente.procesar("Prefiero color azul")
        assert [o["product_id"] for o in preferencia["evidencia"]["opciones"]] == ["B"]
        assert agente.procesar("Confirmo la compra")["estado"] == "aclaracion"
        alternativa = agente.procesar("Elijo la opción uno")
        assert alternativa["estado"] == "cotizacion" and tienda.carrito(agente.sesion) == {"B": 1}
        assert agente.procesar("Confirmo la compra")["estado"] == "pedido"
    finally:
        tienda.cerrar()
    cantidad_con_tope = reglas("Busca hasta dos unidades de cable USB")
    assert cantidad_con_tope.cantidad == 2 and cantidad_con_tope.presupuesto_soles is None
    assert dinero(reglas("Busca un cable USB hasta 80,50 soles").presupuesto_soles) == 8050
    return {
        "alcance": "verificación local por texto con tres fichas ficticias",
        "escenarios_correctos": int(resumen.correcto.sum()), "escenarios": len(resumen),
        "incumplimientos": int(resumen.incumplimientos.sum()),
        "casos_voz_ejecutados": 0, "modelos_ejecutados": False,
        "comprobaciones": ["nDCG con ejemplos de referencia", "reversión ante fallo parcial",
                           "confirmación idempotente", "competencia por stock",
                           "presupuesto verbal y decimal, límite de unidades y confirmación contextual",
                           "conservación de cantidad, filtros de preferencia e invalidación de la cotización",
                           "rebaja y aclaración del presupuesto, incremento de unidades y sustituciones multirenglón",
                           "anotaciones de entidades ASR y comparación de métodos con vectores fijos"],
        "casos": resumen[["escenario", "correcto", "incumplimientos"]].to_dict("records"),
    }


def comprobaciones_esci(origen):
    datos = preparar_datos("datos", origen_local=origen)
    indice = Indice(datos.catalogo)
    resultados, posiciones = evaluar_recuperacion(indice, datos.test, solo_lexico=True)
    a, b, c, familia, otra = anclas_del_catalogo(indice, datos.desarrollo)
    casos = definir_casos(a, b, c, familia, otra)
    escenarios, _ = ejecutar_escenarios(casos, indice, (a, b, c))
    consulta = datos.desarrollo.loc[datos.desarrollo.query_id.eq(10692)]
    original, distractor = "B08KHHGL8X", "B07MPFKD3G"
    if set(consulta.product_id) >= {original, distractor}:
        assert consulta.set_index("product_id").loc[distractor, "esci_label"] == "C"
        tienda = Tienda(indice.fichas)
        try:
            tienda.restaurar({original: (3000, 1), distractor: (5000, 2)}, stock_otros=0)
            asistente = Asistente(indice, tienda)
            asistente.procesar("Busca asiento 24 pulgadas")
            asistente.procesar("Elijo la opción más barata")
            tienda.fijar_stock(original, 0)
            assert asistente.procesar("Confirmo la compra")["estado"] == "sin_opciones"
            assert tienda.carrito(asistente.sesion) == {original: 1} and not tienda.pedidos(asistente.sesion)
        finally:
            tienda.cerrar()
    else:
        raise AssertionError("Faltan las fichas de desarrollo de la regresión de sustituciones.")
    carpeta = Path("verificacion_datos_reales")
    carpeta.mkdir(exist_ok=True)
    resultados.to_csv(carpeta / "bm25_por_consulta.csv", index=False, encoding="utf-8-sig")
    posiciones.to_csv(carpeta / "bm25_top10.csv", index=False, encoding="utf-8-sig")
    ejemplos = analizar_ranking(resultados, posiciones, datos.test, indice.fichas, carpeta)
    escenarios.to_csv(carpeta / "escenarios_texto.csv", index=False, encoding="utf-8-sig")
    resumen = {
        "alcance": "Control local con ESCI oficial, BM25 y reglas por texto",
        "datos": datos.resumen,
        "metricas_bm25": {
            "n_consultas": len(resultados), "ndcg10_media": float(resultados.ndcg10.mean()),
            "ndcg10_mediana": float(resultados.ndcg10.median()), "ndcg10_sd": float(resultados.ndcg10.std()),
            "exacto_en_posicion_uno": int((posiciones.posicion.eq(1) & posiciones.etiqueta.eq("E")).sum()),
        },
        "casos": {
            "correctos": int(escenarios.correcto.sum()), "total": len(escenarios),
            "incumplimientos": int(escenarios.incumplimientos.sum()), "voz_ejecutados": 0,
            "productos": {p: indice.fichas[p]["titulo"] for p in (a, b, c)},
        },
        "modelos_neuronales_ejecutados": False,
        "regresiones": {"complemento_esci_rechazado_ante_falta_de_stock": True},
        "diagnosticos": ejemplos.tipo.value_counts().to_dict(),
    }
    (carpeta / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esci", action="store_true", help="Ejecuta BM25 y compra por texto con el catálogo real.")
    parser.add_argument("--origen", type=Path, help="Carpeta adicional donde buscar los Parquet oficiales.")
    args = parser.parse_args()
    salida = comprobaciones_esci(args.origen) if args.esci else comprobaciones()
    print(json.dumps(salida, ensure_ascii=False, indent=2))
