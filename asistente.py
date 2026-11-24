"""Control del agente, memoria de sesión y respuestas a partir de las herramientas."""

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import re
from time import perf_counter

from esci import normalizar, tokens
from tienda import Restriccion, soles


NUMEROS = dict(zip("cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce".split(), range(13)))
NUMEROS.update({"un": 1, "una": 1, "trece": 13, "catorce": 14, "quince": 15,
                "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20})
NUMEROS.update(dict(zip(
    "veintiuno veintidos veintitres veinticuatro veinticinco veintiseis veintisiete veintiocho veintinueve".split(),
    range(21, 30),
)))
NUMEROS["veintiun"] = 21
DECENAS = dict(zip("treinta cuarenta cincuenta sesenta setenta ochenta noventa".split(), range(30, 100, 10)))
CENTENAS = dict(zip(
    "ciento doscientos trescientos cuatrocientos quinientos seiscientos setecientos ochocientos novecientos".split(),
    range(100, 1000, 100),
))
CENTENAS["cien"] = 100
VALORES_NUMERICOS = {**NUMEROS, **DECENAS, **CENTENAS}
BAJO_CIEN = rf"(?:(?:{'|'.join(DECENAS)})(?: y (?:{'|'.join(NUMEROS)}))?|{'|'.join(NUMEROS)})"
BAJO_MIL = rf"(?:(?:{'|'.join(CENTENAS)})(?: {BAJO_CIEN})?|{BAJO_CIEN})"
NUMERO_HABLADO = rf"(?:(?:{BAJO_MIL} )?mil(?: {BAJO_MIL})?|{BAJO_MIL})"
MONTO = rf"(?:-?(?:\d{{1,3}}(?:\.\d{{3}})+|\d+)(?:[,.]\d+)?|{NUMERO_HABLADO})"
UNIDADES = r"(?:unidad(?:es)?|par(?:es)?)"
PREFIJO_PRESUPUESTO = (
    r"(?:(?:presupuesto|tope|limite)(?:\s+(?:actual|nuevo|maximo))?(?:\s+(?:es|son))?"
    r"(?:\s+(?:de|hasta))?|(?:puedo|podemos|quiero|puedes)\s+gastar"
    r"(?:\s+(?:como\s+maximo|hasta))?|no\s+puedo\s+pasar\s+de)"
)
MENCION_PRESUPUESTO = re.compile(rf"\b{PREFIJO_PRESUPUESTO}\b")
CENTIMOS = rf"(?:\s+(?:con|y)\s+(?P<centimos>\d+|{NUMERO_HABLADO})\s+centimos?)?"
PATRONES_PRESUPUESTO = tuple(re.compile(p) for p in (
    rf"\b{PREFIJO_PRESUPUESTO}\s+(?:s/\.?\s*)?(?P<importe>{MONTO})(?:\s+sol(?:es)?{CENTIMOS})?\b",
    rf"\b(?:hasta|no mas de|maximo de|tengo|dispongo de)\s+"
    rf"(?:s/\.?\s*(?P<antes>{MONTO})|(?P<importe>{MONTO})\s+sol(?:es)?{CENTIMOS})\b",
    rf"^(?:s/\.?\s*(?P<antes>{MONTO})|(?P<importe>{MONTO})\s+sol(?:es)?{CENTIMOS})\s*[.!]?" + "$",
    rf"\bhasta\s+(?P<importe>{MONTO})\s*[.!]?" + "$",
))
ORDINALES = {"primera": 1, "primero": 1, "segunda": 2, "segundo": 2, "tercera": 3, "tercero": 3}
ACCIONES = {
    "buscar", "comparar", "agregar", "sumar", "cambiar", "retirar", "carrito", "cotizar",
    "confirmar", "cancelar", "preferencias", "atributo", "aclarar",
}
PALABRAS_FUNCIONALES = {"a", "al", "con", "de", "del", "el", "en", "la", "las", "los", "para", "por", "un", "una", "y"}
ACCESORIOS = {"adaptador", "bateria", "cargador", "estuche", "funda", "protector", "recambio", "repuesto", "soporte"}
CONFIRMA = re.compile(r"(?:si[,]?\s*)?(?:confirmo(?:\s+(?:la compra|el pedido|la cotizacion))?|acepto(?:\s+la cotizacion)?)")


def es_confirmacion(texto):
    texto = normalizar(texto).strip()
    if "?" in texto or "¿" in texto:
        return False
    texto = texto.strip(" .!¡")
    return texto == "si" or bool(CONFIRMA.fullmatch(texto))


def numero(texto):
    n = normalizar(texto).strip()
    if re.fullmatch(r"-?\d+", n):
        return int(n)
    if not re.fullmatch(NUMERO_HABLADO, n):
        return None
    acumulado, bloque = 0, 0
    for palabra in n.split():
        if palabra == "mil":
            acumulado = 1000 * (bloque or 1)
            bloque = 0
        elif palabra != "y":
            bloque += VALORES_NUMERICOS[palabra]
    return acumulado + bloque


def dinero(valor):
    try:
        importe = str(valor).strip()
        if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?", importe):
            importe = importe.replace(".", "")
        d = Decimal(importe.replace(",", "."))
        if d.is_finite() and 0 < d <= 1_000_000 and d * 100 == (d * 100).to_integral_value():
            return int(d * 100)
    except InvalidOperation:
        pass
    raise Restriccion("presupuesto", "Indica un presupuesto positivo en soles, con hasta dos decimales.")


def presupuesto_explicito(texto):
    texto = normalizar(texto).strip()
    for patron in PATRONES_PRESUPUESTO:
        match = patron.search(texto)
        if match:
            grupos = match.groupdict()
            importe = grupos.get("importe") or grupos["antes"]
            valor = numero(importe)
            fraccion = grupos.get("centimos")
            if fraccion is not None:
                partes = numero(fraccion)
                if valor is None or partes is None or not 0 <= partes <= 99:
                    return ""
                return f"{valor}.{partes:02d}"
            return importe if valor is None else valor
    return None


@dataclass
class Plan:
    accion: str = "aclarar"
    consulta: str = ""
    opcion: int | None = None
    product_id: str = ""
    cantidad: int | None = None
    presupuesto_soles: str | int | float | None = None
    marca: str = ""
    color: str = ""
    menor_precio: bool = False
    atributo: str = ""


def reglas(texto):
    """Reconoce expresiones directas y permite verificar el controlador por separado."""
    if es_confirmacion(texto):
        return Plan("confirmar")
    frase = normalizar(texto).strip(" .!¡¿?")
    if frase in {"carrito", "ver carrito", "muestra el carrito", "que tengo en el carrito"}:
        return Plan("carrito")
    if frase in {"cancelar", "cancela la compra", "cancela el pedido", "cancela todo"}:
        return Plan("cancelar")
    if frase in {"cotiza", "cotiza el carrito", "dame la cotizacion", "cuanto cuesta el carrito"}:
        return Plan("cotizar")
    if frase in {"compara", "compara las opciones", "compara estas opciones"}:
        return Plan("comparar")
    presupuesto = presupuesto_explicito(frase)
    if presupuesto is None and MENCION_PRESUPUESTO.search(frase):
        return Plan("aclarar", atributo="presupuesto")
    if presupuesto is not None and re.match(rf"^(?:mi\s+(?:(?:nuevo|actual)\s+)?)?{PREFIJO_PRESUPUESTO}\b", frase):
        return Plan("preferencias", presupuesto_soles=presupuesto)
    cantidad_lex = rf"-?\d+|{NUMERO_HABLADO}"
    opcion = re.search(r"\b(?:opcion|numero)\s+(" + cantidad_lex + r")\b", frase)
    indice = numero(opcion[1]) if opcion else next((n for palabra, n in ORDINALES.items()
                                                   if re.search(r"\b" + palabra + r"\b", frase)), None)
    atributo = re.match(r"(?:de que|que|dime el)\s+(color|marca|precio|stock)\b", frase)
    if atributo:
        return Plan("atributo", opcion=indice, atributo=atributo[1])
    preferencia = re.match(r"(?:solo|prefiero|cambia a)\s+(color|marca)\s+(.+)", frase)
    if preferencia:
        return Plan("preferencias", **{preferencia[1]: preferencia[2]})
    if frase in {"sin preferencia de color", "sin preferencia de marca"}:
        return Plan("preferencias", **{frase.split()[-1]: "cualquiera"})
    unidades = re.search(rf"(?:^|\s)({cantidad_lex})\s+{UNIDADES}\b", frase)
    incremento = re.search(rf"(?:^|\s)({cantidad_lex})\s+(?:{UNIDADES}\s+)?mas\b", frase)
    otra_unidad = re.search(rf"\botr[oa]\s+{UNIDADES}\b", frase)
    if re.match(r"(?:suma|anade|agrega)\b", frase) and (frase.startswith("suma ") or incremento or otra_unidad):
        numero_expresado = incremento or unidades
        cantidad = numero(numero_expresado[1]) if numero_expresado else 1 if otra_unidad else None
        return Plan("sumar", opcion=indice, cantidad=cantidad)
    if re.match(r"(?:quita|retira|elimina)\b", frase):
        return Plan("retirar", opcion=indice)
    cambio = re.match(r"(?:mejor(?:\s+solo)?|cambia(?:lo)?\s+a|corrijo(?:\s*,)?(?:\s+a)?)\s+(" + cantidad_lex + r")\b", frase)
    if cambio:
        return Plan("cambiar", opcion=indice, cantidad=numero(cambio[1]))
    menor = bool(re.search(r"\b(?:mas barato|mas barata|menor precio)\b", frase))
    if menor and not frase.startswith(("busca", "encuentra", "muestra")):
        return Plan("agregar", menor_precio=True, cantidad=numero(unidades[1]) if unidades else None)
    if (indice is not None and (frase.startswith(("elijo", "agrega", "selecciona", "quiero la", "me quedo con", "la opcion"))
                                or frase in ORDINALES)):
        return Plan("agregar", opcion=indice, cantidad=numero(unidades[1]) if unidades else None)
    buscar = re.match(r"(?:busca|buscame|encuentra|muestra|quiero|necesito)\s+(.+)", frase)
    if buscar:
        pedido = buscar[1]
        if presupuesto is not None:
            pedido = re.split(
                rf"(?:\s+|[,.;]\s*)(?:con\s+(?:un\s+)?(?:presupuesto|tope|limite)"
                rf"|por\s+hasta|hasta|por\s+maximo|no\s+mas\s+de|{PREFIJO_PRESUPUESTO})\b",
                pedido, maxsplit=1,
            )[0].strip(",. ")
        pedido = re.sub(rf"^hasta (?=(?:{cantidad_lex}) {UNIDADES}\b)", "", pedido)
        inicial = re.match(rf"({cantidad_lex})\s+(?:{UNIDADES}\s+de\s+)?(.+)", pedido)
        cantidad = numero(inicial[1]) if inicial else None
        consulta = inicial[2] if inicial else pedido
        if consulta in NUMEROS or not set(tokens(consulta)) - {"algo", "comprar", "producto", "uno", "un", "una", "cosa"}:
            return Plan()
        return Plan("buscar", consulta=consulta, cantidad=cantidad, presupuesto_soles=presupuesto)
    if presupuesto is not None:
        return Plan("preferencias", presupuesto_soles=presupuesto)
    return Plan()


@dataclass
class Memoria:
    presupuesto: int | None = None
    presupuesto_pendiente: bool = False
    preferencias: dict = field(default_factory=dict)
    consulta: str = ""
    candidatos: list = field(default_factory=list)
    cantidad_deseada: int = 1
    ultimo_producto: str | None = None
    consultas_por_producto: dict = field(default_factory=dict)
    cotizacion: int | None = None
    ultima_confirmada: int | None = None
    reemplazo: str | None = None


class Asistente:
    def __init__(self, indice, tienda, planificador=None, sesion="laboratorio"):
        self.indice = indice
        self.tienda = tienda
        self.planificador = planificador
        self.sesion = sesion
        self.memoria = Memoria()
        self.recuperacion_s = 0.0
        self.recomendacion_s = 0.0
        self.estado_anterior = None

    def contexto(self):
        m = self.memoria
        return {
            "consulta": m.consulta, "presupuesto_soles": None if m.presupuesto is None else m.presupuesto / 100,
            "preferencias_requeridas": m.preferencias, "pendiente_confirmacion": m.cotizacion is not None,
            "presupuesto_pendiente": m.presupuesto_pendiente,
            "cantidad_busqueda": m.cantidad_deseada, "ultimo_producto": m.ultimo_producto,
            "opciones": [{"opcion": k, "product_id": p, "titulo": self.indice.fichas[p]["titulo"]}
                         for k, p in enumerate(m.candidatos, 1)],
            "carrito": self.tienda.carrito(self.sesion),
        }

    def cumple(self, pid):
        ficha = self.indice.fichas[pid]
        for campo, pedido in self.memoria.preferencias.items():
            if not ficha[campo] or not set(tokens(pedido)).issubset(tokens(ficha[campo])):
                return False
        return True

    def _compatible(self, consulta, original, candidato):
        requeridos = set(tokens(consulta)) - PALABRAS_FUNCIONALES
        ficha_original = self.indice.fichas[original]
        ficha_candidata = self.indice.fichas[candidato]
        origen = set(tokens(" ".join(ficha_original[k] for k in ("titulo", "marca", "color"))))
        destino = set(tokens(" ".join(ficha_candidata[k] for k in ("titulo", "marca", "color"))))
        return bool(requeridos) and requeridos <= origen & destino and not (destino & ACCESORIOS) - origen

    def _vigente(self):
        self.memoria.cotizacion = None
        self.memoria.ultima_confirmada = None

    def _actualizar_condiciones(self, plan):
        m = self.memoria
        if plan.presupuesto_soles is not None:
            self._vigente()
            m.presupuesto_pendiente = True
            m.presupuesto = dinero(plan.presupuesto_soles)
            m.presupuesto_pendiente = False
        for campo in ("marca", "color"):
            valor = getattr(plan, campo)
            if valor:
                if normalizar(valor) == "cualquiera":
                    m.preferencias.pop(campo, None)
                else:
                    m.preferencias[campo] = valor
                self._vigente()

    def _opciones(self, consulta, cantidad, excluido=None):
        m = self.memoria
        inv = self.tienda.inventario()
        carrito = self.tienda.carrito(self.sesion)
        if excluido is not None:
            carrito.pop(excluido, None)
        total_carrito = sum(inv[p]["precio"] * n for p, n in carrito.items())
        posibles = []
        for pid, item in inv.items():
            if pid == excluido or not self.cumple(pid):
                continue
            existente = carrito.get(pid, 0)
            unidades = cantidad if excluido is None else cantidad + existente
            costo = total_carrito + item["precio"] * (unidades - existente)
            if item["stock"] < unidades or (m.presupuesto is not None and costo > m.presupuesto):
                continue
            if excluido is None or self._compatible(consulta, excluido, pid):
                posibles.append(pid)
        t = perf_counter()
        encontrados = self.indice.buscar(consulta, posibles, limite=4)
        self.recuperacion_s += perf_counter() - t
        m.candidatos = encontrados
        return self._lista(encontrados, inv)

    def _lista(self, opciones, inv=None):
        inv = inv if inv is not None else self.tienda.inventario()
        return [{
            "opcion": k, "product_id": p, "titulo": self.indice.fichas[p]["titulo"],
            "marca": self.indice.fichas[p]["marca"] or "desconocida",
            "color": self.indice.fichas[p]["color"] or "desconocido", **inv[p],
        } for k, p in enumerate(opciones, 1)]

    @staticmethod
    def _decir_opciones(opciones):
        return " ".join(f"Opción {o['opcion']}. {titulo_breve(o['titulo'])}. "
                        f"{soles(o['precio'])} por unidad. Stock {o['stock']}." for o in opciones)

    def _comparar(self, opciones):
        barata = min(opciones, key=lambda o: (o["precio"], o["opcion"]))
        mas_stock = max(opciones, key=lambda o: (o["stock"], -o["opcion"]))
        texto = (f"La opción {barata['opcion']} tiene el menor precio, {soles(barata['precio'])}. "
                 f"La opción {mas_stock['opcion']} tiene {mas_stock['stock']} unidades y el mayor stock. ")
        marcador = getattr(self.planificador, "recomendar", None)
        recomendada = None
        if marcador is not None:
            t = perf_counter()
            recomendada = marcador(self.memoria.consulta, self.memoria.preferencias, opciones)
            self.recomendacion_s += perf_counter() - t
            if recomendada == barata["opcion"]:
                texto += f"Si priorizas el precio, sugiero la opción {recomendada}. "
            elif recomendada == mas_stock["opcion"]:
                texto += f"Si priorizas la disponibilidad, sugiero la opción {recomendada}. "
            elif recomendada in {o["opcion"] for o in opciones}:
                texto += f"Puedes revisar la opción {recomendada} entre las coincidencias recuperadas. "
        return texto + self._decir_opciones(opciones), recomendada

    def _elegir(self, plan, desde_carrito=False):
        m = self.memoria
        carrito = self.tienda.carrito(self.sesion)
        opciones = list(carrito) if desde_carrito and not m.candidatos else m.candidatos
        if plan.opcion is not None:
            if 1 <= plan.opcion <= len(opciones):
                return opciones[plan.opcion - 1]
            raise Restriccion("aclaracion", "Ese número no corresponde a una opción disponible.")
        if plan.product_id and plan.product_id in (set(opciones) | set(carrito)):
            return plan.product_id
        if plan.menor_precio and opciones:
            inv = self.tienda.inventario()
            return min(opciones, key=lambda p: (inv[p]["precio"], p))
        if desde_carrito and plan.opcion is None and not plan.product_id and len(carrito) == 1:
            return next(iter(carrito))
        if not desde_carrito and plan.opcion is None and not plan.product_id and len(opciones) == 1:
            return opciones[0]
        raise Restriccion("aclaracion", "¿A cuál producto te refieres? Indica el número de la opción.")

    def _cotizar(self):
        m = self.memoria
        oferta = self.tienda.cotizar(self.sesion, m.presupuesto, self.cumple)
        m.cotizacion = oferta["id"]
        m.ultima_confirmada = None
        lineas = " ".join(f"{x['cantidad']} de {titulo_breve(self.indice.fichas[x['producto']]['titulo'])}. "
                           f"Subtotal {soles(x['subtotal'])}." for x in oferta["lineas"])
        return "cotizacion", f"{lineas} Total {soles(oferta['total'])}. ¿Confirmas la compra?", {"cotizacion": oferta}

    def _ejecutar(self, plan):
        m = self.memoria
        accion = plan.accion
        if accion == "aclarar" and plan.atributo == "presupuesto":
            self._vigente()
            m.presupuesto_pendiente = True
            return "aclaracion", "¿Cuál es el presupuesto máximo en soles?", {}
        self._actualizar_condiciones(plan)
        if m.presupuesto_pendiente and accion not in {"cancelar", "carrito", "atributo", "retirar"}:
            return "aclaracion", "Indica el nuevo presupuesto máximo en soles para continuar.", {}
        if plan.cantidad is not None and (type(plan.cantidad) is not int or plan.cantidad <= 0):
            raise Restriccion("cantidad", "Indica una cantidad entera mayor que cero.")
        if accion in {"buscar", "comparar", "preferencias"}:
            if accion == "buscar":
                if not plan.consulta.strip():
                    raise Restriccion("aclaracion", "¿Qué producto quieres buscar?")
                m.consulta = plan.consulta
                m.cantidad_deseada = plan.cantidad or 1
                m.reemplazo = None
                self._vigente()
            if accion == "preferencias" and (plan.marca or plan.color):
                carrito = self.tienda.carrito(self.sesion)
                elegido = m.ultimo_producto
                if elegido in carrito and not self.cumple(elegido):
                    m.reemplazo = elegido
                    m.consulta = m.consultas_por_producto.get(elegido, m.consulta)
                    m.cantidad_deseada = carrito[elegido]
            if not m.consulta:
                return "aclaracion", "¿Qué producto quieres buscar con esas condiciones?", {}
            opciones = self._opciones(m.consulta, m.cantidad_deseada, m.reemplazo)
            if not opciones:
                return "sin_opciones", "No encontré opciones con existencias que cumplan las condiciones.", {}
            if accion == "comparar":
                texto, recomendada = self._comparar(opciones)
                return "opciones", texto, {"opciones": opciones, "recomendacion": recomendada}
            return "opciones", self._decir_opciones(opciones), {"opciones": opciones}
        if accion in {"agregar", "sumar", "cambiar", "retirar"}:
            pid = self._elegir(plan, desde_carrito=accion != "agregar")
            carrito = self.tienda.carrito(self.sesion)
            if accion != "agregar" and pid not in carrito:
                raise Restriccion("aclaracion", "Indica una línea que esté en el carrito.")
            if accion in {"cambiar", "sumar"} and plan.cantidad is None:
                raise Restriccion("aclaracion", "Indica la cantidad de unidades.")
            cantidad = 0 if accion == "retirar" else plan.cantidad or m.cantidad_deseada
            reemplazo = m.reemplazo if accion == "agregar" else None
            consulta = m.consulta if accion == "agregar" else m.consultas_por_producto.get(pid, m.consulta)
            if reemplazo:
                consulta = m.consultas_por_producto.get(reemplazo, consulta)
                if pid not in m.candidatos or not self._compatible(consulta, reemplazo, pid):
                    raise Restriccion("aclaracion", "Elige una alternativa compatible de la lista.")
                if plan.cantidad is None:
                    cantidad = carrito[reemplazo]
            if reemplazo or accion == "sumar":
                cantidad += carrito.get(pid, 0)
            cambios = {pid: cantidad}
            if reemplazo:
                cambios[reemplazo] = 0
            nuevo = self.tienda.fijar_carrito(self.sesion, cambios, m.presupuesto, self.cumple)
            self._vigente()
            m.ultimo_producto = pid if cantidad else next(iter(nuevo)) if len(nuevo) == 1 else None
            m.reemplazo = None
            if cantidad and consulta:
                m.cantidad_deseada = cantidad
                m.consultas_por_producto[pid] = consulta
            for retirado in cambios.keys() - nuevo.keys():
                m.consultas_por_producto.pop(retirado, None)
            if nuevo and not m.presupuesto_pendiente:
                return self._cotizar()
            return "carrito", "Retiré el producto del carrito." if nuevo else "El carrito está vacío.", {"carrito": nuevo}
        if accion == "carrito":
            carrito = self.tienda.carrito(self.sesion)
            texto = " ".join(f"{n} de {titulo_breve(self.indice.fichas[p]['titulo'])}." for p, n in carrito.items())
            return "carrito", texto or "El carrito está vacío.", {"carrito": carrito}
        if accion == "cotizar":
            return self._cotizar()
        if accion == "cancelar":
            self.tienda.vaciar(self.sesion)
            self.memoria = Memoria()
            return "cancelado", "Cancelé la compra pendiente y vacié el carrito.", {}
        if accion == "confirmar":
            qid = m.cotizacion or m.ultima_confirmada
            if qid is None:
                raise Restriccion("aclaracion", "Primero prepara una cotización para poder confirmar.")
            try:
                pedido = self.tienda.confirmar(self.sesion, qid, m.presupuesto, self.cumple)
            except Restriccion as exc:
                self._vigente()
                if exc.motivo != "stock":
                    raise
                m.reemplazo = exc.producto
                consulta = m.consultas_por_producto.get(exc.producto, m.consulta)
                m.consulta = consulta
                m.cantidad_deseada = self.tienda.carrito(self.sesion).get(exc.producto, 1)
                opciones = self._opciones(consulta, m.cantidad_deseada, excluido=exc.producto) if consulta else []
                if not opciones:
                    return "sin_opciones", f"{exc} No encontré una alternativa con stock.", {"motivo": exc.motivo}
                return "alternativa", f"{exc} Puedes elegir otra opción. {self._decir_opciones(opciones)}", {
                    "motivo": exc.motivo, "opciones": opciones, "reemplazo": exc.producto,
                }
            m.cotizacion = None
            m.ultima_confirmada = qid
            if pedido["repetido"]:
                return "pedido", f"Tu pedido {pedido['id']} ya está confirmado.", {"pedido": pedido}
            return "pedido", f"Registré el pedido {pedido['id']} por {soles(pedido['total'])}.", {"pedido": pedido}
        if accion == "atributo":
            pid = self._elegir(plan, desde_carrito=True)
            ficha = self._lista([pid])[0]
            valor = ficha.get(plan.atributo)
            if plan.atributo == "precio":
                valor = soles(valor)
            return "informacion", f"{titulo_breve(ficha['titulo'])}. {plan.atributo.capitalize()} {valor}.", {"ficha": ficha}
        return "aclaracion", "¿Qué producto quieres buscar o qué dato de tu compra quieres cambiar?", {}

    def procesar(self, texto):
        inicio = perf_counter()
        if self.planificador is not None and tokens(texto):
            plan = self.planificador(texto, self.contexto())
            origen_plan = getattr(self.planificador, "origen_plan", "reglas")
        else:
            plan = reglas(texto)
            origen_plan = "reglas"
        importe = presupuesto_explicito(texto)
        if importe is None and MENCION_PRESUPUESTO.search(normalizar(texto)):
            plan = Plan("aclarar", atributo="presupuesto")
            origen_plan = "reglas"
        else:
            plan.presupuesto_soles = importe
        if plan.accion not in ACCIONES or (plan.accion == "confirmar" and not es_confirmacion(texto)):
            plan = Plan()
        if plan.accion == "confirmar" and normalizar(texto).strip(" .!¡¿?") == "si" and self.estado_anterior != "cotizacion":
            plan = Plan()
        if plan.accion == "atributo" and plan.atributo not in {"marca", "color", "precio", "stock"}:
            plan = Plan()
        plan_s = perf_counter() - inicio
        self.recuperacion_s = 0.0
        self.recomendacion_s = 0.0
        try:
            estado, respuesta, evidencias = self._ejecutar(plan)
        except Restriccion as exc:
            estado = "aclaracion" if exc.motivo == "aclaracion" else "restriccion"
            respuesta, evidencias = str(exc), {"motivo": exc.motivo, "producto": exc.producto}
        self.estado_anterior = estado
        total = perf_counter() - inicio
        return {
            "texto": texto, "plan": asdict(plan), "origen_plan": origen_plan,
            "estado": estado, "respuesta": respuesta, "evidencia": evidencias,
            "memoria": asdict(self.memoria), "tiempos_s": {"planificacion": plan_s, "recuperacion": self.recuperacion_s,
                                                         "recomendacion_llm": self.recomendacion_s,
                                                         "tienda_y_respuesta": max(0, total - plan_s - self.recuperacion_s - self.recomendacion_s),
                                                         "agente_total": total},
        }


def titulo_breve(titulo):
    if len(titulo) <= 88:
        return titulo
    return titulo[:88].rsplit(" ", 1)[0]
