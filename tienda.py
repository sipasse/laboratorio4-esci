"""Tienda reproducible. Los importes se guardan en céntimos de sol."""

from contextlib import contextmanager
import json
import random
import sqlite3


class Restriccion(Exception):
    def __init__(self, motivo, detalle, producto=None):
        super().__init__(detalle)
        self.motivo = motivo
        self.producto = producto


def soles(centimos):
    entero, decimal = divmod(int(centimos), 100)
    return f"S/ {entero}.{decimal:02d}"


class Tienda:
    def __init__(self, fichas, semilla=42, ruta=":memory:"):
        self.fichas = fichas
        self.semilla = semilla
        self.db = sqlite3.connect(ruta, isolation_level=None, timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS inventario (
                producto TEXT PRIMARY KEY, precio INTEGER NOT NULL CHECK(precio > 0),
                stock INTEGER NOT NULL CHECK(stock >= 0)
            );
            CREATE TABLE IF NOT EXISTS lineas (
                sesion TEXT NOT NULL, producto TEXT NOT NULL REFERENCES inventario(producto),
                cantidad INTEGER NOT NULL CHECK(cantidad > 0), PRIMARY KEY (sesion, producto)
            );
            CREATE TABLE IF NOT EXISTS cotizaciones (
                id INTEGER PRIMARY KEY, sesion TEXT NOT NULL, contenido TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pedidos (
                id INTEGER PRIMARY KEY, cotizacion INTEGER NOT NULL UNIQUE REFERENCES cotizaciones(id),
                sesion TEXT NOT NULL, contenido TEXT NOT NULL
            );
        """)
        if not self.db.execute("SELECT EXISTS(SELECT 1 FROM inventario)").fetchone()[0]:
            self.restaurar()
        elif {r[0] for r in self.db.execute("SELECT producto FROM inventario")} != set(self.fichas):
            raise ValueError("El inventario persistido pertenece a otro catálogo.")

    @contextmanager
    def transaccion(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def cerrar(self):
        self.db.close()

    def restaurar(self, overrides=None, stock_otros=None):
        overrides = overrides or {}
        if set(overrides) - set(self.fichas):
            raise ValueError("Los ajustes de inventario deben referirse a fichas del catálogo.")
        rng = random.Random(self.semilla)
        registros = []
        for pid in sorted(self.fichas):
            precio = rng.randint(1000, 16000)
            stock = rng.randint(0, 18)
            if stock_otros is not None:
                stock = stock_otros
            precio, stock = overrides.get(pid, (precio, stock))
            registros.append((pid, precio, stock))
        with self.transaccion():
            for tabla in ("pedidos", "cotizaciones", "lineas", "inventario"):
                self.db.execute(f"DELETE FROM {tabla}")
            self.db.executemany("INSERT INTO inventario VALUES (?, ?, ?)", registros)

    def inventario(self):
        return {
            r["producto"]: {"precio": r["precio"], "stock": r["stock"]}
            for r in self.db.execute("SELECT * FROM inventario ORDER BY producto")
        }

    def fijar_stock(self, producto, unidades):
        if type(unidades) is not int or unidades < 0:
            raise ValueError("El stock debe ser un entero no negativo.")
        with self.transaccion():
            cursor = self.db.execute("UPDATE inventario SET stock=? WHERE producto=?", (unidades, producto))
            if cursor.rowcount != 1:
                raise KeyError(producto)

    def carrito(self, sesion):
        return {
            r["producto"]: r["cantidad"]
            for r in self.db.execute("SELECT producto, cantidad FROM lineas WHERE sesion=? ORDER BY producto", (sesion,))
        }

    def pedidos(self, sesion):
        return [
            {"id": r["id"], "cotizacion": r["cotizacion"], **json.loads(r["contenido"])}
            for r in self.db.execute("SELECT * FROM pedidos WHERE sesion=? ORDER BY id", (sesion,))
        ]

    def _comprobar(self, carrito, presupuesto=None, cumplir=None):
        if not carrito:
            raise Restriccion("vacio", "El carrito está vacío.")
        if presupuesto is not None and (type(presupuesto) is not int or presupuesto <= 0):
            raise Restriccion("presupuesto", "El presupuesto debe ser positivo.")
        lineas, total = [], 0
        for pid, cantidad in sorted(carrito.items()):
            if type(cantidad) is not int or cantidad <= 0:
                raise Restriccion("cantidad", "La cantidad debe ser un número entero positivo.", pid)
            item = self.db.execute("SELECT * FROM inventario WHERE producto=?", (pid,)).fetchone()
            if item is None:
                raise Restriccion("producto", "Ese producto no aparece en el catálogo.", pid)
            if cumplir is not None and not cumplir(pid):
                raise Restriccion("preferencia", "Una ficha del carrito no cumple las preferencias confirmadas.", pid)
            if item["stock"] < cantidad:
                raise Restriccion(
                    "stock",
                    f"Para {self.fichas[pid]['titulo']} quedan {item['stock']} unidades y se solicitaron {cantidad}.",
                    pid,
                )
            subtotal = cantidad * item["precio"]
            total += subtotal
            lineas.append({"producto": pid, "cantidad": cantidad, "precio": item["precio"], "subtotal": subtotal})
        if presupuesto is not None and total > presupuesto:
            raise Restriccion("presupuesto", f"El total {soles(total)} supera el presupuesto {soles(presupuesto)}.")
        return {"lineas": lineas, "total": total, "presupuesto": presupuesto}

    def fijar_carrito(self, sesion, cambios, presupuesto=None, cumplir=None):
        with self.transaccion():
            actual = self.carrito(sesion)
            for pid, cantidad in cambios.items():
                if type(cantidad) is not int or cantidad < 0:
                    raise Restriccion("cantidad", "Indica una cantidad entera no negativa.", pid)
                if cantidad:
                    actual[pid] = cantidad
                else:
                    actual.pop(pid, None)
            if actual:
                self._comprobar(actual, presupuesto, cumplir)
            self.db.execute("DELETE FROM lineas WHERE sesion=?", (sesion,))
            self.db.executemany("INSERT INTO lineas VALUES (?, ?, ?)",
                                [(sesion, pid, unidades) for pid, unidades in actual.items()])
        return actual

    def vaciar(self, sesion):
        with self.transaccion():
            self.db.execute("DELETE FROM lineas WHERE sesion=?", (sesion,))

    def cotizar(self, sesion, presupuesto=None, cumplir=None):
        with self.transaccion():
            calculo = self._comprobar(self.carrito(sesion), presupuesto, cumplir)
            cursor = self.db.execute("INSERT INTO cotizaciones(sesion, contenido) VALUES(?, ?)",
                                     (sesion, json.dumps(calculo, ensure_ascii=False)))
            return {"id": cursor.lastrowid, **calculo}

    def confirmar(self, sesion, cotizacion, presupuesto=None, cumplir=None):
        with self.transaccion():
            existente = self.db.execute(
                "SELECT id, contenido FROM pedidos WHERE sesion=? AND cotizacion=?", (sesion, cotizacion)
            ).fetchone()
            if existente:
                return {"id": existente["id"], "cotizacion": cotizacion,
                        "repetido": True, **json.loads(existente["contenido"])}
            original = self.db.execute(
                "SELECT contenido FROM cotizaciones WHERE sesion=? AND id=?", (sesion, cotizacion)
            ).fetchone()
            if original is None:
                raise Restriccion("cotizacion", "La cotización no pertenece a esta sesión.")
            actual = self._comprobar(self.carrito(sesion), presupuesto, cumplir)
            if actual != json.loads(original["contenido"]):
                raise Restriccion("cotizacion", "La cotización cambió y necesita una nueva confirmación.")
            for linea in actual["lineas"]:
                cantidad, pid = linea["cantidad"], linea["producto"]
                cursor = self.db.execute(
                    "UPDATE inventario SET stock=stock-? WHERE producto=? AND stock>=?", (cantidad, pid, cantidad)
                )
                if cursor.rowcount != 1:
                    raise Restriccion("stock", "Cambió el stock mientras se registraba el pedido.", pid)
            cursor = self.db.execute("INSERT INTO pedidos(cotizacion, sesion, contenido) VALUES(?, ?, ?)",
                                     (cotizacion, sesion, json.dumps(actual, ensure_ascii=False)))
            self.db.execute("DELETE FROM lineas WHERE sesion=?", (sesion,))
            return {"id": cursor.lastrowid, "cotizacion": cotizacion, "repetido": False, **actual}

    def estado(self, sesion):
        return {"inventario": self.inventario(), "carrito": self.carrito(sesion), "pedidos": self.pedidos(sesion)}
