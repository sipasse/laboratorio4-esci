"""Preparación de ESCI, índice de productos y evaluación por consulta."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from html import unescape
from pathlib import Path
import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


SEED = 42
COMMIT_ESCI = "7916cdf6ab75a462e77f20ab40428a10923998d5"
BASE_GITHUB = (
    f"https://media.githubusercontent.com/media/amazon-science/esci-data/{COMMIT_ESCI}"
    "/shopping_queries_dataset"
)
BASE_COPIA = (
    "https://huggingface.co/datasets/khanmu2003/amazon-shopping-queries-dataset/resolve/main"
)
ARCHIVOS = {
    "examples": (51_286_808, "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"),
    "products": (1_108_857_465, "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265"),
}
GANANCIA = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
CAMPOS = ["titulo", "descripcion", "vinetas", "marca", "color"]
METODOS = ("BM25", "E5 sin BM25", "Híbrido BM25 + E5")


@dataclass
class Datos:
    catalogo: pd.DataFrame
    desarrollo: pd.DataFrame
    test: pd.DataFrame
    resumen: dict


def hash_archivo(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _descargar(url, destino):
    curl = shutil.which("curl.exe") if os.name == "nt" else None
    if curl:
        subprocess.run(
            [curl, "--fail", "--location", "--silent", "--show-error", "--retry", "2",
             "--connect-timeout", "30", "--max-time", "1800", "--output", str(destino), url],
            check=True,
        )
    else:
        solicitud = Request(url, headers={"User-Agent": "Laboratorio4-ESCI"})
        with urlopen(solicitud, timeout=90) as origen, destino.open("wb") as salida:
            shutil.copyfileobj(origen, salida, length=4 * 1024 * 1024)


def obtener_archivos(directorio, descargar=True, origen_local=None):
    """Acepta únicamente los dos Parquet de la versión publicada."""
    directorio = Path(directorio)
    alternativa = Path(origen_local) if origen_local is not None else None
    if descargar:
        directorio.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tipo, (size, expected_hash) in ARCHIVOS.items():
        name = f"shopping_queries_dataset_{tipo}.parquet"
        path = directorio / name
        if not path.is_file() and alternativa is not None and (alternativa / name).is_file():
            path = alternativa / name
        if path.is_file():
            if path.stat().st_size < 512 and path.read_bytes().startswith(b"version https://git-lfs"):
                raise ValueError(f"{path} es una referencia Git LFS. Coloque el Parquet completo o use otra carpeta.")
            if path.stat().st_size != size or hash_archivo(path) != expected_hash:
                raise ValueError(f"El tamaño o SHA-256 de {path} difiere de la versión publicada.")
        elif descargar:
            partial = path.with_suffix(".parquet.part")
            errors = []
            for base in (BASE_GITHUB, BASE_COPIA):
                try:
                    print(f"Descargando {name} ({size / 1e6:.1f} MB)", flush=True)
                    _descargar(f"{base}/{name}", partial)
                    if partial.stat().st_size != size or hash_archivo(partial) != expected_hash:
                        raise ValueError("El archivo descargado no coincide con el SHA-256 publicado.")
                    os.replace(partial, path)
                    break
                except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                    errors.append(f"{base.split('/')[2]}: {exc}")
                    partial.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"No se pudo obtener {name}. " + " | ".join(errors))
        else:
            raise FileNotFoundError(path)
        paths[tipo] = path
    return paths


def limpio(value):
    if not isinstance(value, str):
        return ""
    return " ".join(re.sub(r"<[^>]*>", " ", unescape(value)).split())


def normalizar(value):
    folded = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(c for c in folded if not unicodedata.combining(c))


def tokens(text):
    return re.findall(r"[a-z0-9]+", normalizar(text))


def muestra_por_profundidad(pares, n, semilla):
    """Asigna plazas por proporción y selecciona identificadores con un orden reproducible."""
    profundidad = pares.groupby("query_id", sort=True).size()
    if not 1 <= n <= len(profundidad):
        raise ValueError(f"Se pidieron {n} consultas de una población de {len(profundidad)}.")
    estratos = pd.cut(profundidad, [0, 16, 24, 32, 40, np.inf], labels=False)
    grupos = {int(e): estratos[estratos.eq(e)].index.tolist() for e in sorted(estratos.unique())}
    cuotas = {e: len(ids) * n / len(profundidad) for e, ids in grupos.items()}
    plazas = {e: math.floor(cuota) for e, cuota in cuotas.items()}
    restantes = n - sum(plazas.values())
    for e in sorted(cuotas, key=lambda e: (-(cuotas[e] - plazas[e]), e))[:restantes]:
        plazas[e] += 1
    elegidos = []
    for e, ids in grupos.items():
        ids.sort(key=lambda q: (sha256(f"{semilla}:{q}".encode()).digest(), q))
        elegidos.extend(ids[:plazas[e]])
    return pares.loc[pares.query_id.isin(elegidos)].sort_values(["query_id", "product_id"]).reset_index(drop=True)


def leer_fichas(path, ids):
    columnas = ["product_id", "product_locale", "product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]
    tablas = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=32_768, columns=columnas):
        es = batch.filter(pc.equal(batch.column("product_locale"), "es"))
        if es.num_rows:
            tablas.append(es.to_pandas())
    if not tablas:
        raise ValueError("El Parquet de productos no contiene fichas del mercado es.")
    fichas = pd.concat(tablas, ignore_index=True)
    if fichas.product_id.isna().any() or fichas.product_id.duplicated().any():
        raise ValueError("Las claves de producto ES deben ser únicas y no nulas.")
    disponibles = set(fichas.product_id)
    faltantes = ids - disponibles
    if faltantes:
        raise ValueError(f"Faltan fichas ES para {len(faltantes)} productos anotados.")
    necesarios = max(0, 5000 - len(ids))
    if necesarios:
        extras = sorted(disponibles - ids, key=lambda p: (sha256(f"{SEED}:{p}".encode()).digest(), p))
        ids = ids.union(extras[:necesarios])
    if len(ids) < 5000:
        raise ValueError("El origen no permite elaborar un catálogo de 5.000 productos.")
    fichas = fichas.loc[fichas.product_id.isin(ids)].copy()
    fichas.columns = ["product_id", "locale", *CAMPOS]
    for campo in CAMPOS:
        fichas[campo] = fichas[campo].map(limpio)
    if fichas.titulo.eq("").any():
        raise ValueError("Se encontraron fichas seleccionadas sin título.")
    return fichas.sort_values("product_id").reset_index(drop=True)


def preparar_datos(directorio, n_test=200, n_dev=60, semilla=SEED, descargar=True, origen_local=None):
    if n_test < 200:
        raise ValueError("El conjunto de evaluación exige al menos 200 consultas.")
    directorio = Path(directorio)
    salida = directorio / f"esci_es_{n_test}_{n_dev}_{semilla}"
    manifiesto = salida / "resumen.json"
    archivos = [salida / f"{name}.parquet" for name in ("catalogo", "desarrollo", "test")]
    if manifiesto.is_file() and all(path.is_file() for path in archivos):
        resumen = json.loads(manifiesto.read_text(encoding="utf-8"))
        huellas = resumen.get("sha256_preparados")
        if huellas is not None:
            origen_esperado = {kind: digest for kind, (_, digest) in ARCHIVOS.items()}
            if (resumen.get("version") != COMMIT_ESCI
                    or resumen.get("sha256_origen") != origen_esperado
                    or huellas != {path.name: hash_archivo(path) for path in archivos}):
                raise ValueError(f"El subconjunto preparado en {salida} no coincide con su manifiesto.")
            return Datos(*(pd.read_parquet(path) for path in archivos), resumen)

    raw = obtener_archivos(directorio / "origen", descargar, origen_local)
    ejemplos = pq.read_table(
        raw["examples"],
        columns=["query_id", "query", "product_id", "esci_label", "split"],
        filters=[("product_locale", "=", "es"), ("small_version", "=", 1)],
    ).to_pandas()
    if (ejemplos.isna().any().any() or ejemplos.duplicated(["query_id", "product_id"]).any()
            or not ejemplos.esci_label.isin(GANANCIA).all()
            or set(ejemplos.split) != {"train", "test"}
            or ejemplos.groupby("query_id")["query"].nunique().gt(1).any()):
        raise ValueError("Se encontraron claves, anotaciones o consultas inconsistentes.")
    train = ejemplos.loc[ejemplos.split.eq("train")]
    test = ejemplos.loc[ejemplos.split.eq("test")]
    if set(train.query_id).intersection(test.query_id):
        raise ValueError("Hay consultas compartidas entre train y test.")
    desarrollo = muestra_por_profundidad(train, n_dev, semilla + 1)
    prueba = muestra_por_profundidad(test, n_test, semilla)
    catalogo = leer_fichas(raw["products"], set(desarrollo.product_id).union(prueba.product_id))
    resumen = {
        "version": COMMIT_ESCI, "locale": "es", "small_version": 1, "semilla": semilla,
        "estratos_candidatos": ["1–16", "17–24", "25–32", "33–40", "más de 40"],
        "universo_consultas_train": int(train.query_id.nunique()),
        "universo_consultas_test": int(test.query_id.nunique()),
        "consultas_desarrollo": int(desarrollo.query_id.nunique()),
        "consultas_test": int(prueba.query_id.nunique()),
        "pares_desarrollo": len(desarrollo), "pares_test": len(prueba),
        "productos_distintos_test": int(prueba.product_id.nunique()),
        "catalogo": len(catalogo), "pares_test_sin_ficha": 0,
        "etiquetas_test": {label: int(prueba.esci_label.eq(label).sum()) for label in GANANCIA},
        "texto_disponible": {campo: int(catalogo[campo].ne("").sum()) for campo in CAMPOS},
    }
    salida.mkdir(parents=True, exist_ok=True)
    for tabla, path in zip((catalogo, desarrollo, prueba), archivos):
        tabla.to_parquet(path, index=False)
    resumen["sha256_origen"] = {kind: digest for kind, (_, digest) in ARCHIVOS.items()}
    resumen["sha256_preparados"] = {path.name: hash_archivo(path) for path in archivos}
    manifiesto.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return Datos(catalogo, desarrollo, prueba, resumen)


def documento(ficha):
    return " ".join(
        str(x) for x in (ficha["titulo"], ficha["titulo"], ficha["marca"], ficha["color"],
                         ficha["vinetas"][:480], ficha["descripcion"][:480])
        if x
    )


class Indice:
    def __init__(self, catalogo, encoder=None, directorio_cache=None):
        self.fichas = {r["product_id"]: r for r in catalogo.to_dict("records")}
        self.ids = np.array(sorted(self.fichas))
        self.posiciones = {p: n for n, p in enumerate(self.ids)}
        textos = [documento(self.fichas[p]) for p in self.ids]
        postings = defaultdict(list)
        self.longitudes = np.empty(len(textos), dtype=np.float32)
        for i, text in enumerate(textos):
            freq = Counter(tokens(text))
            self.longitudes[i] = sum(freq.values())
            for term, n in freq.items():
                postings[term].append((i, n))
        self.promedio = max(float(self.longitudes.mean()), 1)
        self.postings = {term: np.asarray(rows, dtype=np.int32) for term, rows in postings.items()}
        self.encoder = encoder
        self.vectores = None
        if encoder is not None:
            identidad = json.dumps([encoder.config, self.ids.tolist(), textos], ensure_ascii=False)
            tag = sha256(identidad.encode()).hexdigest()[:18]
            cache = None if directorio_cache is None else Path(directorio_cache) / f"e5_{tag}.npy"
            if cache is not None and cache.is_file():
                self.vectores = np.load(cache, allow_pickle=False)
            else:
                self.vectores = encoder.encode(textos, "passage")
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, self.vectores, allow_pickle=False)
            if self.vectores.shape != (len(self.ids), encoder.dimension):
                raise ValueError("El índice semántico no coincide con el catálogo.")

    def bm25(self, consulta):
        scores = np.zeros(len(self.ids), dtype=np.float32)
        for term in sorted(set(tokens(consulta))):
            rows = self.postings.get(term)
            if rows is None:
                continue
            ix, freq = rows.T
            idf = math.log1p((len(scores) - len(rows) + 0.5) / (len(rows) + 0.5))
            den = freq + 1.5 * (0.25 + 0.75 * self.longitudes[ix] / self.promedio)
            scores[ix] += idf * freq * 2.5 / den
        return scores

    def ordenar(self, ids, scores):
        return np.lexsort((ids, -scores))

    def puntuar(self, consulta, candidatos=None, vector=None, metodos=None):
        ids = self.ids if candidatos is None else np.asarray(candidatos)
        indices = np.asarray([self.posiciones[p] for p in ids], dtype=np.int32)
        lex = self.bm25(consulta)[indices]
        scores = {METODOS[0]: lex}
        if self.vectores is None or metodos == (METODOS[0],):
            return ids, scores
        if vector is None:
            vector = self.encoder.encode([consulta], "query")[0]
        sem = self.vectores[indices] @ vector
        fusion = np.zeros(len(ids), dtype=np.float32)
        posiciones = np.arange(1, len(ids) + 1, dtype=np.float32)
        for valores in (lex, sem):
            # Las coincidencias léxicas con score cero no aportan a la fusión.
            orden = self.ordenar(ids, valores)
            if valores is lex:
                orden = orden[lex[orden] > 0]
            fusion[orden] += 1.0 / (60 + posiciones[:len(orden)])
        scores.update({METODOS[1]: sem, METODOS[2]: fusion})
        return ids, scores

    def buscar(self, consulta, candidatos, limite=5):
        if not candidatos:
            return []
        ids, scores = self.puntuar(consulta, candidatos)
        valores = scores[METODOS[2]] if self.vectores is not None else scores[METODOS[0]]
        orden = self.ordenar(ids, valores)
        relevantes = scores[METODOS[0]] > 0
        if self.vectores is not None:
            # Este umbral pertenece a la interacción, no al cálculo del ranking anotado.
            relevantes |= scores[METODOS[1]] >= 0.82
        return [str(ids[i]) for i in orden[relevantes[orden]][:limite]]


def ndcg10(ganancias):
    g = np.asarray(ganancias, dtype=float)
    k = min(len(g), 10)
    descuento = np.log2(np.arange(2, k + 2))
    ideal = (np.sort(g)[::-1][:k] / descuento).sum()
    return float((g[:k] / descuento).sum() / ideal) if ideal else 0.0


def evaluar_recuperacion(indice, pares, solo_lexico=False):
    grupos = list(pares.groupby("query_id", sort=True))
    textos = [partes["query"].iloc[0] for _, partes in grupos]
    metodos = (METODOS[0],) if solo_lexico else METODOS
    if not solo_lexico and indice.encoder is None:
        raise ValueError("La comparación completa requiere el índice E5.")
    vectores = None if solo_lexico else indice.encoder.encode(textos, "query")
    resultados, rankings = [], []
    for n, (qid, candidatos) in enumerate(grupos):
        ids, scores = indice.puntuar(textos[n], candidatos.product_id.tolist(),
                                     None if vectores is None else vectores[n], metodos)
        etiquetas = candidatos.esci_label.to_numpy()
        gains = np.asarray([GANANCIA[label] for label in etiquetas])
        for metodo in metodos:
            orden = indice.ordenar(ids, scores[metodo])
            resultados.append({
                "query_id": int(qid), "consulta": textos[n], "metodo": metodo,
                "candidatos": len(ids), "ndcg10": ndcg10(gains[orden]),
            })
            rankings.extend({
                "query_id": int(qid), "metodo": metodo, "posicion": pos,
                "product_id": str(ids[i]), "titulo": indice.fichas[ids[i]]["titulo"],
                "etiqueta": etiquetas[i],
            } for pos, i in enumerate(orden[:10], 1))
    return pd.DataFrame(resultados), pd.DataFrame(rankings)
