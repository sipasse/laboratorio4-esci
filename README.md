# Laboratorio 4 · Asistente de compras por voz

Asistente de compras en español con recuperación sobre el catálogo ESCI y una tienda reproducible. El modelo de lenguaje interpreta la conversación. Las herramientas consultan productos y controlan el carrito, el presupuesto, las cotizaciones y los pedidos.

## Ejecutar en Colab

1. Abra **Laboratorio4.ipynb** en Google Colab y seleccione un entorno con GPU. Ejecute las celdas de preparación y las secciones 1 a 4 en orden. El notebook incluye los módulos de Python.
2. Para usar el subconjunto preparado, cargue **Laboratorio4_ESCI.zip** en `/content` antes de la sección 2. La celda verifica las huellas del catálogo y las particiones. Si el ZIP no está disponible, el notebook prepara el subconjunto a partir de los Parquet de ESCI.
3. **Audios_sinteticos_Lab4_sesion.zip** es opcional. Puede cargarse en `/content` antes de la sección 4 para reutilizar la referencia y las intervenciones de prueba generadas con VoiceDesign.
4. En la sección 5, ejecute **Preparar la conversación**. Para usar una referencia propia o autorizada, ejecute **Configurar mi voz para las respuestas** y valide la transcripción de la toma. Después, ejecute **Conversación por micrófono** y pulse **Finalizar conversación** tras el último turno. La referencia sintética permite continuar directamente a la evaluación.
5. Ejecute **6 · Evaluación operativa y de voz** con la referencia activa. Las tablas y el ZIP de evidencias se generan en esa sesión. Use `descargar_evidencias()` para descargarlo y `mostrar_caso("01_compra_voz")` para revisar los turnos de una prueba.

La demostración mide el tiempo desde que termina la grabación hasta que el navegador puede reproducir la respuesta. También se puede ejecutar un turno con un archivo mediante `turno_wav(ruta)`.

## Verificar en VS Code

La comprobación por texto utiliza las mismas herramientas de compra que el asistente integrado. Requiere **numpy**, **pandas** y **pyarrow**.

~~~powershell
python verificar_local.py
~~~

El siguiente bloque extrae el subconjunto en `datos/` y repite el control local de BM25 y compras por texto.

~~~powershell
python -m zipfile -e Laboratorio4_ESCI.zip .
python verificar_local.py --esci
~~~

El catálogo contiene 6.915 fichas, 60 consultas de desarrollo y 200 de test. Los 5.470 pares anotados de test tienen cobertura completa. El control con el catálogo guarda sus resultados en `verificacion_datos_reales/`. La evaluación neuronal usa los modelos de Colab.

## Decisiones metodológicas

- **Datos.** Versión reducida de ESCI para el mercado español. Se seleccionan 200 consultas de prueba y 60 de desarrollo, estratificadas por cantidad de candidatos con semilla 42. El catálogo conserva las fichas anotadas de ambos grupos. Los juicios de desarrollo validan las fichas de los escenarios y las etiquetas de test se reservan para calcular el ranking. El agente consulta el catálogo, los inventarios y el carrito.
- **Recuperación.** BM25 como base léxica, E5 multilingüe como rama semántica y fusión de posiciones como método completo. La ablación retira BM25. Los métodos reciben los mismos pares y se aplica nDCG@10 a la lista anotada de cada consulta con ganancias E 1, S 0,1, C 0,01 e I 0. El título conserva peso prioritario y los otros campos se incorporan cuando contienen información. El umbral coseno de 0,82 controla qué opciones se presentan durante la interacción. El cálculo de ranking usa las listas completas.
- **Conversación.** Qwen2.5 de 1,5 mil millones de parámetros interpreta los turnos y utiliza en su contexto las opciones recuperadas y el estado de la sesión. Una capa de control valida el plan, conserva los importes explícitos y suspende la confirmación cuando un nuevo presupuesto requiere aclaración. Los reemplazos deben mantener los términos de la solicitud que se pueden verificar en las fichas y las cantidades de las otras líneas de la cesta. Este filtro es conservador ante sinónimos y atributos ausentes. SQLite vuelve a comprobar las condiciones y escribe pedido e inventario en una misma transacción.
- **Voz y memoria del sistema.** Faster‑Whisper small transcribe en español en CPU con cuantización int8. Qwen3‑TTS VoiceDesign genera la referencia y se libera antes de cargar el clonador Qwen3‑TTS Base de 0,6 mil millones de parámetros. El clonador usa bfloat16 en GPU con soporte nativo y float32 en la T4. E5, Qwen2.5 y VoiceDesign pueden usar float16 en la T4. El clonador usa muestreo con semilla y recibe el texto completo. Se acota la longitud de generación en función del número de palabras para limitar repeticiones observadas con la decodificación voraz. Se guardan versiones, dispositivo, carga inicial, tiempos de turno y pico de VRAM.
- **Pruebas.** Veinte escenarios predefinidos con restauración de la tienda y eventos controlados. Los seis casos por voz se repiten con el texto de referencia. Los errores de entidades se calculan contra cantidades, importes y productos anotados en los guiones. La evaluación distingue omisiones, adiciones y sustituciones. El caso de corrección puede enmascarar la cantidad en una entrada sintética y comprueba la transcripción observada y las cantidades de la cotización antes y después de corregirla. Se reportan los errores de entidades con y sin esta intervención. La tasa de error de palabras aproxima la inteligibilidad y el coseno entre representaciones de locutor aproxima la semejanza. La escucha permite revisar las limitaciones de esas medidas.

El reconocimiento conserva el decodificador de Faster-Whisper 1.2.1. En PyAV 19 y versiones posteriores, el código omite el argumento `metadata_errors`, retirado de esa API. [Incidencia de Faster-Whisper](https://github.com/SYSTRAN/faster-whisper/issues/1589).

## Archivos de resultados

| Evidencia | Contenido |
|---|---|
| configuracion.json | Versiones, modelos, semilla, hardware, carga inicial y configuración usada. |
| ranking_por_consulta.csv y resumen_recuperacion.json | nDCG@10 por consulta y comparación emparejada de los tres métodos. |
| top10_test.csv y aciertos_y_fallos.csv | Posiciones, etiquetas y evidencia para examinar aciertos y errores reales del ranking. |
| operacion/resumen_operativo.json y operacion/escenarios.csv | Casos correctos, incumplimientos y cobertura de escenarios por audio. |
| operacion/turnos_de_voz.csv y resumen_medidas_de_voz.csv | Latencias, resultado por modalidad, origen del plan, errores de entidades, WER de entrada y respuesta y similitud. |
| operacion/escenario/modo/traza.json | Transcripciones, operaciones, evidencias de voz y estados de la tienda. |

## Demostración para el video

| Tiempo | Contenido |
|---|---|
| 0–4 min | Explique cómo se construyó el catálogo, cómo se preservaron las listas de candidatos y qué aporta cada modelo. |
| 4–8 min | Muestre la recuperación, el plan del agente, la memoria, la transacción de compra y la gestión de memoria de Colab. |
| 8–16 min | Muestre por voz la compra válida y la recuperación por cambio de stock o la aclaración. En uno de los turnos muestre el tiempo real desde la grabación hasta la respuesta. |
| 16–20 min | Presente las tablas producidas en Colab, sus aciertos, sus errores y las limitaciones observadas. |

La voz de referencia y los audios de prueba generados se identifican como sintéticos. La validación por texto de VS Code y las métricas reales del kernel Colab corresponden a ejecuciones diferentes.

## Referencias

- Reddy et al. (2022), *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search*. https://arxiv.org/abs/2206.06588
- Repositorio oficial ESCI y licencia del proyecto. https://github.com/amazon-science/esci-data
- E5 multilingüe. https://huggingface.co/intfloat/multilingual-e5-small
- Qwen2.5 Instruct. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Qwen3‑TTS. https://github.com/QwenLM/Qwen3-TTS
- Faster‑Whisper. https://github.com/SYSTRAN/faster-whisper
