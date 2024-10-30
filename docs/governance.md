# Prácticas de Gobernanza para el proyecto
#### Team 10
Armando Cerda A01570376\
Karla Fonseca A01795313\
Juan Antonio Cruz A01795375\
Carlos Villena A01795127\
Renzo Torres A01795380

## 1. Política de Gobernanza de Modelos

### Objetivo

Garantizar que todos los modelos de machine learning del proyecto sean reproducibles, rastreables y mantengan un desempeño constante en distintos entornos. Esto asegura que el modelo siga cumpliendo con los estándares de calidad y que cualquier cambio en el modelo esté debidamente registrado.

### Alcance

Esta política se aplica a todos los modelos desarrollados, entrenados, y desplegados en este proyecto, abarcando el control de versiones, la trazabilidad de datos, y las evaluaciones periódicas de desempeño.


### Procedimientos

1. **Control de Versiones con DVC y MLFlow**
   
    - **Versionado de Datasets**: Usar DVC para rastrear versiones de datasets y evitar que los cambios en los datos afecten los resultados del modelo sin registro.
    - **Versionado de Modelos**: Con MLFlow, cada modelo entrenado se registra y se almacena con etiquetas de versión. Cada nueva versión debe tener notas que especifiquen los cambios en parámetros, datos y métricas de evaluación.

2. **Monitoreo de Desempeño**

    - **Evaluaciones Periódicas**: Configurar MLFlow para realizar una evaluación periódica del desempeño de los modelos, incluyendo métricas como precisión, recall y F1-score. Esto se documenta automáticamente en cada ejecución y se almacena en un archivo de reporte.
    - **Alertas de Desviación**: Implementar alertas automáticas para que el equipo reciba notificaciones si el rendimiento del modelo baja en más de un 10% en relación con la línea base inicial. _Pendiente de implementación_

3. **Pruebas de Reproducibilidad**

    - **Pipeline de Reproducción**: Usar DVC para crear pipelines automatizados que permitan la regeneración del modelo usando los mismos datos y parámetros. Realizar estas pruebas al menos una vez antes de cada despliegue.
    - **Documentación de Variables Críticas**: Registrar las variables de entorno, la configuración de hardware y las versiones de software en el archivo de configuración de cada pipeline para asegurar que se pueda reproducir exactamente el mismo modelo en el futuro.

### Resultados Esperados

La trazabilidad del modelo y los datos permite que el equipo identifique la procedencia de cualquier versión del modelo, asegure su reproducibilidad y mantenga un historial completo de las métricas de desempeño, brindando seguridad y transparencia en el ciclo de vida del modelo.

## 2: Política de Estándares de Código

### Objetivo
Establecer una base clara de estándares de codificación que aseguren la coherencia, legibilidad y mantenimiento del código, alineando el proyecto con buenas prácticas de desarrollo y minimizando errores en producción.

### Alcance
Esta política abarca todo el código fuente del proyecto y es obligatoria para todos los miembros del equipo de desarrollo, incluidos becarios y colaboradores externos.

### Procedimientos

1. **Normas de Estilo de Código**

    - **Guía de Estilo**: El equipo debe seguir las guías de estilo PEP 8 para Python y otras guías específicas para las tecnologías utilizadas (como JavaScript o SQL).
    - **Herramientas de Formato**: Usar black o pylint en Python y linters específicos en otros lenguajes para asegurar que el código cumple con el estándar de estilo definido.

2. **Integración Continua (CI) con GitHub Actions**
    - **Pruebas Automatizadas**: Implementar un flujo CI/CD en GitHub Actions que ejecute pruebas automatizadas en cada push al repositorio. Esto asegura que el código nuevo no rompa funcionalidades existentes.  _Pendiente de implementación_
    - **Revisión de Código**: Todo cambio debe pasar por revisión de al menos un miembro del equipo. Las revisiones se deben centrar en mejorar la legibilidad y eficiencia del código, así como identificar errores potenciales.

3. **Documentación y Comentarios**

    - **Documentación Completa**: Cada función y clase debe tener una breve descripción de su propósito y de los parámetros que recibe. Los archivos de configuración deben tener comentarios que expliquen el propósito de cada parámetro relevante.
    - **Manual de Código:** Generar y mantener un manual de código en el repositorio donde se detallen las convenciones específicas adoptadas, como nombres de variables o estructuras de carpeta.

### Resultados Esperados

La estandarización y revisión de código permitirá un mantenimiento más sencillo, la incorporación ágil de nuevos colaboradores, y la identificación de posibles errores de forma temprana, elevando la calidad y confiabilidad del código.
