"""
Patrones comunes de ambigüedad en historias de usuario.
"""

AMBIGUITY_PATTERNS = {
    "lexicos": {
        # Palabras con múltiples significados comunes
        "fácil", "simple", "rápido", "eficiente",
        "mejor", "adecuado", "apropiado",
        "flexible", "dinámico", "intuitivo",
        "amigable", "óptimo", "robusto",
        
        # Cuantificadores ambiguos
        "algunos", "varios", "muchos", "pocos",
        "más", "menos", "suficiente",
        "bastante", "demasiado", "aproximadamente",
        
        # Temporales ambiguos
        "pronto", "rápidamente", "periódicamente",
        "regularmente", "ocasionalmente", "frecuentemente",
        "en tiempo real", "instantáneamente", "ágilmente",
        
        # Calificadores ambiguos
        "moderno", "innovador", "avanzado",
        "inteligente", "sofisticado", "elegante",
        
        # Términos de usabilidad ambiguos
        "user-friendly", "intuitivo", "natural",
        "seamless", "fluido", "sin problemas"
    },
    
    "sintacticos": {
        # Patrones de estructura ambigua
        r"(.*?) y (.*?) con (.*?)",  # Ambigüedad de alcance de "con"
        r"no (.*?) y (.*?)",  # Ambigüedad de alcance de negación
        r"(.*?) o (.*?) y (.*?)",  # Ambigüedad de operadores lógicos
        r"(.*?) pero (.*?) si (.*?)",  # Condiciones ambiguas
        r"(.*?) cuando (.*?) o (.*?)",  # Temporalidad ambigua
        r"(.*?) excepto (.*?) y (.*?)",  # Exclusiones ambiguas
        r"(.*?) antes de (.*?) y (.*?)",  # Secuencia temporal ambigua
        r"(.*?) después de (.*?) o (.*?)",  # Secuencia condicional ambigua
    },
    
    "semanticos": {
        # Frases que suelen indicar ambigüedad
        "si es posible",
        "cuando sea necesario",
        "si se requiere",
        "según corresponda",
        "como sea apropiado",
        "en caso de ser necesario",
        "dependiendo del caso",
        "si aplica",
        "cuando corresponda",
        "si es factible",
        "en la medida de lo posible",
        "siempre que sea posible"
    },
    
    "contextuales": {
        # Dependencias implícitas
        r"(?i)similar a (.*?)",  # Referencias vagas
        r"(?i)como en (.*?)",   # Comparaciones ambiguas
        r"(?i)igual que (.*?)", # Referencias no específicas
        
        # Suposiciones de conocimiento
        r"(?i)de la manera usual",
        r"(?i)como siempre",
        r"(?i)de forma estándar",
        
        # Referencias ambiguas
        r"(?i)esto",
        r"(?i)eso",
        r"(?i)aquello",
        r"(?i)lo mismo"
    }
}

# Términos técnicos que no son ambiguos en el contexto
TECHNICAL_TERMS = {
    # Autenticación y Seguridad
    "OAuth", "autenticación", "autorización",
    "token", "JWT", "SSO", "2FA", "MFA",
    
    # Datos y Almacenamiento
    "base de datos", "SQL", "NoSQL", "cache",
    "índice", "backup", "restauración",
    
    # Frontend
    "responsive", "CSS", "HTML", "JavaScript",
    "React", "Angular", "Vue", "DOM",
    
    # Backend
    "API", "REST", "GraphQL", "webhook",
    "microservicio", "contenedor", "Docker",
    
    # Operaciones
    "logging", "monitoreo", "alertas",
    "deployment", "CI/CD", "pipeline",
    
    # Términos de negocio
    "ROI", "KPI", "SLA", "métrica",
    "dashboard", "reporte", "análisis"
}

# Patrones de estructura de historia de usuario
USER_STORY_PATTERNS = {
    'estandar': r"(?i)^como\s+(.+?),?\s+quiero\s+(.+?)(?:\s+para\s+(?:que\s+)?(.+))?$",
    'modal': r"(?i)^(?:un|una|el|la)\s+(.+?)\s+(?:puede|debe|debería)\s+(.+?)(?:\s+para\s+(?:que\s+)?(.+))?$",
    'pasiva': r"(?i)^(?:el|la|los|las)\s+(.+?)\s+(?:debe|deben|debería|deberían)\s+(?:ser|estar)\s+(.+?)(?:\s+para\s+(?:que\s+)?(.+))?$",
    'declarativa': r"(?i)^(?:los|las)\s+(.+?)\s+(?:deben|deberían)\s+(.+?)(?:\s+para\s+(?:que\s+)?(.+))?$",
    'necesidad': r"(?i)^(?:necesito|necesitamos|se\s+necesita)\s+(.+?)(?:\s+para\s+(?:que\s+)?(.+))?$",
    'deseo': r"(?i)^(?:deseo|deseamos|se\s+desea)\s+(.+?)(?:\s+para\s+(?:que\s+)?(.+))?$"
}

# Patrones para detectar ambigüedades léxicas
PATRONES_AMBIGUEDAD_LEXICA = [
    {
        "patron": r"\b(rápido|eficiente|fácil|simple|intuitivo|amigable|flexible|robusto)\b(?![^{]*\})",
        "tipo": "adjetivo_subjetivo",
        "descripcion": "Uso de adjetivos subjetivos que pueden interpretarse de diferentes maneras"
    },
    {
        "patron": r"\b(varios|algunos|muchos|pocos|diversos|múltiples)\b(?!\s+(?:formatos?|tipos?|archivos?|reportes?)\s+(?:como|:|\(|\{))",
        "tipo": "cuantificador_ambiguo",
        "descripcion": "Uso de cuantificadores ambiguos que no especifican una cantidad concreta"
    },
    {
        "patron": r"\b(etc|etcétera|entre otros|y más|y otros)\b",
        "tipo": "enumeracion_incompleta",
        "descripcion": "Uso de expresiones que dejan la enumeración incompleta o abierta"
    },
    {
        "patron": r"\b(sistema|aplicación|plataforma|herramienta|solución)\b(?!\s+(?:debe|debería|tiene que|ha de))",
        "tipo": "termino_generico",
        "descripcion": "Uso de términos genéricos que no especifican la funcionalidad concreta"
    }
]

# Patrones para detectar ambigüedades sintácticas
PATRONES_AMBIGUEDAD_SINTACTICA = [
    {
        "patron": r"(?i)(?<![\w{])(y|o|y/o)(?!\s+(?:\d+|\{|\w+\s*[=:<>]))",
        "tipo": "coordinacion_ambigua",
        "descripcion": "Uso de coordinaciones que pueden crear ambigüedad en la interpretación"
    },
    {
        "patron": r"(?i)\b(esto|eso|aquello|el cual|la cual|lo cual|que)\b(?!\s+(?:significa|implica|requiere|incluye))",
        "tipo": "referencia_ambigua",
        "descripcion": "Uso de referencias ambiguas que pueden tener múltiples antecedentes"
    },
    {
        "patron": r"(?i)\b(si|cuando|mientras|después|antes|luego)\b(?!\s+(?:el|la|los|las|se)\s+(?:\w+\s+){0,3}(?:\d+|específico|definido))",
        "tipo": "condicion_temporal_ambigua",
        "descripcion": "Uso de condiciones o referencias temporales ambiguas"
    },
    {
        "patron": r"(?i)(poder|deber|necesitar|querer)\s+\w+\s+(y|o)\s+\w+(?!\s+(?:en|durante|cada|por)\s+(?:\d+|un|una)\s+(?:segundo|minuto|hora)s?)",
        "tipo": "alcance_verbo_modal",
        "descripcion": "Ambigüedad en el alcance de verbos modales con múltiples acciones"
    }
]

# Sugerencias de mejora para cada tipo de ambigüedad
SUGERENCIAS_MEJORA = {
    "adjetivo_subjetivo": [
        "Especificar métricas o criterios medibles (ej: tiempo de respuesta en segundos)",
        "Definir valores concretos o rangos aceptables",
        "Usar términos más específicos y cuantificables"
    ],
    "cuantificador_ambiguo": [
        "Especificar cantidades exactas o rangos definidos",
        "Listar explícitamente los elementos o tipos",
        "Definir límites mínimos y máximos"
    ],
    "enumeracion_incompleta": [
        "Listar todos los elementos requeridos",
        "Especificar criterios de inclusión/exclusión",
        "Definir el alcance completo de la funcionalidad"
    ],
    "termino_generico": [
        "Especificar la funcionalidad concreta",
        "Describir las características técnicas específicas",
        "Detallar los componentes o módulos involucrados"
    ],
    "coordinacion_ambigua": [
        "Separar en historias de usuario independientes",
        "Usar listas numeradas o viñetas para clarificar",
        "Especificar la relación entre los elementos"
    ],
    "referencia_ambigua": [
        "Repetir el sustantivo al que se hace referencia",
        "Usar referencias específicas y directas",
        "Evitar pronombres ambiguos"
    ],
    "condicion_temporal_ambigua": [
        "Especificar intervalos de tiempo exactos",
        "Definir el orden preciso de las acciones",
        "Usar referencias temporales específicas (ej: cada 5 minutos)"
    ],
    "alcance_verbo_modal": [
        "Separar las acciones en requisitos independientes",
        "Especificar las condiciones para cada acción",
        "Definir la prioridad o secuencia de las acciones"
    ]
} 