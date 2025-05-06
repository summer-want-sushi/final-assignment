from typing import Dict, List, Optional, Union
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import re
from patterns import (
    PATRONES_AMBIGUEDAD_LEXICA,
    PATRONES_AMBIGUEDAD_SINTACTICA,
    SUGERENCIAS_MEJORA
)

class SemanticAnalyzer:
    """
    Analizador semántico que utiliza embeddings para comparar textos.
    """
    def __init__(self, model_name: str = "PlanTL-GOB-ES/roberta-base-bne"):
        """
        Inicializa el analizador semántico.
        
        Args:
            model_name (str): Nombre del modelo de HuggingFace a utilizar
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Error cargando el modelo {model_name}: {str(e)}")

    def get_embedding(self, texto: str) -> np.ndarray:
        """
        Obtiene el embedding de un texto usando el modelo de transformers.
        
        Args:
            texto (str): Texto a procesar
            
        Returns:
            np.ndarray: Vector de embedding
        """
        inputs = self.tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]

    def calcular_similitud(self, texto1: str, texto2: str) -> float:
        """
        Compara la similitud semántica entre dos textos.
        
        Args:
            texto1 (str): Primer texto
            texto2 (str): Segundo texto
            
        Returns:
            float: Score de similitud entre 0 y 1
        """
        emb1 = self.get_embedding(texto1)
        emb2 = self.get_embedding(texto2)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

class AmbiguityClassifier:
    """
    Clasificador de ambigüedades en historias de usuario.
    Detecta ambigüedades léxicas y sintácticas, y proporciona sugerencias de mejora.
    """
    
    def __init__(self, model_name: str = "PlanTL-GOB-ES/roberta-base-bne"):
        """
        Inicializa el clasificador de ambigüedades.
        
        Args:
            model_name (str): Nombre del modelo de HuggingFace a utilizar
        """
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            raise RuntimeError("Es necesario instalar el modelo es_core_news_sm. Ejecute: python -m spacy download es_core_news_sm")
            
        self.semantic_analyzer = SemanticAnalyzer(model_name)

    def __call__(self, texto: str) -> Dict[str, Union[bool, List[str], float]]:
        """
        Analiza una historia de usuario en busca de ambigüedades.
        
        Args:
            texto (str): Historia de usuario a analizar
            
        Returns:
            Dict: Resultado del análisis con tipos de ambigüedad y sugerencias
        """
        if not texto or not isinstance(texto, str):
            return {
                "tiene_ambiguedad": False,
                "ambiguedad_lexica": [],
                "ambiguedad_sintactica": [],
                "sugerencias": ["El texto está vacío o no es válido"],
                "score_ambiguedad": 0.0
            }

        # Procesar el texto con spaCy
        doc = self.nlp(texto.strip())
        
        # Detectar ambigüedades léxicas
        ambiguedades_lexicas = []
        for patron in PATRONES_AMBIGUEDAD_LEXICA:
            if re.search(patron["patron"], texto, re.IGNORECASE):
                ambiguedades_lexicas.append({
                    "tipo": patron["tipo"],
                    "descripcion": patron["descripcion"]
                })

        # Detectar ambigüedades sintácticas
        ambiguedades_sintacticas = []
        for patron in PATRONES_AMBIGUEDAD_SINTACTICA:
            if re.search(patron["patron"], texto, re.IGNORECASE):
                ambiguedades_sintacticas.append({
                    "tipo": patron["tipo"],
                    "descripcion": patron["descripcion"]
                })

        # Generar sugerencias de mejora
        sugerencias = []
        if ambiguedades_lexicas or ambiguedades_sintacticas:
            for ambiguedad in ambiguedades_lexicas + ambiguedades_sintacticas:
                tipo = ambiguedad["tipo"]
                if tipo in SUGERENCIAS_MEJORA:
                    sugerencias.extend(SUGERENCIAS_MEJORA[tipo])

        # Calcular score de ambigüedad
        score = len(ambiguedades_lexicas) * 0.4 + len(ambiguedades_sintacticas) * 0.6
        score_normalizado = min(1.0, score / 5.0)  # Normalizar a un rango de 0 a 1

        return {
            "tiene_ambiguedad": bool(ambiguedades_lexicas or ambiguedades_sintacticas),
            "ambiguedad_lexica": [amb["descripcion"] for amb in ambiguedades_lexicas],
            "ambiguedad_sintactica": [amb["descripcion"] for amb in ambiguedades_sintacticas],
            "sugerencias": sugerencias if sugerencias else ["No se encontraron ambigüedades"],
            "score_ambiguedad": round(score_normalizado, 2)
        }

    def analizar_similitud_semantica(self, texto1: str, texto2: str) -> float:
        """
        Compara la similitud semántica entre dos textos.
        
        Args:
            texto1 (str): Primer texto
            texto2 (str): Segundo texto
            
        Returns:
            float: Score de similitud entre 0 y 1
        """
        return self.semantic_analyzer.calcular_similitud(texto1, texto2) 