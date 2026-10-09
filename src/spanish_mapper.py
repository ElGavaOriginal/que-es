# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026


from config import TARGET_OBJECTS

class SpanishLabelMapper:
    """
    Converts a class label and instance count into a Spanish noun
    phrase with the appropriate grammatical article.
     * Common Objects in Context (COCO)

    Each entry in LABEL_MAP follows this structure:
        "coco_label": (singular, plural, gender)
        gender: "m" = masculine, "f" = feminine

    The gender, singular, and plural forms in Spanish are not always predictable 
    from the English label. This is the manual mapping:
            - Indefinite singular:  un (masc), una (fem)
            - Indefinite plural:  unos (masc), unas (fem)
            - Definite singular:  el (masc), la (fem)
            - Definite plural:  los (masc), las (fem)

    """

    LABEL_MAP = {
    "refrigerator": ("refrigerador",  "refrigeradores", "m"),
    "oven":         ("horno",         "hornos",         "m"),
    "sink":         ("fregadero",     "fregaderos",     "m"),
    "couch":        ("sofá",          "sofás",          "m"),
    "chair":        ("silla",         "sillas",         "f"),
    "potted plant": ("planta",        "plantas",        "f"),
    "book":         ("libro",         "libros",         "m"),
    "dining table": ("mesa",          "mesas",          "f"),
    "cup":          ("taza",          "tazas",          "f"),
    "bowl":         ("tazón",         "tazones",        "m"),
    "fork":         ("tenedor",       "tenedores",      "m"),
    "knife":        ("cuchillo",      "cuchillos",      "m"),
    "spoon":        ("cuchara",       "cucharas",       "f"),
    "laptop":       ("computadora",   "computadoras",   "f"),
    "keyboard":     ("teclado",       "teclados",       "m"),
    "mouse":        ("ratón",         "ratones",        "m"),
    "scissors":     ("tijeras",       "tijeras",        "f"),
    "cell phone":   ("teléfono",      "teléfonos",      "m"),
    "backpack":     ("mochila",       "mochilas",       "f"),
    "car":          ("carro",         "carros",         "m"),
}

    # Article lookup table keyed by gender, number, and article_type
    ARTICLES = {
        ("m", "singular", "indefinite"): "un",
        ("f", "singular", "indefinite"): "una",
        ("m", "plural",   "indefinite"): "unos",
        ("f", "plural",   "indefinite"): "unas",
        ("m", "singular", "definite"):   "el",
        ("f", "singular", "definite"):   "la",
        ("m", "plural",   "definite"):   "los",
        ("f", "plural",   "definite"):   "las",
    }


    def get_label(self, coco_label: str, count: int,
                  article_type: str = "indefinite") -> str:
        """
        Returns a Spanish noun phrase for a detected object.

        Args:
            coco_label:   The COCO class name (e.g. "dog").
            count:        Number of instances detected in the frame.
            article_type: "indefinite" (un/una/unos/unas)
                          or "definite"  (el/la/los/las).

        Returns:
            A Spanish noun phrase string, e.g. "un perro" or "dos perros".
            Returns the raw COCO label if no mapping exists.

        Examples:
            get_label("dog", 1)  -> "un perro"
            get_label("dog", 3)  -> "tres perros"
            get_label("book", 1, "definite") -> "el libro"
        """
        if coco_label not in self.LABEL_MAP:
            return coco_label  # fallback: return raw label unchanged

        singular, plural, gender = self.LABEL_MAP[coco_label]
        number = "singular" if count == 1 else "plural"
        noun   = singular   if count == 1 else plural

        article = self.ARTICLES.get((gender, number, article_type), "")

        # For plural counts greater than 2, prepend the Spanish number word
        if count > 1:
            number_word = self._to_spanish_number(count)
            return f"{number_word} {noun}"

        return f"{article} {noun}"


    def get_all_labels(self, detections: dict,
                       article_type: str = "indefinite") -> dict:
        """
        Converts a dictionary of COCO detections to Spanish label phrases.

        Args:
            detections:   Dict of {coco_label: count}, e.g. {"dog": 2, "book": 1}.
            article_type: "indefinite" or "definite".

        Returns:
            Dict of {coco_label: spanish_phrase}, e.g.
            {"dog": "dos perros", "book": "un libro"}.
        """
        return {
            label: self.get_label(label, count, article_type)
            for label, count in detections.items()
            if label in TARGET_OBJECTS
        }


    @staticmethod
    def _to_spanish_number(n: int) -> str:
        """
        Returns the Spanish word for numbers 1-10.
        Falls back to the numeral for anything higher.
        """
        number_words = {
            1:  "un",
            2:  "dos",
            3:  "tres",
            4:  "cuatro",
            5:  "cinco",
            6:  "seis",
            7:  "siete",
            8:  "ocho",
            9:  "nueve",
            10: "diez",
        }
        return number_words.get(n, str(n))
    