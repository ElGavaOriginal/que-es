# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026


from config import TARGET_OBJECTS

class SpanishLabelMapper:
    """
    Converts a class label into a Spanish noun phrase with the appropriate grammatical article.
     * Common Objects in Context (COCO)

    Each entry in LABEL_MAP follows this structure:
        "coco_label": (singular, plural, gender)
        gender: "m" = masculine, "f" = feminine

    Singular (one instance detected):
        - indefinite:           un (masc), una (fem)        -> "un libro"
        - definite:             el (masc), la (fem)         -> "el libro"
 
    Plural (more than one instance detected):
        - indefinite plural:    unos (masc), unas (fem)     -> "unos libros"
        - definite plural:      los (masc), las (fem)       -> "los libros"

    Nouns in PLURAL_ONLY (e.g. "tijeras" - the only noun for this program), and it 
    conventionally exists in plural form in spanish to refer to a single cutting instrument. 
    While grammatically, the singular 'tijera' is grammatically accepted, it is not commonly 
    used in practice and seems to be falling out of common usage within contemporary Spanish
    of the United States. 
    
    """

    ARTICLE_TYPES = ("indefinite", "definite")

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

    # Nouns that only exist in plural form in this program
    PLURAL_ONLY = {"scissors"}


    # Article lookup table keyed by gender and article_type
    SINGULAR_ARTICLES = {
        ("m", "indefinite"): "un",
        ("f", "indefinite"): "una",
        ("m", "definite"):   "el",
        ("f", "definite"):   "la",
    }

    PLURAL_ARTICLES = {
        ("m", "indefinite"): "unos",
        ("f", "indefinite"): "unas",
        ("m", "definite"):   "los",
        ("f", "definite"):   "las",
    }


    def _is_plural(self, coco_label: str, count: int) -> bool:
        """
        A label is plural if it is a plural-only noun (e.g. scissors) or 
        if more than one instance was detected.
        """
        
        return coco_label in self.PLURAL_ONLY or count > 1


    def get_label(self, coco_label: str, count: int = 1,
                  article_type: str = "indefinite") -> str:
        """
        Returns a Spanish noun phrase for a detected object.

        Args:
            coco_label:   The COCO class name (e.g. "book").
            count:        The number of instances detected. Must be >= 1.
                          only decides whether to use the singular or plural form.
            article_type: "indefinite" (un/una/unos/unas) 
                          "definite"   (el/la/los/las)

        Returns:
            A Spanish noun phrase string, e.g. "un libro" or "las sillas".
            Returns the raw COCO label if no mapping exists.

        Examples:
            get_label("book", 1)                -> "un libro"
            get_label("book", 2)                -> "unos libros"
            get_label("chair", 1, "definite")   -> "la silla"
            get_label("chair", 2, "definite")   -> "las sillas"
            get_label("scissors", 1)            -> "unas tijeras"
        """

        if article_type not in self.ARTICLE_TYPES:
            raise ValueError(f"article_type must be {self.ARTICLE_TYPES}, "
                             f"got {article_type!r}"
            )

        if count < 1:
            raise ValueError(f"count must be >= 1, got {count!r}")
        
        if coco_label not in self.LABEL_MAP:
            return coco_label  # fallback: return raw label unchanged

        singular, plural, gender = self.LABEL_MAP[coco_label]

        if self._is_plural(coco_label, count):
            noun = plural
            article = self.PLURAL_ARTICLES[(gender, article_type)]
        else:
            noun = singular
            article = self.SINGULAR_ARTICLES[(gender, article_type)]

        return f"{article} {noun}"


    def get_all_labels(self, counts, article_type: str = "indefinite") -> dict:
        """
        Converts COCO labels to Spanish label phrases.

        Args:
            counts:         A dict mapping {coco_label: count}, 
                            ex. {"chair": 2, "book": 1}
            article_type:   "indefinite" or "definite" 

        Returns:
            Dict of {coco_label: spanish_phrase}, e.g.
            {"chair": "unas sillas", "book": "un libro"}.
            Labels not in TARGET_OBJECTS are excluded from the result.
        """

        return {
            label: self.get_label(label, count, article_type)
            for label, count in counts.items()
            if label in TARGET_OBJECTS
        }
