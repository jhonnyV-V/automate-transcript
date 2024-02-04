def whisper_lang_to_seamless_lang(code):
    LANGUAGES = {
        "en": "eng",
        "zh": "cmn",
        "de": "deu",
        "es": "spa",
        "ru": "rus",
        "ko": "kor",
        "fr": "fra",
        "ja": "jpn",
        "pt": "por",
        "tr": "tur",
        "pl": "pol",
        "ca": "cat",
        "nl": "nld",
        "ar": "arb",
        "sv": "swe",
        "it": "ita",
        "id": "ind",
        "hi": "hin",
        "fi": "fin",
        "vi": "vie",
        "he": "heb",
        "uk": "ukr",
        "el": "ell",
        "ms": "zsm",
        "cs": "ces",
        "ro": "ron",
        "da": "dan",
        "hu": "hun",
        "ta": "tam",
        "no": "nob",
        "th": "tha",
        "ur": "urd",
        "hr": "hrv",
        "bg": "bul",
        "lt": "lit",
        "cy": "cym",
        "sk": "slk",
        "te": "tel",
        "fa": "pes",
        "lv": "lvs",
        "bn": "ben",
        "sr": "srp",
        "az": "azj",
        "sl": "slv",
        "kn": "kan",
        "et": "est",
        "mk": "mkd",
        "eu": "eus",
        "is": "isl",
        "hy": "hye",
        "ne": "npi",
        "mn": "khk",
        "bs": "bos",
        "kk": "kaz",
        "sw": "swh",
        "gl": "glg",
        "mr": "mar",
        "pa": "pan",
        "km": "khm",
        "sn": "sna",
        "yo": "yor",
        "so": "som",
        "af": "afr",
        "oc": "oci",
        "ka": "kat",
        "be": "bel",
        "tg": "tgk",
        "sd": "snd",
        "lo": "lao",
        "uz": "uzn",
        "ps": "pbt",
        "mt": "mlt",
        "lb": "ltz",
        "tl": "tgl",
        "as": "asm",
        "jw": "jav",
        "yue": "yue",
    }
    return LANGUAGES[code]
