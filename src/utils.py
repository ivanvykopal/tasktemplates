import re
import numpy as np

LANGUAGES = {
    "ab": "Abkhazian",
    "aa": "Afar",
    "af": "Afrikaans",
    "ak": "Akan",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "an": "Aragonese",
    "hy": "Armenian",
    "as": "Assamese",
    "av": "Avaric",
    "ae": "Avestan",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "bm": "Bambara",
    "ba": "Bashkir",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bi": "Bislama",
    "bs": "Bosnian",
    "br": "Breton",
    "bg": "Bulgarian",
    "my": "Burmese",
    "ca": "Catalan, Valencian",
    "ch": "Chamorro",
    "ce": "Chechen",
    "ny": "Chichewa, Chewa, Nyanja",
    "zh": "Chinese",
    "cu": "Church Slavic, Old Slavonic, Church Slavonic, Old Bulgarian, Old Church Slavonic",
    "cv": "Chuvash",
    "kw": "Cornish",
    "co": "Corsican",
    "cr": "Cree",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "dv": "Divehi, Dhivehi, Maldivian",
    "nl": "Dutch, Flemish",
    "dz": "Dzongkha",
    "en": "English",
    "eo": "Esperanto",
    "et": "Estonian",
    "ee": "Ewe",
    "fo": "Faroese",
    "fj": "Fijian",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ff": "Fulah",
    "gd": "Gaelic, Scottish Gaelic",
    "gl": "Galician",
    "lg": "Ganda",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek, Modern (1453–)",
    "kl": "Kalaallisut, Greenlandic",
    "gn": "Guarani",
    "gu": "Gujarati",
    "ht": "Haitian, Haitian Creole",
    "ha": "Hausa",
    "he": "Hebrew",
    "hz": "Herero",
    "hi": "Hindi",
    "ho": "Hiri Motu",
    "hu": "Hungarian",
    "is": "Icelandic",
    "io": "Ido",
    "ig": "Igbo",
    "id": "Indonesian",
    "ia": "Interlingua (International Auxiliary Language Association)",
    "ie": "Interlingue, Occidental",
    "iu": "Inuktitut",
    "ik": "Inupiaq",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "kn": "Kannada",
    "kr": "Kanuri",
    "ks": "Kashmiri",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "ki": "Kikuyu, Gikuyu",
    "rw": "Kinyarwanda",
    "ky": "Kirghiz, Kyrgyz",
    "kv": "Komi",
    "kg": "Kongo",
    "ko": "Korean",
    "kj": "Kuanyama, Kwanyama",
    "ku": "Kurdish",
    "lo": "Lao",
    "la": "Latin",
    "lv": "Latvian",
    "li": "Limburgan, Limburger, Limburgish",
    "ln": "Lingala",
    "lt": "Lithuanian",
    "lu": "Luba-Katanga",
    "lb": "Luxembourgish, Letzeburgesch",
    "mk": "Macedonian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ml": "Malayalam",
    "mt": "Maltese",
    "gv": "Manx",
    "mi": "Maori",
    "mr": "Marathi",
    "mh": "Marshallese",
    "mn": "Mongolian",
    "na": "Nauru",
    "nv": "Navajo, Navaho",
    "nd": "North Ndebele",
    "nr": "South Ndebele",
    "ng": "Ndonga",
    "ne": "Nepali",
    "no": "Norwegian",
    "nb": "Norwegian Bokmål",
    "nn": "Norwegian Nynorsk",
    "ii": "Sichuan Yi, Nuosu",
    "oc": "Occitan",
    "oj": "Ojibwa",
    "or": "Oriya",
    "om": "Oromo",
    "os": "Ossetian, Ossetic",
    "pi": "Pali",
    "ps": "Pashto, Pushto",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi, Panjabi",
    "qu": "Quechua",
    "ro": "Romanian, Moldavian, Moldovan",
    "rm": "Romansh",
    "rn": "Rundi",
    "ru": "Russian",
    "se": "Northern Sami",
    "sm": "Samoan",
    "sg": "Sango",
    "sa": "Sanskrit",
    "sc": "Sardinian",
    "sr": "Serbian",
    "sn": "Shona",
    "sd": "Sindhi",
    "si": "Sinhala, Sinhalese",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "st": "Southern Sotho",
    "es": "Spanish, Castilian",
    "su": "Sundanese",
    "sw": "Swahili",
    "ss": "Swati",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ty": "Tahitian",
    "tg": "Tajik",
    "ta": "Tamil",
    "tt": "Tatar",
    "te": "Telugu",
    "th": "Thai",
    "bo": "Tibetan",
    "ti": "Tigrinya",
    "to": "Tonga (Tonga Islands)",
    "ts": "Tsonga",
    "tn": "Tswana",
    "tr": "Turkish",
    "tk": "Turkmen",
    "tw": "Twi",
    "ug": "Uighur, Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "ve": "Venda",
    "vi": "Vietnamese",
    "vo": "Volapük",
    "wa": "Walloon",
    "cy": "Welsh",
    "wo": "Wolof",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "za": "Zhuang, Chuang",
    "zu": "Zulu",
}


def match_variable(text, example):
    match_string = r'\{\{([^{}]+)\}\}'

    def replace(match):
        expression = match.group(1)
        index_match = r'\[(.*?)\]'
        if re.search(index_match, expression):
            match = re.search(index_match, key)
            index = int(match.group(1))
            key = re.sub(index_match, '', expression)
            value = example.get(key, [])
            if isinstance(value, list):
                return value[index] if index < len(value) else value[-1]
            else:
                return value
        else:
            try:
                result = eval(expression, {}, example)
                return str(result)
            except Exception as e:
                print(e)
            return example.get(key, match.group(0))
    return re.sub(match_string, replace, text)


def match_function(text, example):
    match_string = r'\{\{([^{}]+)\}\}'

    def replace(match):
        expression = match.group(1)
        try:
            # Dynamically define a lambda function,
            matched_func = re.search(r'lambda (.*): (.*)', expression)
            variable = matched_func.group(1)
            func = matched_func.group(2)

            func = re.sub(r'\b' + variable + r'\b',
                          f'example.get("{variable}")', func)

            func = eval(f'lambda example: {func}', {}, example)

            # Call the lambda function with the example dictionary
            result = func(example)

            return str(result)
        except Exception as e:
            return str(e)  # If there's any error, return the error message

    return re.sub(match_string, replace, text)


def apply_preprocess_steps(text, preprocess_steps):
    for step in preprocess_steps:
        text = step(text)
    return text


def process_template(text, example, choices, preproces_steps):
    match_string = r'\{\{([^{}]+)\}\}'

    def replace(match):
        expression = match.group(1)
        function_match = r'\((.*?)\)'
        # index_match = r'\[(.*?)\]'
        index_match = r'(.*?)\[(.*?)\]'
        if re.search(function_match, expression) and not re.search(r'lambda', expression):
            match = re.search(function_match, expression)
            variable = re.match(index_match, match.group(1))
            expression = re.sub(variable.group(
                1), f'example.get("{variable.group(1)}")', expression)
            result = eval(expression, {"example": example}, example)
            return apply_preprocess_steps(str(result), preproces_steps)
        elif re.search(index_match, expression) and not re.search(r'lambda', expression):
            match = re.search(index_match, expression)
            expression = re.sub(match.group(
                1), f'example.get("{match.group(1)}")', expression)
            results = eval(expression, {"example": example}, example)
            return apply_preprocess_steps(str(results), preproces_steps)
        else:
            try:
                # Check if the expression is a lambda function
                matched_func = re.match(r'lambda (.*): (.*)', expression)
                if matched_func:
                    variable = matched_func.group(1)
                    func = matched_func.group(2)

                    func = re.sub(r'\b' + variable + r'\b',
                                  f'example.get("{variable}")', func)

                    func = eval(
                        f'lambda example: {func}',
                        {
                            'np': np,
                            'choices': choices
                        },
                        example
                    )

                    # Call the lambda function with the example dictionary
                    result = func(example)
                    return apply_preprocess_steps(
                        str(result), preproces_steps)
                else:
                    # Evaluate other expressions normally
                    result = eval(expression, {}, example)
                    return apply_preprocess_steps(
                        str(result), preproces_steps)
            except Exception as e:
                raise e

    return re.sub(match_string, replace, text)
