"""Catálogos escritos para recolectar dactilología, independientes por lengua.
Las entradas son etiquetas para grabar: no contienen gestos ni pesos entrenados.
"""
from string import ascii_uppercase
from sign_languages import language_info, labels_for, register_label, LANGUAGES

LATIN = list(ascii_uppercase)
CATALOGS = {
    'lsm': {'title': 'Abecedario español', 'symbols': list('ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'),
            'extras': list('ÁÉÍÓÚÜ'), 'note': '27 letras, incluida Ñ. Las tildes son variantes escritas opcionales; registra su representación según la convención que uses en LSM.',
            'source': 'https://nuevaescuelamexicana.sep.gob.mx/contenido/recurso/40967/'},
    'dgs': {'title': 'Abecedario alemán', 'symbols': LATIN + list('ÄÖÜß'),
            'extras': [], 'note': 'A–Z y los caracteres Ä, Ö, Ü y ß. Graba la dactilología de DGS, incluyendo sus movimientos y convenciones.',
            'source': 'https://www.taubblindenwerk.de/haeufig-gestellte-fragen/fingeralphabet/'},
    'asl': {'title': 'Abecedario inglés', 'symbols': LATIN, 'extras': [],
            'note': '26 letras para recolectar dactilología de ASL. Conserva el movimiento de las letras dinámicas.', 'source': ''},
    'jsl': {'title': 'Kana para dactilología japonesa',
            'symbols': list('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'),
            'extras': list('がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっゎゔー'),
            'note': '46 kana básicos. Puedes añadir variantes sonoras, kana pequeños y la marca de vocal larga. Este catálogo no representa todos los kanji ni asigna un gesto distinto a cada forma escrita.',
            'source': 'https://www.city.nara.lg.jp/soshiki/34/184979.html'},
    'csl': {'title': 'Letras para deletreo chino', 'symbols': LATIN + ['ZH', 'CH', 'SH', 'NG'],
            'extras': ['Ü'], 'note': 'Catálogo de letras de deletreo: A–Z y ZH, CH, SH, NG. El mandarín usa caracteres; estas etiquetas sirven para recolectar deletreo basado en pinyin, no todos los hanzi. Ü se ofrece como variante escrita opcional.',
            'source': 'https://www.moe.gov.cn/jyb_sjzl/ziliao/A19/201404/t20140422_167548.html'},
    'lsf': {'title': 'Abecedario francés', 'symbols': LATIN,
            'extras': list('ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ'),
            'note': '26 letras básicas. Acentos y ligaduras son variantes escritas opcionales; usa la convención de deletreo de LSF que estés aprendiendo.', 'source': ''},
    'libras': {'title': 'Abecedario portugués', 'symbols': LATIN,
            'extras': list('ÁÀÂÃÉÊÍÓÔÕÚÇ'),
            'note': '26 letras, incluidas K, W y Y. Acentos y Ç son variantes escritas opcionales; graba su representación en Libras.', 'source': ''},
}


def alphabet_for(language):
    language_info(language)
    return {**CATALOGS[language], 'symbols': list(CATALOGS[language]['symbols']), 'extras': list(CATALOGS[language]['extras'])}


def find_symbol_label(labels, symbol):
    # lower conserva ß distinto de SS y mantiene los diacríticos.
    return next((row for row in labels if row['text'].lower() == symbol.lower()), None)


def alphabet_status(language, rows, model):
    catalog = alphabet_for(language)
    trained = {label['id'] for label in (model or {}).get('labels', [])}
    def entries(symbols):
        result = []
        for symbol in symbols:
            row = find_symbol_label(rows, symbol)
            result.append({'symbol': symbol, 'label_id': row['id'] if row else None,
                           'samples': row['samples'] if row else 0, 'trained': bool(row and row['id'] in trained)})
        return result
    return {**catalog, 'symbols': entries(catalog['symbols']), 'extras': entries(catalog['extras'])}


def add_alphabet(language, extras=False):
    if not isinstance(extras, bool):
        raise ValueError('La opción de variantes debe ser verdadera o falsa.')
    catalog = alphabet_for(language)
    symbols = catalog['extras'] if extras else catalog['symbols']
    before = {row['id'] for row in labels_for(language)}
    labels = [register_label(language, symbol) for symbol in symbols]
    return {'language': language, 'added': sum(label['id'] not in before for label in labels), 'total': len(labels)}


def add_all_alphabets():
    return [add_alphabet(language['id']) for language in LANGUAGES]
