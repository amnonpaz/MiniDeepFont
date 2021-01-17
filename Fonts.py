attributes = {
    'Skylark': {
        'color': 'r',
        'code': 0
    },
    'Sweet Puppy': {
        'color': 'b',
        'code': 1
    },
    'Ubuntu Mono': {
        'color': 'g',
        'code': 2
    }
}


def encode_name(font_name: str):
    return attributes[font_name]['code']


def decode_name(font_code: int):
    for k in attributes:
        if attributes[k]['code'] == font_code:
            return k


def get_list():
    return list(attributes.keys())


def get_number():
    return len(attributes)


def bb_color(font):
    tmp = font.decode('UTF-8')
    res = 'b'
    for font_name, data in attributes.items():
        if tmp == font_name:
            res = data['color']
            break
    return res
