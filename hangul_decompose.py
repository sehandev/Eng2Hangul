# -*- coding: utf-8 -*-

from __future__ import unicode_literals

CHO = (
    u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

JOONG = (
    u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅗ', u'ㅘ',
    u'ㅙ', u'ㅚ', u'ㅛ', u'ㅜ', u'ㅝ', u'ㅞ', u'ㅟ', u'ㅠ', u'ㅡ', u'ㅢ', u'ㅣ'
)

JONG = (
    u'', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ', u'ㄺ',
    u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

JAMO = CHO + JOONG + JONG[1:]

NUM_CHO = len(CHO)
NUM_JOONG = len(JOONG)
NUM_JONG = len(JONG)

FIRST_HANGUL_UNICODE = 0xAC00  # '가'
LAST_HANGUL_UNICODE = 0xD7A3  # '힣'


# phrase가 한글인지 아닌지 확인
def is_hangul(phrase): 
    for letter in phrase:
        code = ord(letter)
        if (code < FIRST_HANGUL_UNICODE or code > LAST_HANGUL_UNICODE) and (letter not in JAMO):
            return False
    return True

def combine(cho,jung,jong):
  return chr((cho*21+jung)+jong)

def decompose_index(code):
    jong = int(code % NUM_JONG)
    code /= NUM_JONG
    joong = int(code % NUM_JOONG)
    code /= NUM_JOONG
    cho = int(code)

    return cho, joong, jong


# 한글 한 글자가 hangul_letter로 들어오면 초성, 중성, 종성을 분리해서 return
def decompose(hangul_letter):
    if len(hangul_letter) < 1:
        raise NotLetterException('')
    elif not is_hangul(hangul_letter):
        raise NotHangulException('')

    code = ord(hangul_letter) - FIRST_HANGUL_UNICODE
    cho, joong, jong = decompose_index(code)

    try:
        return CHO[cho], JOONG[joong], JONG[jong]
    except:
        print("%d / %d  / %d" % (cho, joong, jong))
        print("%s / %s " % (JOONG[joong].encode("utf8"), JONG[jong].encode('utf8')))
        raise Exception()
