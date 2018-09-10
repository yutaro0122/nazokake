#!/usr/bin/env python
# coding: utf-8

import MeCab
import re
import codecs
import math
from gensim.models import KeyedVectors
import gzip

MIN_HOMONYMS_COUNT = 4
MIN_PAGEVIEWS = 499
MAX_IDF = 10.
THRESHOLD = .26

class Parser():
    def __init__(self):
        self.tagger = MeCab.Tagger()
        self.whitelist = [u'名詞', u'形容詞', u'動詞']

    def make_entity_pair(self, entity, sentence):
        words = set()
        sentence = sentence.encode('utf-8')
        parsed = self.tagger.parse(sentence)
        if parsed is None:
            return []
        for chunk in parsed.splitlines()[:-1]:
            surface, others = chunk.split('\t')
            others = others.split(',')
            pos, pronunciation = others[0], others[-1]
            surface, pos, pronunciation = surface.decode('utf-8'), pos.decode('utf-8'), pronunciation.decode('utf-8')
            if pronunciation == "*": continue
            if any([pos.startswith(x) for x in self.whitelist]):
                words.add((entity, surface, pronunciation, pos))
        return list(words)

def write_idf():
    df = {}
    D = set()
    with codecs.open('related_words.txt', 'r', 'utf-8') as rf:
        for line in rf:
            entity, word, pronun, pos = line.strip().rsplit('\t', 3)
            if entity not in D:
                D.add(entity)
                if len(D) % 10000 == 0:
                    print len(D)
            key = word + '\t' + pronun
            if key in df:
                df[key] += 1
            else:
                df[key] = 1
    len_D = len(D) * 1.0
    idf = {k: math.log(len_D / v) for k, v in df.items()}
    with codecs.open('idf.tsv', 'w', 'utf-8') as wf:
        wf.write(
            '\n'.join([k + '\t' + str(v) for k, v in sorted(idf.items(), key=lambda x: x[1])])
        + '\n'
        )

def load_entity_index():
    entity_of = {}
    with codecs.open('../data/entity_index.txt', 'r', 'utf-8') as rf:
        for line in rf:
            line = line.strip()
            if line.endswith('0'):
                entity_id, entity, _ = line.split('\t', 2)
                entity_of[entity_id] = entity
    return entity_of

def write_pairs():
    with codecs.open('../data/source.txt', 'r', 'utf-8') as rf, codecs.open('related_words.txt', 'w', 'utf-8') as wf:
        r = re.compile(r'^#\d+$')
        entity = ''
        pairs = []  # such like [[(entity, surface, pronunciation), (...), ..., (...)], [(entity, ...), ...]
        i = 1
        for line in rf:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith('#'):
                if r.match(line) is not None:  # if the row is entity id
                    entity = entity_of[line[1:]] if line[1:] in entity_of else ''
                    continue
            if len(entity) > 0:
                pairs.append(parser.make_entity_pair(entity, line))
            if len(pairs) > 0 and len(pairs) % 10000 == 0:
                wf.write('\n'.join(['\t'.join(words) for pair in pairs for words in pair]) + '\n')
                print i * len(pairs)
                i += 1
                pairs = []
        if len(pairs) > 0:
            wf.write('\n'.join(['\t'.join(words) for pair in pairs for words in pair]) + '\n')
            pairs = []

def get_homonyms():
    pronunciation2surface = {}
    with codecs.open('related_words.txt', 'r', 'utf-8') as rf:
        for line in rf:
            _, surface, pronunciation, _ = line.strip().split('\t')
            if pronunciation in pronunciation2surface:
                pronunciation2surface[pronunciation].add(surface)
            else:
                pronunciation2surface[pronunciation] = set([surface])
    with codecs.open('pronunciation2surface.txt', 'w', 'utf-8') as wf:
        wf.write('\n'.join([k + '\t' + '\t'.join(v) for k, v in pronunciation2surface.items()]) + '\n')
    return {k: v for k, v in pronunciation2surface.items() if len(v) > MIN_HOMONYMS_COUNT}

def nazokake(a):
    max_score = 0
    answers = []
    # 1. search candidates of C from input A (HIGHER similarity is better)
    if a not in entity2surface:
        return [(a, '-', '-', '-', '-', 0)]
    c_list = entity2surface[a]
    for tmp_c, tmp_pronunciation in c_list:
        if a == tmp_c:
            continue
        if a not in word2vec or tmp_c not in word2vec:
            continue
        a2c_sim = abs(word2vec.wv.similarity(a, tmp_c))
        # 2. search candidates of C' which is same pronunciation as C (Lower similarity is better)
        tmp_c__list = pronunciation2surface[tmp_pronunciation]
        for tmp_c_ in tmp_c__list:
            if tmp_c == tmp_c_:
                continue
            if tmp_c_ not in word2vec or tmp_c_ not in surface2entity:
                continue
            c2c_sim = abs(word2vec.wv.similarity(tmp_c, tmp_c_))
            # 3. search candidates of B which has C' as a related word (HIGHER similarity is better)
            tmp_b_list = surface2entity[tmp_c_]
            for tmp_b in tmp_b_list:
                if tmp_b == tmp_c_:
                    continue
                if tmp_b not in word2vec:
                    continue
                b2c__sim = abs(word2vec.wv.similarity(tmp_b, tmp_c_))
                score = math.log(a2c_sim) - math.log(c2c_sim) + math.log(b2c__sim) 
                # if max_score == 0 or score > max_score:
                #     b, c, c_, pronunciation = tmp_b, tmp_c, tmp_c_, tmp_pronunciation
                answers.append((a, tmp_b, tmp_c, tmp_c_, tmp_pronunciation, score))
    # 4. output A, B, C and C'
    # return a, b, c, c_, pronunciation, score
    return sorted(answers, key=lambda x: x[5], reverse=True)[:5]

if __name__=="__main__":
    ''' prepare stuff '''
    entity_of = load_entity_index()
    parser = Parser()
    # write_pairs()
    # write_idf()

    word2vec = KeyedVectors.load_word2vec_format('../data/word2vec.bin', binary=True, unicode_errors='ignore')

    # popular_pronunciations = {}  # pick popular pronunciations 
    # with codecs.open('idf.tsv', 'r', 'utf-8') as rf:
    #     for i, line in enumerate(rf):
    #         surface, pronunciation, idf_score = line.strip().split('\t')
    #         idf_score = float(idf_score)
    #         if idf_score > MAX_IDF:
    #             break
    #         else:
    #             popular_pronunciations[(pronunciation, surface)] = float(idf_score)

    # take homonymy surfaces as C/C'
    pronunciation2surface = get_homonyms()

    # take popular entities as A/B
    pageviews_of = {}
    with gzip.open('../data/pageviews.txt.gz', 'r', 'utf-8') as rf:
        for line in rf:
            _, entity, view = line.decode('utf-8').strip().split('\t')
            entity, view = entity.strip().replace(' ', '_'), int(view)
            pageviews_of[entity] = view
    entities = [k for k, v in pageviews_of.items() if v > MIN_PAGEVIEWS]

    entity2surface = {}
    surface2entity = {}
    with codecs.open('related_words.txt', 'r', 'utf-8') as rf:
        # 189334808 lines 
        for line in rf:
            entity, surface, pronunciation, pos = line.strip().split('\t', 3)
            if pronunciation not in pronunciation2surface or entity not in entities or pos != u'名詞':
                continue
            if entity in word2vec and surface in word2vec:
                # higher similarity between A and C (B and C') is better
                # A/B is an entity, C/C' is surface 
                sim = word2vec.wv.similarity(entity, surface)
                if sim > THRESHOLD:
                    if entity in entity2surface:
                        entity2surface[entity].append((surface, pronunciation))
                    else:
                        entity2surface[entity] = [(surface, pronunciation)]
                        if len(entity2surface) % 100 == 0:
                            print len(entity2surface)
                    if surface in surface2entity:
                        surface2entity[surface].append((entity))
                    else:
                        surface2entity[surface] = [(entity)]
    with codecs.open('entity2surface.txt', 'w', 'utf-8') as wf:
        wf.write(
            '\n'.join(
                [k + '\t' + '\t'.join([x + '-' + y for x, y in v]) for k, v in entity2surface.items()]
                ) + '\n'
            )
    with codecs.open('surface2entity.txt', 'w', 'utf-8') as wf:
        wf.write(
            '\n'.join(
                [k + '\t' + '\t'.join(v) for k, v in surface2entity.items()]
                ) + '\n'
            )
