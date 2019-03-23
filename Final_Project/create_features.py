import csv
import os
import json
from collections import OrderedDict
import subprocess as sp
from conllu import parse, parse_tree

UDPIPE = "udpipe/src/udpipe"
PATH = "udpipe/src/ru.udpipe"

def map_roles(role):
    if role:
        role = role.strip()
    for ex_role in roles:
        tmp = ex_role.strip("/")
        if role == ex_role or role == tmp[0].strip() or len(tmp) > 1 and role == tmp[1].strip():
            return roles[role]
    return role


def cleansed_text(fname):
    with open(fname) as f_in:
        contents = f_in.read()
        text = re.sub('<[^>]*>', '', contents)
        with open(TEXT_PATH, 'a') as f_out:
            f_out.write(text + '\n')                


def process_conllu(inp):
    tree = parse_tree(inp)
    root = tree[0]
    data = {}
    #path = 0
    for const in depth_first(root):
        #print(const)
        w = const.token['form']
        deprel = const.token['deprel']
        data[w] = deprel
    return data


def depth_first(node):
    yield node
    for child in node.children:
        for n in depth_first(child):
            yield n


def features(ex):
    args = OrderedDict()
    # lex+, pos+, gram+, prev_gr+, prev_lex+, path from predicate+, 
    # syntrel with parent+, lemma of a predicate+, preposition+,
    # embedding for a predicate, embedding for an argument
    lex, pos, gram, prev_gr, prev_lex, rel, pred_lemma = [None]*7
    cl = 'noclass'
    sent = ' '.join(x for x in ex)
    p1 = sp.Popen(["echo", sent], stdout=sp.PIPE)
    p2 = sp.Popen([UDPIPE, "--tokenize", "--tag", "--parse", 
                   PATH], stdin=p1.stdout, stdout=sp.PIPE)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    output = p2.communicate()[0].decode('utf-8')
    if not output.strip():
        return []
    synt_data = process_conllu(output)
    n_word = 0
    for w in ex:
        try:
            rel = synt_data[w]
        except KeyError:
            rel = None
        if n_word == 0:
            prev_word, prev_gr, prev_lex = None, None, None
        instance = ex[w]
        lex, gr, sem, sem2, role, rank = instance
        try:
            pos, gram = gr.split(' ', maxsplit=1)
        except:
            try:
                pos, gram = gr.split(',', maxsplit=1)
            except:
                pos, gram = gr, None
        if prev_word:
            prev_lex, prev_gr = prev_word[0:2]
        """
        if instance[-1] is None and instance[-2] is None:
            # cl = 'noclass'
            continue
        else:
            cl = instance[-1]
        """
        if instance[-2]:
            cl = map_roles(role)
        else:
            continue
        d = [lex, pos, gram, prev_gr, prev_lex, rel, pred_lemma, cl]
        args[w] = d
        prev_word = instance
        #print(w, d)
        n_word += 1
    return args
    

def predicates(in_f):
    preds = {}
    with open(in_f, 'r', encoding='utf-8') as f:
        js = f.read()
    examples = json.loads(js, object_pairs_hook=OrderedDict)
    for ex in examples:
        len_words = len(examples[ex])
        n_word = 0
        for w in examples[ex]:
            if n_word == 0:
                prev_word = None
            instance = examples[ex][w]
            lex, gr, sem, sem2, role, rank = instance
            try:
                pos, gram = gr.split(' ', maxsplit=1)
            except ValueError:
                try:
                    pos, gram = gr.split(',', maxsplit=1)
                except:
                    pos, gram = gr, None
            prev_gr, prev_lex = None, None
            if prev_word:
                prev_lex, prev_gr = prev_word[0:2]
            if instance[-1] is None and instance[-2] is None:
                n_word += 1
                preds[(ex, w)] = [lex, pos, gram, sem, sem2, n_word, prev_gr, prev_lex, 0]
                prev_word = instance
                continue
            if instance[-1] == 'Предикат':
                preds[(ex, w)] = [lex, pos, gram, sem, sem2, n_word, prev_gr, prev_lex, 1]
                prev_word = instance
            n_word += 1
    with open('predicates.csv', 'w', encoding='utf-8') as p:
        header = ('ExID', 'Wordform', 'Lex', 'POS', 'Gram', 'Sem', 'Sem2', 'WordID', 'prev_lex', 'prev_gr', 'Predicate')
        p.write('\t'.join(header) + '\n')
        for pr in preds:
            p.write('\t'.join(list(pr) + [str(x) for x in preds[pr]]) + '\n')

            
def arguments(in_f):
    args = OrderedDict()
    with open(in_f, 'r', encoding='utf-8') as f:
        js = f.read()
    examples = json.loads(js, object_pairs_hook=OrderedDict)
    for ex in examples:
        len_words = len(examples[ex])
        n_word = 0
        for w in examples[ex]:
            if n_word == 0:
                prev_word = None
            instance = examples[ex][w]
            lex, gr, sem, sem2, role, rank = instance
            try:
                pos, gram = gr.split(' ', maxsplit=1)
            except ValueError:
                try:
                    pos, gram = gr.split(',', maxsplit=1)
                except:
                    pos, gram = gr, None
            prev_gr, prev_lex = None, None
            if prev_word:
                prev_lex, prev_gr = prev_word[0:2]
            if instance[-1] is None and instance[-2] is None:
                n_word += 1
                args[(ex, w)] = [lex, pos, gram, sem, sem2, n_word, prev_gr, prev_lex, '0']
                prev_word = instance
                continue
            if instance[-1] != 'Предикат' and instance[-2] != '-' and instance[-2] is not None and instance[-2] != '?':
                role = map_roles(role)
                args[(ex, w)] = [lex, pos, gram, sem, sem2, n_word, prev_gr, prev_lex, '1']
                prev_word = instance
            n_word += 1
    with open('arguments.csv', 'w', encoding='utf-8') as p:
        writer = csv.writer(p, delimiter='\t')
        header = ('ExID', 'Wordform', 'Lex', 'POS', 'Gram', 'Sem', 'Sem2', 'WordID', 'prev_gr', 'prev_lex', 'Class')
        writer.writerow(header)
        for a in args:
            row = list(a) + [str(x) for x in args[a]]
            writer.writerow(row)


def together(in_f):
    args = {}
    with open(in_f, 'r', encoding='utf-8') as f:
        js = f.read()
    examples = json.loads(js, object_pairs_hook=OrderedDict)
    for ex in examples:
        len_words = len(examples[ex])
        n_word = 0
        for w in examples[ex]:
            if n_word == 0:
                prev_word = None
            instance = examples[ex][w]
            lex, gr, sem, sem2, role, rank = instance
            try:
                pos, gram = gr.split(' ', maxsplit=1)
            except ValueError:
                try:
                    pos, gram = gr.split(',', maxsplit=1)
                except:
                    pos, gram = gr, None
            prev_gr, prev_lex = None, None
            if prev_word:
                prev_lex, prev_gr = prev_word[0:2]
            if instance[-1] is None and instance[-2] is None:
                n_word += 1
                args[(ex, w)] = [lex, pos, gram, sem, sem2, n_word, prev_gr, prev_lex, 'noclass']
                prev_word = instance
                continue
            if instance[-1] == 'Предикат':
                args[(ex, w)] = [lex, pos, gram, sem, sem2, n_word, prev_gr, prev_lex, rank]
                prev_word = instance
                continue
            if instance[-1] != 'Предикат' and instance[-2] != '-':
                role = map_roles(role)
                args[(ex, w)] = [lex, pos, gram, sem, sem2, n_word, prev_gr, prev_lex, role]
                prev_word = instance
            n_word += 1
    with open('arguments_predicates.csv', 'w', encoding='utf-8') as p:
        header = ('ExID', 'Wordform', 'Lex', 'POS', 'Gram', 'Sem', 'Sem2', 'WordID', 'prev_lex', 'prev_gr', 'Class')
        p.write('\t'.join(header) + '\n')
        for a in args:
            p.write('\t'.join(list(a) + [str(x) for x in args[a]]) + '\n')

            
def arg_pred(in_f, out_f):
    args = OrderedDict()
    with open(in_f, 'r', encoding='utf-8') as f:
        js = f.read()
    examples = json.loads(js, object_pairs_hook=OrderedDict)
    with open(out_f, 'w', encoding='utf-8') as p:
        header = ('word', 'lex', 'pos', 'gram', 'prev_gr', 'prev_lex', 'rel', 'pred_lemma', 'class')
        p.write('\t'.join(header) + '\n')
    with open(out_f, 'a', encoding='utf-8') as p:
        for ex in examples:
            args = features(examples[ex])
            for a in args:
                p.write('\t'.join([a] + [str(x) for x in args[a]]) + '\n')
    

if __name__ == '__main__':
    roles = {}
    with open('Roles.csv', 'r', encoding='utf-8') as r:
        reader = csv.reader(r, delimiter=',')
        header = next(reader)
        for row in reader:
            roles[row[0]] = row[2]
    #predicates('parsed_framebank_roles_big.json')
    arg_pred('parsed_framebank_roles_big.json', 'roles_syntax_roles.csv')
    #arguments('parsed_framebank_roles_big.json')
    #together('parsed_framebank_roles.json')
