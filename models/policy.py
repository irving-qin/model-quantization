
import os, sys

def read_policy(filename, section='init'):
    policies = []
    attr = None
    valid = False
    with open(filename) as f:
        while(True):
            line = f.readline()
            if len(line) == 0:
                break

            items = line.strip('\n').strip(' ')
            if len(items) == 0 or items[0] == "#":
                continue

            items = items.split(':')
            if len(items) < 2:
                break

            if section in items[0]:
                policies = []
                continue

            if 'by_' in items[0]:
                if attr is None:
                    attr = dict()
                elif valid:
                    policies.append(attr)
                    valid = False
                    attr = dict()

            if attr is None:
                continue

            items[0] = items[0].strip()
            items[1] = items[1].strip()
            if ',' in items[1]:
                items[1] = items[1].split(',')
            if isinstance(items[1], list):
                items[1] = [ i.strip() for i in items[1]]
            elif ' ' in items[1]:
                items[1] = items[1].split(' ')

            for i, t in enumerate(items[1]):
                if t in ['True', 'true']:
                    items[1][i] = True
                elif t in ['False', 'false']:
                    items[1][i] = False

            attr[items[0]] = items[1]
            if 'by_' not in items[0]:
                valid = True

        if attr is not None and valid:
            policies.append(attr)

    return policies

def deploy_on_init(model, filename):
    if not hasattr(model, 'modules'):
        return
    if not os.path.isfile(filename):
        return

    policies = read_policy(filename, 'init')
    for p in policies:
        attributes = p
        assert isinstance(attributes, dict), "Error attributes"
        for m in model.modules():
            if hasattr(m, 'update_quantization_parameter'):
                m.update_quantization_parameter(**attributes)

