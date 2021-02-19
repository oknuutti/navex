import copy
from argparse import ArgumentParser, Namespace

import yaml

from ..models import MODELS


class ExperimentConfigParser(ArgumentParser):
    def __init__(self, config_file_param='config', definition=None):
        super(ExperimentConfigParser, self).__init__(description='Train a local feature detector and descriptor')
        self.add_argument('--' + config_file_param.replace('_', '-'), metavar='FILE',
                          help='path to experiment config file')
        self.config_file_param = config_file_param
        self.groups = {}

        if definition is not None:
            with open(definition, 'r') as fh:
                self.definition = yaml.safe_load(fh)

            def _add_subtree(obj, lv, path, d):
                for k, v in d.items():
                    name = (path+'__'+k) if path else k
                    if not isinstance(v, dict):
                        pass
                    elif v.pop('__group__', False):
                        group = obj.add_argument_group(k) if lv == 0 else obj
                        self.groups[k] = group
                        _add_subtree(group, lv+1, name, v)
                    else:
                        alt = v.pop('alt', [])
                        if 'type' in v:
                            v['type'] = eval(v['type'])

                        help = v.pop('help', '')
                        if len(help) > 2 and help[0] == '`' and help[-1] == '`':
                            help = eval(help[1:-1])

                        default = v.pop('default', None)
                        if default is not None:
                            help += " (default: %s)" % (default,)

                        v.pop('trial', None)
                        obj.add_argument('--' + name.replace('_', '-'), *alt, dest=name,
                                         default=default, help=help, **v)

            _add_subtree(self, 0, '', copy.deepcopy(self.definition))
        else:
            self.definition = None

    def parse_args(self, args=None, namespace=None):
        raw_args = super(ExperimentConfigParser, self).parse_args(args=args, namespace=namespace)

        def hyp_constr(l, s, n):
            t = s.split('_')
            cls = dict([(c.__name__, c) for c in (bool, int, float, str)] + [('tune', 'tune')]).get(t[0], None)
            assert cls is not None, 'Unknown hyperparameter class: "%s"' % s

            if isinstance(cls, str) and cls[:4] == 'tune':
                from ray import tune
                double = cls == 'tune2'
                cls = getattr(tune, '_'.join(t[1:]))

                if double:
                    from ..ray.base import DoubleSampler
                    cls = DoubleSampler(cls)

            return HyperParam(n.value, cls)

        yaml.add_multi_constructor('!h_', hyp_constr, yaml.loader.SafeLoader)

        with open(getattr(raw_args, self.config_file_param), 'r') as fh:
            args = yaml.safe_load(fh)

        trial = args['training'].get('trial', None)
        assert trial is not None, 'Required training.trial parameter not given'

        for key, val in raw_args.__dict__.items():
            if val is None:
                continue

            parts = key.split('__')
            f = args
            for i, p in enumerate(parts):
                if i == len(parts) - 1:
                    if p in f and isinstance(f[p], HyperParam):
                        f[p].value = val
                    else:
                        f[p] = val
                else:
                    if p not in f:
                        f[p] = {}
                    f = f[p]

        if self.definition is not None:
            def _check_subtree(path, a, d):
                ak = set(a.keys())
                dk = {k for k, v in d.items() if k != '__group__' and isinstance(v, dict)
                                                 and trial in v.get('trial', [trial])}
                assert dk.issubset(ak), "Keys %s not found at '%s'" % (dk - ak, path)
                for k in dk:
                    v = d[k]
                    if v.get('__group__', False):
                        _check_subtree((path+'.'+k) if path else k, a[k], v)
            _check_subtree('', args, self.definition)

        hparam_paths = []
        f_hparams = {}
        def _hp_subtree(hparam_paths, hparams, path, a):
            for k, v in a.items():
                path_k = (path+'.'+k) if path else k
                if isinstance(v, HyperParam):
                    hparam_paths.append(path_k)
                    hparams[k] = v.value
                    a[k] = v.value
                elif isinstance(v, dict):
                    hparams[k] = {}
                    _hp_subtree(hparam_paths, hparams[k], path_k, v)
        _hp_subtree(hparam_paths, f_hparams, '', args)

        hparams = {}
        def _prune_subtree(pruned, path, full):
            keep_p = False
            for k, v in full.items():
                if isinstance(v, dict):
                    if len(v) > 0:
                        path[k] = {}
                        keep_c = _prune_subtree(pruned, path[k], v)
                        if keep_c:
                            pruned[k] = path[k]
                            keep_p = True
                else:
                    path[k] = v
                    keep_p = True
            return keep_p
        _prune_subtree(hparams, {}, f_hparams)

        if not 'hparams' in args:
            args['hparams'] = hparams

        return to_namespace(args)


class HyperParam:
    def __init__(self, value, cls):
        self.cls = cls
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.cls.__module__[:9] in ('ray.tune.', 'navex.ray'):
            import random
            from ray import tune
            import numpy as np
            val = eval('(%s,)' % value, {'random': random, 'tune': tune, 'np': np})
            self._value = self.cls(*val)
        else:
            self._value = self.cls(value)

    def __str__(self):
        return "hparam %s(%s)" % (self.cls.__name__, self.value)


def flatten_dict(nested, sep='.'):
    flat = {}
    def _hp_subtree(f, path, n):
        for k, v in n.items():
            path_k = (path + sep + k) if path else k
            if isinstance(v, dict):
                _hp_subtree(f, path_k, v)
            else:
                f[path_k] = v
    _hp_subtree(flat, '', nested)
    return flat


def set_nested(nested, key, value, sep='.'):
    t = key.split(sep)
    n = nested
    for i, k in enumerate(t):
        if i < len(t)-1:
            n = n[k]
        else:
            n[k] = value


def nested_update(orig, new):
    def _upd_subtree(o, n):
        for k, v in n.items():
            if isinstance(v, dict):
                if k not in o:
                    o[k] = {}
                _upd_subtree(o[k], v)
            else:
                o[k] = v
    _upd_subtree(orig, new)


def nested_filter(tree, test_fn, mod_fn):
    def _filter_subtree(o, n, test_fn, mod_fn):
        for k, v in o.items():
            if isinstance(v, dict):
                if k not in n:
                    n[k] = {}
                _filter_subtree(v, n[k], test_fn, mod_fn)
            elif test_fn(v):
                n[k] = mod_fn(v)
    new_tree = {}
    _filter_subtree(tree, new_tree, test_fn, mod_fn)
    return new_tree


def prune_nested(tree):
    def _prune_subtree(o):
        n = {}
        for k, v in o.items():
            if isinstance(v, dict):
                if len(v) == 0:
                    ns = {}
                else:
                    ns = _prune_subtree(v)
                if len(ns) > 0:
                    n[k] = ns
            else:
                n[k] = v
        return n

    new_tree = _prune_subtree(tree)
    return new_tree


def to_dict(ns):
    def _conv_subtree(from_node, to_node):
        for k, v in (from_node if isinstance(from_node, dict) else from_node.__dict__).items():
            if isinstance(v, (Namespace, dict)):
                to_node[k] = {}
                _conv_subtree(v, to_node[k])
            else:
                to_node[k] = v

    mapping = {}
    _conv_subtree(ns, mapping)
    return mapping


def to_namespace(mapping):
    def _conv_subtree(from_node, to_node):
        for k, v in from_node.items():
            if isinstance(v, dict):
                ns = Namespace()
                setattr(to_node, k, ns)
                _conv_subtree(v, ns)
            else:
                setattr(to_node, k, v)

    ns = Namespace()
    _conv_subtree(mapping, ns)
    return ns
