import re
import inspect

import numpy as np

class Input(object):
    def __init__(self, name, obj, vocab):
        self.name = name
        self.obj = obj
        self.vocab = vocab
        self.rules = []

class Output(object):
    def __init__(self, name, obj, vocab):
        self.name = name
        self.obj = obj
        self.vocab = vocab
        self.rules = []
        
        

class Rule(object):
    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.matches = {}
        self.effects = {}

        code = inspect.getsource(function)
        m = re.match(r'[^(]+\([^(]*\):',code)
        code = 'if True:'+code[m.end():]
        self.rule = compile(code, '<production-%s>'%self.name, 'exec')


    def match(self, *args, **kwargs):
        if len(args)>0: raise Exception('invalid match in rule "%s"'%self.name)
        for k, v in kwargs.iteritems():
            if k not in self.inputs:
                raise Exception('No module named "%s" found for match in rule "%s"'%(k, self.name))
            assert k not in self.matches    
            self.matches[k] = v     
        
    def effect(self, *args, **kwargs):
        if len(args)>0: raise Exception('invalid effect in rule "%s"'%self.name)
        for k, v in kwargs.iteritems():
            if k not in self.outputs:
                raise Exception('No module named "%s" found for effect in rule "%s"'%(k, self.name))
            assert k not in self.effects    
            self.effects[k] = v     
        
    def process(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
    
        globals = dict(match=self.match, effect=self.effect)
        globals.update(inputs)
    
        eval(self.rule, {}, globals)
    
        
    
        
            
        


class Rules(object):
    def __init__(self, rules):
        self.rules = []
        for name,func in inspect.getmembers(rules):
            if inspect.ismethod(func):
                if not name.startswith('__'):
                    self.rules.append(Rule(name, func))
                    
    @property
    def count(self):
        return len(self.rules)                
        
    def process(self, spa):
        self.inputs = {}
        self.outputs = {}
        for name, m in spa.modules.iteritems():
            for label, (obj, vocab) in m.outputs.iteritems():
                n = name
                if label!='default':
                    n+='_'+label
                self.inputs[n] = Input(n, obj, vocab)
            for label, (obj, vocab) in m.inputs.iteritems():
                n = name
                if label!='default':
                    n+='_'+label
                self.outputs[n] = Output(n, obj, vocab)
            
        for rule in self.rules:
            rule.process(self.inputs, self.outputs)
            
    def get_inputs(self):
        inputs = {}
    
        for name, input in self.inputs.iteritems():
            transform = []
            assert input.vocab is not None
            
            for rule in self.rules:
                if name in rule.matches:
                    row = input.vocab.parse(rule.matches[name]).v
                else:
                    row = [0]*input.vocab.dimensions
                transform.append(row)    
            transform = np.array(transform)        
            if np.count_nonzero(transform)>0:
                inputs[input.obj] = transform
        return inputs        

    def get_outputs(self):
        inputs = {}
    
        for name, output in self.outputs.iteritems():
            transform = []
            assert output.vocab is not None
            
            for rule in self.rules:
                if name in rule.effects:
                    row = output.vocab.parse(rule.effects[name]).v
                else:
                    row = [0]*output.vocab.dimensions
                transform.append(row)    
            transform = np.array(transform)        
            if np.count_nonzero(transform)>0:
                inputs[output.obj] = transform.T
        return inputs        

            
        
    @property    
    def names(self):
        return [r.name for r in self.rules]
