'''
Created on 2014-07-30

Random utility functions...

@author: Andre R. Erler, GPL v3
'''

# named exception
class ArgumentError(Exception):
  """Exception indicating an Error with the HGS Ensemble."""
  pass

## define function for recursion 
# basically, loop over each list independently
def _loop_recursion(*args, **kwargs):
  ''' handle any number of loop variables recursively '''
  # interpete arguments (kw-expansion is necessary)
  if len(args) == 1:
    # initialize dictionary of lists (only first recursion level)
    loop_list = args[0][:] # use copy, since it will be decimated 
    list_dict = {key:list() for key in kwargs.keys()}
  elif len(args) == 2:
    loop_list = args[0][:] # use copy of list, to avoid interference with other branches
    list_dict = args[1] # this is not a copy: all branches append to the same lists!
  # handle loops
  if len(loop_list) > 0:
    # initiate a new recursion layer and a new loop
    arg_name = loop_list.pop(0)
    for arg in kwargs[arg_name]:
      kwargs[arg_name] = arg # just overwrite
      # new recursion branch
      list_dict = _loop_recursion(loop_list, list_dict, **kwargs)
  else:
    # terminate recursion branch
    for key,value in kwargs.items():
      list_dict[key].append(value)
  # return results 
  return list_dict

# helper function to check lists
def _prepareList(exp_list, kwargs):
  ''' helper function to clean list elements '''
  # get exp_list arguments
  exp_list = [el for el in exp_list if el in kwargs] # remove missing entries
  exp_dict = {el:kwargs[el] for el in exp_list}
  for el in exp_list: del kwargs[el]
  for el in exp_list: # check types 
    if not isinstance(exp_dict[el], (list,tuple)): raise TypeError(el)
  return exp_list, exp_dict


# helper function to form inner and outer product of multiple lists
def expandArgumentList(inner_list=None, outer_list=None, expand_list=None, lproduct='outer', **kwargs):
  ''' A function that generates a list of complete argument dict's, based on given kwargs and certain 
      expansion rules: kwargs listed in expand_list are expanded and distributed element-wise, 
      either as inner ('inner_list') or outer ('outer_list') product, while other kwargs are repeated 
      in every argument dict. 
      Arguments can be expanded simultaneously (in parallel) within an outer product by specifying
      them as a tuple within the outer product argument list ('outer_list'). '''
  if not (expand_list or inner_list or outer_list): 
    arg_dicts = [kwargs] # return immediately - nothing to do
  else:
      
    # handle legacy arguments
    if expand_list is not None:
      if inner_list is not None or outer_list is not None: raise ArgumentError("Can not mix input modes!")
      if lproduct.lower() == 'inner': inner_list = expand_list
      elif lproduct.lower() == 'outer': outer_list = expand_list
      else: raise ArgumentError(lproduct)
    outer_list = outer_list or []; inner_list = inner_list or []
      
    # handle outer product expansion first
    if len(outer_list) > 0:
      kwtmp = {key:value for key,value in kwargs.items() if key not in inner_list}
      
      # detect variables for parallel expansion
      # N.B.: parallel outer expansion is handled by replacing the arguments in each parallel expansion group
      #       with a single (fake) argument that is a tuple of the original argument values; the tuple is then,
      #       after expansion, disassembled into its former constituent arguments
      par_dict = dict()
      for kw in outer_list:
        if isinstance(kw,(tuple,list)):
          # retrieve parallel expansion group 
          par_args = [kwtmp.pop(name) for name in kw]
          if not all([len(args) == len(par_args[0]) for args in par_args]): 
            raise ArgumentError("Lists for parallel expansion arguments have to be of same length!")
          # introduce fake argument and save record
          fake = 'TMP_'+'_'.join(kw)+'_{:d}'.format(len(kw)) # long name that is unlikely to interfere...
          par_dict[fake] = kw # store record of parallel expansion for reassembly later
          kwtmp[fake] = list(zip(*par_args)) # transpose lists to get a list of tuples                      
        elif not isinstance(kw,str): raise TypeError(kw)
      # replace entries in outer list
      if len(par_dict)>0:
        outer_list = outer_list[:] # copy list
        for fake,names in par_dict.items():
          if names in outer_list:
            outer_list[outer_list.index(names)] = fake
      assert all([ isinstance(arg,str) for arg in outer_list])
      
      outer_list, outer_dict = _prepareList(outer_list, kwtmp)
      lstlen = 1
      for el in outer_list:
        lstlen *= len(outer_dict[el])
      # execute recursive function for outer product expansion    
      list_dict = _loop_recursion(outer_list, **outer_dict) # use copy of
      # N.B.: returns a dictionary where all kwargs have been expanded to lists of appropriate length
      assert all(key in outer_dict for key in list_dict.keys()) 
      assert all(len(list_dict[el])==lstlen for el in outer_list) # check length    
      assert all(len(ld)==lstlen for ld in list_dict.values()) # check length  
      
      # disassemble parallel expansion tuple and reassemble as individual arguments
      if len(par_dict)>0:
        for fake,names in par_dict.items():
          assert fake in list_dict
          par_args = list(zip(*list_dict.pop(fake))) # transpose, to get an expanded tuple for each argument
          assert len(par_args) == len(names) 
          for name,args in zip(names,par_args): list_dict[name] = args
         
    # handle inner product expansion last
    if len(inner_list) > 0:
      kwtmp = kwargs.copy()
      if len(outer_list) > 0: 
        kwtmp.update(list_dict)
        inner_list = outer_list + inner_list
      # N.B.: this replaces all outer expansion arguments with lists of appropriate length for inner expansion
      inner_list, inner_dict = _prepareList(inner_list, kwtmp)
      # inner product: essentially no expansion
      lst0 = inner_dict[inner_list[0]]; lstlen = len(lst0) 
      for el in inner_list: # check length
        if len(inner_dict[el]) == 1: 
          inner_dict[el] = inner_dict[el]*lstlen # broadcast singleton list
        elif len(inner_dict[el]) != lstlen: 
          raise TypeError('Lists have to be of same length to form inner product!')
      list_dict = inner_dict
      
    ## generate list of argument dicts
    arg_dicts = []
    for n in range(lstlen):
      # assemble arguments
      lstargs = {key:lst[n] for key,lst in list_dict.items()}
      arg_dict = kwargs.copy(); arg_dict.update(lstargs)
      arg_dicts.append(arg_dict)    
  # return list of arguments
  return arg_dicts


# decorator class for batch-loading datasets into an ensemble using a custom load function
class BatchLoad(object):
  ''' A decorator class that wraps custom functions to load specific datasets. List arguments can be
      expanded to load multiple datasets and places them in a list or Ensemble. 
      Keyword arguments are passed on to the dataset load functions; arguments listed in load_list 
      are applied to the datasets according to expansion rules, otherwise they are applied to all. '''
  
  def __init__(self, load_fct):
    ''' initialize wrapping of original operation '''
    self.load_fct = load_fct
    
  def __call__(self, load_list=None, lproduct='outer', inner_list=None, outer_list=None, 
               lensemble=None, ens_name=None, ens_title=None, **kwargs):
    ''' wrap original function: expand argument list, execute load_fct over argument list, 
        and return a list or Ensemble of datasets '''
    # decide, what to do
    if load_list is None and inner_list is None and outer_list is None:
      # normal operation: no expansion      
      datasets =  self.load_fct(**kwargs)
    else:
      # expansion required
      lensemble = ens_name is not None if lensemble is None else lensemble
      # figure out arguments
      kwargs_list = expandArgumentList(expand_list=load_list, lproduct=lproduct, 
                                       inner_list=inner_list, outer_list=outer_list, **kwargs)
      # load datasets
      datasets = []
      for kwargs in kwargs_list:    
        # load dataset
        datasets.append(self.load_fct(**kwargs))    
      # construct ensemble
      if lensemble:
        from ensemble.base import Ensemble 
        datasets = Ensemble(members=datasets, name=ens_name, title=ens_title, basetype='Dataset')
    # return list or ensemble of datasets
    return datasets

