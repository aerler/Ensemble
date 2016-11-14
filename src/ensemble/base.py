"""
Created on 2013-08-19

Variable and Dataset classes for handling geographical datasets.

@author: Andre R. Erler, GPL v3
"""

# numpy imports
import numpy as np

# named exception
class EnsembleError(Exception):
  """ Exception indicating an Error with the HGS Ensemble """
  pass

# a function that executes a class/instance method for use in apply_async
def apply_method(member, attr, **kwargs): 
  """ execute the method 'attr' of instance 'member' with keyword arguments 'kwargs';
      return a tuple containing member instance (possibly changed) and method results """
  return member,getattr(member, attr)(**kwargs) # returns a TUPLE!!!


## define ensemble wrapper class
class EnsembleWrapper(object):
  """ 
    A class that applies an attribute or method call on the ensemble class to all of its members
    and returns a list of the results. The ensemble class is assumed to have an attribute 
    'members', which is a list of the ensemble member.
    The EnsembleWrapper class is instantiated with the ensemble class and the called attibure or
    method in the __getattr__ method and is returned instead of the class attribute.
  """

  def __init__(self, klass, attr):
    """ the object has to be initialized with the ensemlbe class 'klass' and the called attribute
        'attr' in the __getattr__ method """
    self.klass = klass # the object that the attribute is called on
    self.attr = attr # the attribute name that is called
    
  def __call__(self, lparallel=False, NP=None, inner_list=None, outer_list=None, callback=None, **kwargs):
    """ this method is called instead of a class or instance method; it applies the arguments 
        'kwargs' to each ensemble member; it also supports argument expansion with inner and 
        outer product (prior to application to ensemble) and parallelization using multiprocessing """
    # expand kwargs to ensemble list
    kwargs_list = expandArgumentList(inner_list=inner_list, outer_list=outer_list, **kwargs)
    if len(kwargs_list) == 1: kwargs_list = kwargs_list * len(self.klass.members)
    elif len(kwargs_list) != len(self.klass.members): 
      raise ArgumentError('Length of expanded argument list does not match ensemble size! {} ~= {}'.format(
                          len(kwargs_list),len(self.klass.members)))
    # loop over ensemble members and execute function
    if lparallel:
      # parallelize method execution using multiprocessing
      pool = multiprocessing.Pool(processes=NP) # initialize worker pool
      if callback is not None and not callable(callback): raise TypeError(callback)
      # N.B.: the callback function is passed a result from the apply_method function, 
      #       which returns a tuple of the form (member, exit_code)
      # define work loads (function and its arguments) and start tasks      
      results = [pool.apply_async(apply_method, (member,self.attr), kwargs, callback=callback) 
                                      for member,kwargs in zip(self.klass.members,kwargs_list)]          
      # N.B.: Beware Pickling!!!
      pool.close(); pool.join() # wait to finish
      # retrieve and assemble results 
      results = [result.get() for result in results]
      # divide members and results (apply_method returns both, in case members were modified)
      self.klass.members = [result[0] for result in results]
      results = [result[1] for result in results]
    else:
      # get instance methods
      methods = [getattr(member,self.attr) for member in self.klass.members]
      # just apply sequentially
      results = [method(**kwargs) for method,kwargs in zip(methods,kwargs_list)]
    if len(results) != len(self.klass.members): 
      raise ArgumentError('Length of results list does not match ensemble size! {} ~= {}'.format(
                          len(results),len(self.klass.members)))
    return tuple(results)
  

class Ensemble(object):
  """
    A container class that holds several datasets ("members" of the ensemble),
    furthermore, the Ensemble class provides functionality to execute Dataset
    class methods collectively for all members, and return the results in a tuple.
  """
  members   = None    # list of members of the ensemble
  basetype  = None # base class of the ensemble members
  idkey     = 'name'  # property of members used for unique identification
  ens_name  = ''      # name of the ensemble
  ens_title = ''      # printable title used for the ensemble
  
  def __init__(self, *members, **kwargs):
    """ Initialize an ensemble from a list of members (the list arguments);
    keyword arguments are added as attributes (key = attribute name, 
    value = attribute value).
    
  Attributes:
    members      = list/tuple of members of the ensemble
    basetype     = class of the ensemble members
    idkey        = property of members used for unique identification
    ens_name     = name of the ensemble (string)
    ens_title    = printable title used for the ensemble (string)
    """
    # add members
    self.members = list(members)
    # add certain properties
    self.ens_name = kwargs.pop('name','')
    self.ens_title = kwargs.pop('title','')
    # no need to be too restrictive
    if 'basetype' in kwargs:
      self.basetype = kwargs.pop('basetype') # don't want to add that later! 
      if isinstance(self.basetype,basestring):
        self.basetype = globals()[self.basetype]
    elif isinstance(members[0],Dataset): self.basetype = Dataset
    elif isinstance(members[0],Variable): self.basetype = Variable
    else: self.basetype = members[0].__class__
    if len(members) > 0 and not all(isinstance(member,self.basetype) for member in members):
      raise TypeError, "Not all members conform to selected type '{}'".format(self.basetype.__name__)
    self.idkey = kwargs.get('idkey','name')
    # add keywords as attributes
    for key,value in kwargs.iteritems():
      self.__dict__[key] = value
    # add short-cuts and keys
    self.idkeys = []
    for member in self.members:
      memid = getattr(member, self.idkey)
      self.idkeys.append(memid)
      if not isinstance(memid, basestring): raise TypeError, "Member ID key '{:s}' should be a string-type, but received '{:s}'.".format(str(memid),memid.__class__)
      if memid in self.__dict__:
        raise AttributeError, "Cannot overwrite existing attribute '{:s}'.".format(memid)
      self.__dict__[memid] = member
      
  def _recastList(self, fs):
    """ internal helper method to decide if a list or Ensemble should be returned """
    if all(f is None for f in fs): return # suppress list of None's
    elif all([not callable(f) and not isinstance(f, (Variable,Dataset)) for f in fs]): return fs  
    elif all([isinstance(f, (Variable,Dataset)) for f in fs]):
      # N.B.: technically, Variable instances are callable, but that's not what we want here...
      if all([isinstance(f, Axis) for f in fs]): 
        return fs
      # N.B.: axes are often shared, so we can't have an ensemble
      elif all([isinstance(f, Variable) for f in fs]): 
        # check for unique keys
        if len(fs) == len(set([f.name for f in fs if f.name is not None])): 
          return Ensemble(*fs, idkey='name') # basetype=Variable,
        elif len(fs) == len(set([f.dataset.name for f in fs if f.dataset is not None])): 
#           for f in fs: f.dataset_name = f.dataset.name 
          return Ensemble(*fs, idkey='dataset_name') # basetype=Variable, 
        else:
          #raise KeyError, "No unique keys found for Ensemble members (Variables)"
          # just re-use current keys
          for f,member in zip(fs,self.members):
            if self.idkey == 'dataset_name':
              if f.dataset_name is None: 
                f.dataset_name = member.dataset_name
              elif not f.dataset_name == member.dataset_name: 
                raise DatasetError, f.dataset_name
            elif not hasattr(f, self.idkey): 
              setattr(f, self.idkey, getattr(member,self.idkey))
            else: raise DatasetError, self.idkey
#             f.__dict__[self.idkey] = getattr(member,self.idkey)
          return Ensemble(*fs, idkey=self.idkey) # axes from several variables can be the same objects
      elif all([isinstance(f, Dataset) for f in fs]): 
        # check for unique keys
        if len(fs) == len(set([f.name for f in fs if f.name is not None])): 
          return Ensemble(*fs, idkey='name') # basetype=Variable,
        else:
#           raise KeyError, "No unique keys found for Ensemble members (Datasets)"
          # just re-use current keys
          for f,member in zip(fs,self.members): 
            f.name = getattr(member,self.idkey)
          return Ensemble(*fs, idkey=self.idkey) # axes from several variables can be the same objects
      else:
        raise TypeError, "Resulting Ensemble members have inconsisent type."
  
  @property
  def size(self):
    assert len(self.members) == len(self.rundirs) == len(self.hgsargs) 
    return len(self.members) 
  
  def __len__(self): 
    assert len(self.members) == len(self.rundirs) == len(self.hgsargs)
    return len(self.members)
  
  def __iter__(self):
    return self.members.__iter__()
  
  def __setattr__(self, attr, value):
    """ redirect setting of attributes to ensemble members if the ensemble class does not have it """
    if attr in self.__dict__ or attr in EnsHGS.__dict__: 
      super(EnsHGS, self).__setattr__(attr, value)
    else:
      for member in self.members: setattr(member, attr, value)
  
  def __getattr__(self, attr):
    """ execute function call on ensemble members, using the same arguments; list expansion with 
        inner_list/outer_list is also supported"""
    # N.B.: this method returns an on-the-fly EnsembleWrapper instance which expands the argument       
    #       list and applies it over the list of methods from all ensemble members
    # N.B.: this method is only called as a fallback, if no class/instance attribute exists,
    #       i.e. Variable methods and attributes will always have precedent 
    # determine attribute type
    attrs = [callable(getattr(member, attr)) for member in self.members]
    if not any(attrs):
      # treat as regular attributes and return list of attibutes of all members
      return [getattr(member, attr) for member in self.members]
    elif all(attrs):
      # instantiate new wrapper with current arguments and return wrapper instance
      return EnsembleWrapper(self,attr)
    else: raise EnsembleError("Inconsistent attribute type '{}'".format(attr))
     
  def __str__(self):
    """ Built-in method; we just overwrite to call 'prettyPrint()'. """
    return self.prettyPrint(short=False) # print is a reserved word  

  def prettyPrint(self, short=False):
    """ Print a string representation of the Ensemble. """
    if short:      
      string = '{0:s} {1:s}'.format(self.__class__.__name__,self.ens_name)
      string += ', {:2d} Members ({:s})'.format(len(self.members),self.basetype.__name__)
    else:
      string = '{0:s}   {1:s}\n'.format(self.__class__.__name__,str(self.__class__))
      string += 'Name: {0:s},  '.format(self.ens_name)
      string += 'Title: {0:s}\n'.format(self.ens_title)
      string += 'Members:\n'
      for member in self.members: string += ' {0:s}\n'.format(member.prettyPrint(short=True))
      string += 'Basetype: {0:s},  '.format(self.basetype.__name__)
      string += 'ID Key: {0:s}'.format(self.idkey)
    return string

  def hasMember(self, member):
    """ check if member is part of the ensemble; also perform consistency checks """
    if isinstance(member, self.basetype):
      # basetype instance
      memid = getattr(member,self.idkey)
      if member in self.members:
        assert memid in self.__dict__
        assert member == self.__dict__[memid]
        return True
      else: 
        assert memid not in self.__dict__
        return False
    elif isinstance(member, basestring):
      # assume it is the idkey
      if member in self.__dict__:
        assert self.__dict__[member] in self.members
        assert getattr(self.__dict__[member],self.idkey) == member
        return True
      else: 
        assert member not in [getattr(m,self.idkey) for m in self.members]
        return False
    else: raise TypeError, "Argument has to be of '{:s}' of 'basestring' type; received '{:s}'.".format(self.basetype.__name__,member.__class__.__name__)       
      
  def addMember(self, member):
    """ add a new member to the ensemble """
    if not isinstance(member, self.basetype): 
      raise TypeError, "Ensemble members have to be of '{:s}' type; received '{:s}'.".format(self.basetype.__name__,member.__class__.__name__)       
    self.members.append(member)
    self.__dict__[getattr(member,self.idkey)] = member
    return self.hasMember(member)
  
  def insertMember(self, i, member):
    """ insert a new member at location 'i' """
    if not isinstance(member, self.basetype): 
      raise TypeError, "Ensemble members have to be of '{:s}' type; received '{:s}'.".format(self.basetype.__name__,member.__class__.__name__)       
    self.members.insert(i,member)
    self.__dict__[getattr(member,self.idkey)] = member
    return self.hasMember(member)
  
  def removeMember(self, member):
    """ remove a member from the ensemble """
    if not isinstance(member, (self.basetype,basestring)): 
      raise TypeError, "Argument has to be of '{:s}' of 'basestring' type; received '{:s}'.".format(self.basetype.__name__,member.__class__.__name__)
    if self.hasMember(member):
      if isinstance(member, basestring): 
        memid = member
        member = self.__dict__[memid]
      else: memid = getattr(member,self.idkey)
      assert isinstance(member,self.basetype)
      # remove from dict 
      del self.__dict__[memid]
      # remove from list
      del self.members[self.members.index(member)]
    # return check
    return not self.hasMember(member)
  
  def __mul__(self, n):
    """ how to combine with other objects """
    if isInt(n):
      return self.members*n
    else:
      raise TypeError

  def __add__(self, other):
    """ how to combine with other objects """
    if isinstance(other, Ensemble):
      for member in other: self.addMember(member)
      return self
    elif isinstance(other, list):
      return self.members + other
    elif isinstance(other, tuple):
      return tuple(self.members) * other
    else:
      raise TypeError

  def __radd__(self, other):
    """ how to combine with other objects """
    if isinstance(other, Ensemble):
      for member in other: self.addMember(member)
      return self
    elif isinstance(other, list):
      return other + self.members
    elif isinstance(other, tuple):
      return other + tuple(self.members)
    else:
      raise TypeError

  def __getitem__(self, item):
    """ Yet another way to access members by name... conforming to the container protocol. 
        If argument is not a member, it is called with __getattr__."""
    if isinstance(item, basestring): 
      if self.hasMember(item):
        # access members like dictionary
        return self.__dict__[item] # members were added as attributes
      else:
        try:
          # dispatch to member attributes 
          atts = [getattr(member,item) for member in self.members]
          if any([callable(att) and not isinstance(att, (Variable,Dataset)) for att in atts]): raise AttributeError
          return self._recastList(atts)
          # N.B.: this is useful to load different Variables from Datasets by name, 
          #       without having to use getattr()
        except AttributeError:
          if self.basetype is Dataset: raise DatasetError, item
          elif self.basetype is Variable: raise VariableError, item
          else: raise AttributeError, item
        #return self.__getattr__(item) # call like an attribute
    elif isinstance(item, (int,np.integer,slice)):
      # access members like list/tuple 
      return self.members[item]
    elif isinstance(item, (list,tuple,np.ndarray)):
      # index/label list like ndarray
      members = [self[i] for i in item] # select members
      kwargs = dict(basetype=self.basetype, idkey=self.idkey, name=self.ens_name, title=self.ens_title)
      return Ensemble(*members,**kwargs) # return new ensemble with selected members
    else: raise TypeError
  
  def __setitem__(self, name, member):
    """ Yet another way to add a member, this time by name... conforming to the container protocol. """
    idkey = getattr(member,self.idkey)
    if idkey != name: raise KeyError, "The member ID '{:s}' is not consistent with the supplied key '{:s}'".format(idkey,name)
    return self.addMember(member) # add member
    
  def __delitem__(self, member):
    """ A way to delete members by name... conforming to the container protocol. """
    if not isinstance(member, basestring): raise TypeError
    if not self.hasMember(member): raise KeyError
    return self.removeMember(member)
  
  def __iter__(self):
    """ Return an iterator over all members... conforming to the container protocol. """
    return self.members.__iter__() # just the iterator from the member list
    
  def __contains__(self, member):
    """ Check if the Ensemble instance has a particular member Dataset... conforming to the container protocol. """
    return self.hasMember(member)

  def __len__(self):
    """ return number of ensemble members """
    return len(self.members)
  
  def __iadd__(self, member):
    """ Add a Dataset to an existing Ensemble. """
    if isinstance(member, self.basetype):
      assert self.addMember(member), "A problem occurred adding Dataset '{:s}' to Ensemble.".format(member.name)    
    elif isinstance(member, Variable):
      assert all(self.addVariable(member)), "A problem occurred adding Variable '{:s}' to Ensemble Members.".format(member.name)    
    elif all([isinstance(m, Variable) for m in member]):
      assert all(self.addVariable(member)), "A problem occurred adding Variable '{:s}' to Ensemble Members.".format(member.name)    
    return self # return self as result

  def __isub__(self, member):
    """ Remove a Dataset to an existing Ensemble. """      
    if isinstance(member, basestring) and self.hasMember(member):
      assert self.removeMember(member), "A proble occurred removing Dataset '{:s}' from Ensemble.".format(member)    
    elif isinstance(member, self.basetype):
      assert self.removeMember(member), "A proble occurred removing Dataset '{:s}' from Ensemble.".format(member.name)
    elif isinstance(member, (basestring,Variable)):
      assert all(self.removeVariable(member)), "A problem occurred removing Variable '{:s}' from Ensemble Members.".format(member.name)    
    return self # return self as result

  
## run a test    
if __name__ == '__main__':

  pass
