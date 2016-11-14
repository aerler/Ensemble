'''
Created on 2013-08-24 

Unittest for the GeoPy main package geodata.

@author: Andre R. Erler, GPL v3
'''

import unittest
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import scipy.stats as ss
import os
import gc
from copy import deepcopy
import shutil

# internal imports
# from ensemble.base import Ensemble
from ensemble.expand import expandArgumentList


## tests related to loading datasets
class ArgumentTest(unittest.TestCase):  
   
  def setUp(self):
    ''' create two test variables '''
    pass
      
  def tearDown(self):
    ''' clean up '''
    gc.collect()

  def testExpArgList(self):
    ''' test function to expand argument lists '''    
    # test arguments
    args1 = [0,1,2]; args2 = ['0','1','2']; args3 = ['test']*3; arg4 = 'static1'; arg5 = 'static2' 
    explist = ['arg1','arg2','arg3']
    # test inner product expansion
    arg_list = expandArgumentList(arg1=args1, arg2=args2, arg3=args3, arg4=arg4, arg5=arg5,
                                  expand_list=explist, lproduct='inner')
    assert len(arg_list) == len(args1) and len(arg_list) == len(args2)
    for args,arg1,arg2,arg3 in zip(arg_list,args1,args2,args3):
      assert args['arg1'] == arg1
      assert args['arg2'] == arg2
      assert args['arg3'] == arg3
      assert args['arg4'] == arg4
      assert args['arg5'] == arg5
    # test outer product expansion
    arg_list = expandArgumentList(arg1=args1, arg2=args2, arg3=args3, arg4=arg4, arg5=arg5,
                                  expand_list=explist, lproduct='outer')
    assert len(arg_list) == len(args1) * len(args2) * len(args3)
    n = 0
    for arg1 in args1:
      for arg2 in args2:
        for arg3 in args3:
          args = arg_list[n]
          assert args['arg1'] == arg1
          assert args['arg2'] == arg2
          assert args['arg3'] == arg3
          assert args['arg4'] == arg4
          assert args['arg5'] == arg5
          n += 1
    assert n == len(arg_list)
    # test simultaneous inner and outer product expansion
    n1 = len(args2) * len(args3) / len(args1)
    tmp1 = args1*n1
    arg_list = expandArgumentList(arg1=tmp1, arg2=args2, arg3=args3, arg4=arg4, arg5=arg5,
                                  outer_list=['arg2','arg3'], inner_list=['arg1'])
    assert len(arg_list) == len(args2) * len(args3) == len(tmp1)
    n = 0
    for arg2 in args2:
      for arg3 in args3:
          args = arg_list[n]
          assert args['arg1'] == tmp1[n]
          assert args['arg2'] == arg2
          assert args['arg3'] == arg3
          assert args['arg4'] == arg4
          assert args['arg5'] == arg5
          n += 1
    assert n == len(arg_list)
    # test parallel outer product expansion
    assert len(args1) == len(args2) # necessary for test
    arg_list = expandArgumentList(arg1=args1, arg2=args2, arg3=args3, arg4=arg4, arg5=arg5,
                                  expand_list=[('arg1','arg2'),'arg3'], lproduct='outer')
    assert len(arg_list) == len(args1) * len(args3)
    n = 0
    for arg1,arg2 in zip(args1,args2):
      for arg3 in args3:
        args = arg_list[n]
        assert args['arg1'] == arg1
        assert args['arg2'] == arg2
        assert args['arg3'] == arg3
        assert args['arg4'] == arg4
        assert args['arg5'] == arg5
        n += 1
    assert n == len(arg_list)
    

## simple tests for the Container protocol
class ContainerTest(unittest.TestCase):  
  
  def setUp(self):
    ''' create some objects for testing '''
    pass
  
  def tearDown(self):
    ''' clean up '''     
    gc.collect() 
    
  def testEnsemble(self):
    ''' simple test for the Ensemble container class '''
    # make test objects
    test_1 = 'test 1'; test_2 = 'test 2'; test_3 = 'test 3'
    # instantiate ensemble
    ens = Ensemble(test_1, test_2, name='ensemble', title='Test Ensemble')
    # basic functionality
    assert len(ens.members) == len(ens) == 2
    assert test_1 in ens and test_2 in ens
    # collective add/remove
    # test adding a new member
    ens += test_3 # this is an ensemble operation
    assert len(ens) == 3
    assert test_3 in ens
    # remove
    del ens[-1]
    assert len(ens) == 2
    assert test_3 not in ens
    # print representation
    print(''); print(ens); print('')


## tests for the method redirect functionality
class MethodTest(unittest.TestCase):  
    
  def setUp(self):
    ''' create Dataset with Axes and a Variables for testing '''
    pass
  
  def tearDown(self):
    ''' clean up '''     
    gc.collect()

  def testEnsemble(self):
    ''' test the Ensemble container class '''
    # test object
    dataset = self.dataset
    dataset.load()
    # make a copy
    copy = dataset.copy()
    copy.name = 'copy of {}'.format(dataset.name)
    yacod = dataset.copy()
    yacod.name = 'yacod' # used later    
    # instantiate ensemble
    ens = Ensemble(dataset, copy, name='ensemble', title='Test Ensemble', basetype='Dataset')
    # basic functionality
    assert len(ens.members) == len(ens)
    # these var/ax names are specific to the test dataset...
    if all(ens.hasVariable('var')):      
      assert isinstance(ens.var,Ensemble) 
      assert ens.var.basetype == Variable
      #assert ens.var == Ensemble(dataset.var, copy.var, basetype=Variable, idkey='dataset_name')
      assert ens.var.members == [dataset.var, copy.var]
      #print ens.var
      #print Ensemble(dataset.var, copy.var, basetype=Variable, idkey='dataset_name')
    #print(''); print(ens); print('')        
    #print ens.time
    assert ens.time == [dataset.time , copy.time]
    # Axis ensembles are not supported anymore, since they are often shared.
    #assert isinstance(ens.time,Ensemble) and ens.time.basetype == Variable
    # collective add/remove
    ax = Axis(name='ax', units='none', coord=(1,10))
    var1 = Variable(name='new',units='none',axes=(ax,))
    var2 = Variable(name='new',units='none',axes=(ax,))
    ens.addVariable([var1,var2], copy=False) # this is a dataset operation
    assert ens[0].hasVariable(var1)
    assert ens[1].hasVariable(var2)
    assert all(ens.hasVariable('new'))
    # test adding a new member
    ens += yacod # this is an ensemble operation
    #print(''); print(ens); print('')    
    ens -= 'new' # this is a dataset operation
    assert not any(ens.hasVariable('new'))
    ens -= 'test'
    # fancy test of Variable and Dataset integration
    assert not any(ens[self.var.name].mean(axis='time').hasAxis('time'))
    print(ens.prettyPrint(short=True))
    # apply function to dataset ensemble
    if all(ax.units == 'month' for ax in ens.time):
      maxens = ens.seasonalMax(lstrict=not lsimple); del maxens
    # test call
    tes = ens(time=slice(0,3,2))
    assert all(len(tax)==2 for tax in tes.time)
    # test list indexing
    sne = ens[range(len(ens)-1,-1,-1)]
    assert sne[-1] == ens[0] and sne[0] == ens[-1]


if __name__ == "__main__":


    # use Intel MKL multithreading: OMP_NUM_THREADS=4
#     import os
    print('OMP_NUM_THREADS = {:s}\n'.format(os.environ['OMP_NUM_THREADS']))    
        
    specific_tests = []
#     specific_tests += ['Ensemble']

    # list of tests to be performed
    tests = [] 
    # list of Container tests
    tests += ['Argument'] 
    # list of Container tests
#     tests += ['Container'] 
    # list of Method tests
#     tests += ['Method']
    
    # construct dictionary of test classes defined above
    test_classes = dict()
    local_values = locals().copy()
    for key,val in local_values.iteritems():
      if key[-4:] == 'Test':
        test_classes[key[:-4]] = val
    
    
    # run tests
    report = []
    for test in tests: # test+'.test'+specific_test
      if specific_tests: 
        test_names = ['ensemble_test.'+test+'Test.test'+s_t for s_t in specific_tests]
        s = unittest.TestLoader().loadTestsFromNames(test_names)
      else: s = unittest.TestLoader().loadTestsFromTestCase(test_classes[test])
      report.append(unittest.TextTestRunner(verbosity=2).run(s))
      
    # print summary
    runs = 0; errs = 0; fails = 0
    for name,test in zip(tests,report):
      #print test, dir(test)
      runs += test.testsRun
      e = len(test.errors)
      errs += e
      f = len(test.failures)
      fails += f
      if e+ f != 0: print("\nErrors in '{:s}' Tests: {:s}".format(name,str(test)))
    if errs + fails == 0:
      print("\n   ***   All {:d} Test(s) successfull!!!   ***   \n".format(runs))
    else:
      print("\n   ###     Test Summary:      ###   \n" + 
            "   ###     Ran {:2d} Test(s)     ###   \n".format(runs) + 
            "   ###      {:2d} Failure(s)     ###   \n".format(fails)+ 
            "   ###      {:2d} Error(s)       ###   \n".format(errs))
    
