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

# work directory settings ("global" variable)
# the environment variable RAMDISK contains the path to the RAM disk
RAM = bool(os.getenv('RAMDISK', '')) # whether or not to use a RAM disk
# either RAM disk or data directory
workdir = os.getenv('RAMDISK', '') if RAM else '{:s}/test/'.format(os.getenv('DATA_ROOT', '')) 
if not os.path.exists(workdir): raise IOError, workdir

class BaseVarTest(unittest.TestCase):  
  
  # some test parameters (TestCase does not take any arguments)  
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    ''' create Axis and a Variable instance for testing '''
    self.folder = workdir
    self.dataset_name = 'TEST'
    # some setting that will be saved for comparison
    self.size = (48,2,4) # size of the data array and axes
    # the 4-year time-axis is for testing some time-series analysis functions
    te, ye, xe = self.size
    self.atts = dict(name = 'test',units = 'n/a',FillValue=-9999)
    data = np.arange(self.size[0], dtype='int16').reshape(self.size[:1]+(1,))%12 +1
    data = data.repeat(np.prod(self.size[1:]),axis=1,).reshape(self.size)
    # N.B.: the value of the field should be the count of the month (Jan=1,...,Dec=12)
    #print data
    self.data = data
    # create axis instances
    t = Axis(name='time', units='month', coord=(0,te-1,te)) # January == 0
    y = Axis(name='y', units='none', coord=(1,ye,ye))
    x = Axis(name='x', units='none', coord=(1,xe,xe))
    self.axes = (t,y,x)
    # create axis and variable instances (make *copies* of data and attributes!)
    self.var = Variable(name=self.atts['name'],units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    self.rav = Variable(name=self.atts['name'],units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    self.pax = Variable(name='pax',units=self.atts['units'],axes=self.axes[0:1],
                        data=np.arange(len(self.axes[0])),atts=self.atts.copy())
        
  def tearDown(self):
    ''' clean up '''     
    self.var.unload() # just to do something... free memory
    self.rav.unload()
    
  ## basic tests every variable class should pass

  def testEnsemble(self):
    ''' test the Ensemble container class '''
    # test object
    var = self.var
    # make a copy
    copy = var.copy()
    copy.name = 'copy of {}'.format(var.name)
    yacov = var.copy()
    yacov.name = 'yacod' # used later    
    # instantiate ensemble
    ens = Ensemble(var, copy, name='ensemble', title='Test Ensemble')
    # basic functionality
    assert len(ens.members) == len(ens)
    # these var/ax names are specific to the test dataset...
    if all(ens.hasAxis('time')):
      print ens.time 
      assert ens.time == [var.time , copy.time]
    # collective add/remove
    # test adding a new member
    ens += yacov # this is an ensemble operation
#     print(''); print(ens); print('')
    ens -= yacov # this is a ensemble operation
    assert not ens.hasMember(yacov)
    # perform a variable operation
    ens.mean(axis='time')
    print(ens.prettyPrint(short=True))
    ens -= var.name # subtract by name
#     print(''); print(ens); print('')    
    assert not ens.hasMember(var.name)
    # test call
    tes = ens(time=slice(0,3,2))
    assert all(len(tax)==2 for tax in tes.time)
      

class BaseDatasetTest(unittest.TestCase):  
  
  # some test parameters (TestCase does not take any arguments)
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    ''' create Dataset with Axes and a Variables for testing '''
    self.folder = workdir
    self.dataset_name = 'TEST'
    # some setting that will be saved for comparison
    self.size = (12,3,3) # size of the data array and axes
    te, ye, xe = self.size
    self.atts = dict(name = 'var',units = 'n/a',FillValue=-9999)
    self.data = np.random.random(self.size)   
    # create axis instances
    t = Axis(name='time', units='month', coord=(1,te,te))
    y = Axis(name='y', units='none', coord=(1,ye,ye))
    x = Axis(name='x', units='none', coord=(1,xe,xe))
    self.axes = (t,y,x)
    # create axis and variable instances (make *copies* of data and attributes!)
    var = Variable(name='var',units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    lar = Variable(name='lar',units=self.atts['units'],axes=self.axes[1:],
                        data=self.data[0,:].copy(),atts=self.atts.copy())    
    rav = Variable(name='rav',units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    pdata = np.random.random((len(self.axes[0]),)) # test float matching
    pdata = np.asarray(pdata, dtype='|S14') # test string matching
    pax = Variable(name='pax',units=self.atts['units'],axes=self.axes[0:1],
                        data=pdata,atts=None)
    self.var = var; self.lar =lar; self.rav = rav; self.pax = pax 
    # make dataset
    self.dataset = Dataset(varlist=[var, lar, rav, pax], name='test')
    # check if data is loaded (future subclasses may initialize without loading data by default)
    if not self.var.data: self.var.load(self.data.copy()) # again, use copy!
    if not self.rav.data: self.rav.load(self.data.copy()) # again, use copy!
        
  def tearDown(self):
    ''' clean up '''     
    self.var.unload() # just to do something... free memory
    self.rav.unload()
    
  ## basic tests every variable class should pass

  def testEnsemble(self):
    ''' test the Ensemble container class '''
    lsimple = self.__class__ is BaseDatasetTest
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
    # list of variable tests
    tests += ['BaseVar'] 
    # list of dataset tests
    tests += ['BaseDataset']
    
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
        test_names = ['geodata_test.'+test+'Test.test'+s_t for s_t in specific_tests]
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
    
