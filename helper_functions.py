import cantera as ct
import numpy as np
from scipy.stats import binned_statistic_dd
from scipy.interpolate import griddata, RegularGridInterpolator, LinearNDInterpolator



gas = ct.Solution('./Cantera_red_mechs/mechs/Yao_nDodecane/nDodecane_sk54.xml')

# Need a species list because the 2D and 3D data is in this order while cantera species 
# are in different order so need a mapping
specs=['aC3H5','C10H20','C12H24','C12H25O2','C12OOH','C2H2','C2H3','C2H3CHO','C2H4','C2H5','C2H6','C3H6','C4H7',
'C4H81','C5H10','C5H9','C6H12','C7H14','C8H16','C9H18','CH2','CH2*','CH2CHO','CH2O','CH3','CH3O','CH4',
 'CO','CO2','H','H2','H2O','H2O2','HCO','HO2','N2','NC12H26','nC3H7','O','O2','O2C12H24OOH','OC12H23OOH','OH','pC4H9','PXC10H21','PXC12H25','PXC5H11',
 'PXC6H13','PXC7H15','PXC8H17','PXC9H19','S3XC12H25','SXC12H25']

# Species order in nonpremixed flamelets generated from FlameMaster
specs_NP=['N2','H','O2','OH','O','H2','H2O','HO2','H2O2','CO2','CO','HCO','CH2O','CH2','CH3','C2H2','CH2*','CH3O','CH4','C2H4',
          'C2H6','C2H5','C2H3','aC3H5','CH2CHO','C2H3CHO','C3H6','nC3H7','C4H7','C4H81','pC4H9','C5H9','C5H10','PXC5H11','C6H12','PXC6H13',
          'C7H14','PXC7H15','C8H16','PXC8H17','C9H18','PXC9H19','C10H20','PXC10H21','C12H24','PXC12H25','S3XC12H25','SXC12H25',
          'NC12H26','C12H25O2','C12OOH','O2C12H24OOH','OC12H23OOH']


def get_mean_molar_mass(molar_mass,data):
    mean_mass = np.zeros((data.shape[0]))
    mean_mass = 1.0/np.sum(data[:,0:molar_mass.shape[0]]/molar_mass[None,:],1)
    
    #for i in range(molar_mass.shape[0]):
    #    mean_mass[i]=mean_mass[i]+np.sum(data[0:molar_mass.shape[0],i]*molar_mass)
    return mean_mass



def get_data_2d(file):
    f = h5py.File(file,'r')
    dset = f['DATA'][:,:,3:54+3]
    T  = f['DATA'][:,:,2]
    dset=np.delete(dset,1,axis=2)
    dset = np.reshape(dset,(dset.shape[0]*dset.shape[1],53))
    T = np.reshape(T,(T.shape[0]*T.shape[1],1))
    data = dset[:,map_spec]
    data = np.append(data,T,axis=1)
    data = data[0:-1:2,:]
    return data
    

    
def get_reac_2d(file):
    f = h5py.File(file,'r')
    dset = f['DATA'][:,:,65:119]
    HRR  = -f['DATA'][:,:,60]
    dset=np.delete(dset,1,axis=2)
    dset = np.reshape(dset,(dset.shape[0]*dset.shape[1],53))
    HRR = np.reshape(HRR,(HRR.shape[0]*HRR.shape[1],1))
    data = dset[:,map_spec]
    data = np.append(data,HRR,axis=1)
    data = data[0:-1:2,:]
    return data



def get_atoms_conservation(data):
    out = np.zeros((data.shape[0],4))
    out = np.matmul((data[:,0:53]/molar_mass),aij*atom_mass)
    return out
   

def get_cp(data):
  cp_mass = np.zeros(data.shape[0])
  for i in range(0,data.shape[0]):
    comp = dict(zip(specs,data[i,0:53]))
    gas.TPY = data[i,53],6079500,comp
    cp_mass[i] = gas.cp_mass
  return cp_mass


def get_enthalpy(data):
  enth_mass3d_p = np.zeros(data.shape[0])
  for i in range(0,data.shape[0]):
    comp = dict(zip(specs,data[i,0:53]))
    gas.TPY = data[i,53],6079500,comp
    enth_mass3d_p[i] = gas.enthalpy_mass
  return enth_mass3d_p


def get_viscosity(data):
  visc = np.zeros(data.shape[0])
  for i in range(0,data.shape[0]):
    comp = dict(zip(specs,data[i,0:53]))
    gas.TPY = data[i,53],6079500,comp
    visc[i] = gas.viscosity
  return visc

def get_conductivity(data):
  cond = np.zeros(data.shape[0])
  for i in range(0,data.shape[0]):
    comp = dict(zip(specs,data[i,0:53]))
    gas.TPY = data[i,53],6079500,comp
    cond[i] = gas.thermal_conductivity
  return cond

def get_reaction(data):
  rr = np.zeros((data.shape[0],54))
  for i in range(0,data.shape[0]):
    comp = dict(zip(specs,data[i,0:53]))
    gas.TPY = data[i,53],6079500,comp
    q1 = ct.Quantity(gas)
    rr[i,:]=q1.net_production_rates
  specs_cantera=gas.species_names
  specs_cantera.remove('AR')
  rr = np.delete(rr,1,1)
  map_spec=[]
  for i in range(0,53):
     for j in range(0,53):
         if(specs[i]==specs_cantera[j]):
             map_spec.append(j)
  rr = rr[:,map_spec]
  return rr    


def get_diffusion(data):
  diff = np.zeros((data.shape[0],54))
  for i in range(0,data.shape[0]):
    comp = dict(zip(specs,data[i,0:53]))
    gas.TPY = data[i,53],6079500,comp
    diff[i,:]=gas.mix_diff_coeffs_mass
  specs_cantera=gas.species_names
  specs_cantera.remove('AR')
  diff = np.delete(diff,1,1)
  map_spec=[]
  for i in range(0,53):
     for j in range(0,53):
         if(specs[i]==specs_cantera[j]):
             map_spec.append(j)
  diff = diff[:,map_spec]

  return diff

def opt_est(Xt,data,nbins):
    
       
    cond_mean, _ , bins = binned_statistic_dd(Xt,data,bins=nbins,expand_binnumbers=True)
    cond_mean[np.isnan(cond_mean)]=0
    bins = bins-1
    bins[bins==nbins]=nbins-1
    pred = np.zeros(Xt.shape[0])
    #for i in range(0,Xt.shape[0]):
    pred=cond_mean[bins[0,:],bins[1,:],bins[2,:],bins[3,:],bins[4,:]]
    return pred

def read_data(fname,nc):    
    data = np.fromfile(fname,dtype=np.single)
    data = np.reshape(data,(int(data.size/nc),nc))
    #HRR = data[:,0]
    data = np.delete(data,0,1)
    return data

def read_data_mem(fname,nc):
    data = np.memmap(fname, dtype=np.single, mode='r')
    data = np.reshape(data,(int(data.size/nc),nc))
    #HRR = data[:,0]
    data = np.delete(data,0,1)
    return data

def read_reaction(fname):
    data = np.fromfile(fname,dtype=np.single)
    data = np.reshape(data,(int(data.size/56),56))
    HRR = data[:,0]
    data = np.delete(data,0,1)
    data[:,53]=HRR
    data = np.delete(data,54,1)
    return data

def read_reaction_mem(fname):
    data = np.memmap(fname, dtype=np.single, mode='r')
    data = np.reshape(data,(int(data.size/56),56))
    HRR = data[:,0]
    data = np.delete(data,0,1)
    data[:,53]=HRR
    data = np.delete(data,54,1)
    return data

def do_normalization(data,data2,which):
    if(which=='range'):
        datanorm = (data-np.mean(data2,0))/(np.max(data2,0)-np.min(data2,0))
        return datanorm
    elif(which=='std'):
        datanorm = (data-np.mean(data2,0))/(np.std(data2,0))
        return datanorm
    elif(which=='level'):
        datanorm = (data-np.mean(data2,0))/(np.mean(data2,0))
        return datanorm
    elif(which=='vast'):
        datanorm = (data-np.mean(data2,0))/(np.std(data2,0))*np.mean(data2,0)
        return datanorm
    elif(which=='pareto'):
        datanorm = (data-np.mean(data2,0))/np.sqrt(np.std(data2,0))
        return datanorm
    elif(which=='minmax'):
        datanorm = (data-np.min(data2,0))/(np.max(data2,0)-np.min(data2,0))
        return datanorm
    elif(which=='none'):

        return np.copy(data)

def do_inverse_norm(data,datanorm,which):
    if(which=='range'):
        data_inv = datanorm*(np.max(data,0)-np.min(data,0))+np.mean(data,0)
        return data_inv
    if(which=='std'):
        data_inv = datanorm*(np.std(data,0))+np.mean(data,0)
        return data_inv
    if(which=='level'):
        data_inv = datanorm*(np.mean(data,0))+np.mean(data,0)
        return data_inv
    if(which=='vast'):
        data_inv = datanorm*(np.std(data,0))/np.mean(data,0)+np.mean(data,0)
        return data_inv
    if(which=='pareto'):
        data_inv = datanorm*np.sqrt(np.std(data,0))+np.mean(data,0)
        return data_inv
    if(which=='minmax'):
        data_inv = datanorm*(np.max(data,0)-np.min(data,0))+np.min(data,0)
        return data_inv



def opt_est(Xt,data,nbins):

    cond_mean, _ , bins = binned_statistic_dd(Xt,data,bins=nbins,expand_binnumbers=True)
    cond_mean[np.isnan(cond_mean)]=0
    bins = bins-1
    bins[bins==nbins]=nbins-1
    pred = np.zeros(Xt.shape[0])
    pred=cond_mean[bins[0,:],bins[1,:],bins[2,:],bins[3,:],bins[4,:]]
    return pred

def get_table_noholes(Xt,data,Xt3d,nbins):

    grid = np.linspace(np.min(Xt,0),np.max(Xt,0),nbins)
    xi,yi = np.meshgrid(grid[:,0],grid[:,1])
    cond_mean = griddata((Xt[:,0],Xt[:,1]),data,(xi,yi),method='linear')
    indices = np.argwhere(np.isnan(cond_mean))
    cond_mean[indices[:,0],indices[:,1]] = griddata((Xt[:,0],Xt[:,1]),data,(xi[indices[:,0],indices[:,1]],yi[indices[:,0],indices[:,1]]),method='nearest')

    cond_mean = np.transpose(cond_mean)

    interp=RegularGridInterpolator((grid[:,0], grid[:,1]),cond_mean[0,:,:],method='linear',bounds_error=False,fill_value=0)

    return interp(Xt3d)


def cond_mean(Xt,data,nbins):

    cond_mean, _ , _ = binned_statistic_dd(Xt,data,bins=nbins,expand_binnumbers=True)
    cond_mean[np.isnan(cond_mean)]=0
    cond_mean = np.transpose(cond_mean)

    return cond_mean

def do_CMA(data):
    cmas = np.ones(data.shape[0])*5
    idxs = np.logical_or(data[:,53] < 1120, data[:,3] > 0.05*max(data[:,3]))
    cmas[idxs]=0
    idxs = np.logical_and(np.logical_and(np.logical_and(data[:,53] >=1120, data[:,3] < 0.05*max(data[:,3])),data[:,42] < 0.05*max(data[:,42]))
            ,data[:,54]>0.046)
    cmas[idxs]=1
    idxs = np.logical_and(np.logical_and(np.logical_and(data[:,53] >=1120, data[:,3] < 0.05*max(data[:,3])),data[:,42] < 0.05*max(data[:,42]))
            ,data[:,54]<0.046)
    cmas[idxs]=2
    idxs = np.logical_and(np.logical_and(data[:,53] >=1120, data[:,3] < 0.05*max(data[:,3])),data[:,42] > 0.05*max(data[:,42]))

    cmas[idxs]=3


    return cmas
