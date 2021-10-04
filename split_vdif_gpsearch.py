import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from pylab import *
import os, sys, shutil, time, glob, itertools, resource
from astropy.time import Time
import astropy.units as u, astropy.constants as c
from baseband.helpers import sequentialfile as sf
from baseband import vdif
from pulsar.predictor import Polyco
from scipy.ndimage.filters import median_filter, uniform_filter1d
import pyfftw.interfaces.numpy_fft as fftw
import math
import argparse
from pathlib import Path

import mpi4py.rc
mpi4py.rc.threads = False

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
mstr = f'[{rank:2d}/{size:2d}]:'

_fftargs = {'threads': int(os.environ.get('OMP_NUM_THREADS', 2)), 
            'planner_effort': 'FFTW_ESTIMATE', 
            'overwrite_input': True}

D = 4148.808 * u.s * u.MHz**2 * u.cm**3 / u.pc
width = 5000 
#dd_duration = 0.02*10 *u.s

base = '/scratch/p/pen/hsiuhsil/gp_search/ARO_R3/'
#base = '/scratch/p/pen/hsiuhsil/gp_search/ARO_crab/'
triggers_output = base+'triggers/'
cohdd_int_output = base+'cohdd_int/'
Path(triggers_output).mkdir(parents=True, exist_ok=True)
Path(cohdd_int_output).mkdir(parents=True, exist_ok=True)

# usings argv to assign the search starting and ending after fh.start_time
#ti, tf = np.float(sys.argv[1]), np.float(sys.argv[2])
#print('ti, tf (mins):',ti, tf)

# using argv to assign searching parameter
# ti, tf, (secs), Nsub, DM

parser=argparse.ArgumentParser(description='starting the process')
parser.add_argument('-fi','--start-vdif', type=int, help='starting number of vdif files',required=True)
parser.add_argument('-ff','--end-vdif', type=int, help='ending number of vdif files',required=True)
parser.add_argument('-ti','--start-sec', type=float, help='starting seconds after fh.start_time',required=True)
parser.add_argument('-tf','--end-sec', type=float, help='ending seconds after fh.start_time',required=True)
parser.add_argument('-ddt','--dd-duration', type=float, help='the total duration of coherent dedispersion (sec)',required=True)
parser.add_argument('-Nsub','--num-sub', type=int, default=16, help='the total number of subbands',required=True)
parser.add_argument('-dm','--DM', type=float, help='DM value without the unit of pc/cm**3')
args = parser.parse_args()

fi = args.start_vdif
ff = args.end_vdif
ti = args.start_sec
tf = args.end_sec
dd_duration = (args.dd_duration) *u.s
Nsub = args.num_sub
dm = args.DM

# make a new folder for dd profiles
#if rank==0:
#    shutil.rmtree('/scratch/p/pen/hsiuhsil/gp_search/saved/')
#    os.mkdir('/scratch/p/pen/hsiuhsil/gp_search/saved/')

class AROPulsarAnalysis():
    """Class to analysis pulsar data at ARO.""" 
    def __init__(self,):

        '''2020 Dec, ARO B0329 and R3'''
        self.psr_name = 'R3'
        self.DM = dm * u.pc / u.cm**3 #348.82
#        self.folder = ('/scratch/p/pen/fleaf5/ARO/2012/Ndisk/20201210T044000Z_aro_vdif/1[0-5]*/*/')
#        self.folder = ('/scratch/p/pen/hsiuhsil/gp_search/symlinks/20201210T044000Z_aro_vdif/0001201')
#        self.folder = ('/scratch/p/pen/hsiuhsil/daily_vdif_aro/B0531+21/82238071/')
        self.folder = ('/scratch/p/pen/hsiuhsil/gp_search/20201211T202021Z_aro_vdif/Nsub_'+str(Nsub)+'/'+str(fi)+'/')
        self.filenames = 'Nsub'+str(Nsub)+'_0*.vdif'
        print('self.folder + self.filenames',self.folder + self.filenames)
        self.files = sorted(glob.glob(self.folder + self.filenames))#[fi:ff]
        print('len(self.files)',len(self.files))
        self.polyco = Polyco('/home/p/pen/hsiuhsil/psr_B0329+54/polycoB0329+54_aro_mjd58725_58735.dat')

        self.fref = 800. * u.MHz
        self.full_bw = 400. * u.MHz
        self.nchan = 1024
        self.npols = 2
        self.chan_bw = self.full_bw / self.nchan
        self.dt = (1 / self.chan_bw).to(u.s)
#        print('dt', self.dt)

        fh = self.get_file_handle(self.files)
        self.start_time = fh.start_time + ti * u.s

        if tf == -1: # set the stop time to be fh.stop_time
            self.stop_time = fh.stop_time
        else:
            self.stop_time = fh.start_time + tf *u.s #fh.stop_time #+ 5 * u.min

        f0 = self.fref - self.full_bw
        wrap_time = (D * self.DM * (1/f0**2 - 1/self.fref**2)).to(u.s)
        wrap_samples = (wrap_time/self.dt).decompose().value
        self.wrap = int(np.ceil(wrap_samples))

        self.ftop = np.linspace(self.fref, self.fref - self.full_bw,
                                self.nchan, endpoint=False)


    def get_file_handle(self,files):
        """Returns file handle for a given list of channels."""

        fraw = sf.open(files, 'rb')
        fh = vdif.open(fraw, mode='rs', sample_rate=self.chan_bw)
        return fh

    def old_coherent_dedispersion(self, z, channel, axis=0):
        """Coherently dedisperse signal."""

        fcen = self.ftop[channel]
        tag = "{0:.2f}-{1:.2f}M_{2}".format(self.fref.value, fcen.value,
                                            z.shape[axis])
        ddcoh_file = "/scratch/p/pen/hsiuhsil/gp_search/saved/ddcoh_{0}.npy".format(tag)
        try:
            dd_coh = np.load(ddcoh_file)
        except:
            f = fcen + np.fft.fftfreq(z.shape[axis], self.dt)
            dang = D * self.DM * u.cycle * f * (1./self.fref - 1./f)**2
            with u.set_enabled_equivalencies(u.dimensionless_angles()):
                dd_coh = np.exp(dang * 1j).conj().astype(np.complex64).value
            np.save(ddcoh_file, dd_coh)
        if z.ndim > 1:
            ind = [np.newaxis] * z.ndim
            ind[axis] = slice(None)
        if z.ndim > 1: dd_coh = dd_coh[ind]
        z = fftw.fft(z, axis=axis, **_fftargs)
        z = fftw.ifft(z * dd_coh, axis=axis, **_fftargs)
#        z = np.fft.fft(z, axis=axis)
#        z = np.fft.ifft(z * dd_coh, axis=axis)
        return z
 
    def rebin(self, matrix, xbin, ybin):

        shape1=matrix.shape[0]//xbin
        shape2=matrix.shape[1]//ybin
        return np.nanmean(np.reshape(matrix[:shape1*xbin,:shape2*ybin],(shape1,xbin,shape2,ybin)),axis=(1,3))


    def old_process_file(self, timestamp, num_samples):
        """Seeks, reads and dedisperses signal from a given timestamp"""

        if num_samples <= self.wrap:
            raise Exception(f'num_samples must be larger than {self.wrap}!')
        else:
            t0 = time.time()
            fh = self.fh#self.get_file_handle()
            print('fh.start_time', fh.start_time)
            print('fh.stop_time', fh.stop_time)
            print('fh.shape', fh.shape)
            fh.seek(timestamp)
            print('fh.seek(timestamp)', fh.seek(timestamp))
            print ('timestamp', timestamp)
            print('num_samples',num_samples)
#            method 1
            z = fh.read(num_samples).astype(np.complex64)
#            method 2
#            z = np.memmap('z_tmp', dtype='complex64', mode='w+', shape=(num_samples, 256, 8))
#            print('create memmap')
#            fh = self.get_file_handle()
#            fh.seek(0)
#            z1 = fh.read(num_samples // 2).astype(np.complex64)
#            z[:num_samples // 2] = z1
#            print('finished z1')
#            fh = self.get_file_handle()
#            fh.seek(num_samples // 2)
#            z2 = fh.read(num_samples // 2).astype(np.complex64)
#            z[num_samples // 2:] = z2
#            print('finished z2')  

#           method 3 (subband)

#            freq = self.fref - np.arange(0, self.nchan) * self.full_bw / self.nchan
#            index = np.ceil(D * self.DM * (freq**-2 - self.fref**-2) / (self.dt)).astype(int)

#            num_subband = 4
#            freq_index = np.linspace(0, 1023, num_subband+1).astype(int)
#            for i in range(num_subband):
#                start_index, stop_index = index[freq_index[i]], index[freq_index[i+1]]

            print ('z_original.shape', z.shape)
            if z.shape[-1] != 1024:
                z = z.reshape(z.shape[0],z.shape[1],4,2).transpose(0,2,1,3).reshape(z.shape[0],z.shape[1]*int(z.shape[-1]/2),2) # z shape now is (ntime, nchan, npols)
                z = z.transpose(0,2,1) # for folding, the shape should be (ntime, npol, nchan)

            if False: #remove Bad_freq
                z[:,:,Bad_freq]=0

#                np.save('origin.npy',z) 
#                z = self.remove_rfi(z)
#            print('remove RFI done')   
#            print('z_reshape.shape',z.shape)
            t1 = time.time()
            print(f'{mstr} Took {t1 - t0:.2f}s to read.')
            t2 = time.time()
            
            for channel in range(self.nchan):
#                print('channel:',channel)
                z[..., channel] = self.coherent_dedispersion(z[..., channel],                                         channel)
            z = z[:-self.wrap]
            t3 = time.time()
            print(f'{mstr} Took {t3 - t2:.2f}s to dedisperse.')
        print ('z return shape', z.shape)

        if True: #saving DD timestream
            # z in the shape of (time, 2pol, 1024freq)
            intensity = np.nansum(abs(z.transpose(2,1,0))**2,axis=1) #(freq, time)

            # subtract noise
#            intensity-= intensity.mean(axis=1, keepdims=True)

            print('intensity.shape',intensity.shape)
            tbin, fbin = 50, 32
            rebin_I = self.rebin(intensity, fbin, tbin)
            
            np.save(cohdd_int_output+f"{self.psr_name}_{int(x.nchan/fbin)}c_{timestamp}_{num_samples}_rebinI.npy", rebin_I)

        return z

    def coherent_dedispersion(self, z, fref, DM, channel, axis=0):
        """Coherently dedisperse signal."""
        
        fcen = self.ftop[channel]
        tag = "{0:.2f}-{1:.2f}M_{2}".format(fref.value, fcen.value,
                                            z.shape[axis])
        ddcoh_file = "/scratch/p/pen/hsiuhsil/gp_search/saved/ddcoh_{0}.npy".format(tag)
        try:
            dd_coh = np.load(ddcoh_file)
        except:
            f = fcen + np.fft.fftfreq(z.shape[axis], self.dt)
            dang = D * DM * u.cycle * f * (1./fref - 1./f)**2
            with u.set_enabled_equivalencies(u.dimensionless_angles()):
                dd_coh = np.exp(dang * 1j).conj().astype(np.complex64).value
            np.save(ddcoh_file, dd_coh)
            if channel%250==0:
                print('f',f)
                print('f.shape',f.shape)
                print('dang',dang)
                print('dang.shape',dang.shape)
                print('dd_coh',dd_coh)
                print('dd_coh.shape',dd_coh.shape)

        if z.ndim > 1:
            ind = [np.newaxis] * z.ndim
            ind[axis] = slice(None)
        if z.ndim > 1: dd_coh = dd_coh[ind]
#        print('dd_coh.shape',dd_coh.shape,'z.ndim',z.ndim, 'z.shape',z.shape)
        z = fftw.fft(z, axis=axis, **_fftargs)
        z = fftw.ifft(z * dd_coh, axis=axis, **_fftargs)
#            z = np.fft.fft(z, axis=axis)
#            z = np.fft.ifft(z * dd_coh, axis=axis)
        return z


    def process_file(self, timestamp): 

        '''getting the baseband data to be coherently de-dispersed'''
        #fh: baseband data
        #T0: the beginning time of the baseband data at the reference frequency
        #DM: the de-dispersion value
        #dd_duration: how much time of baseband data to be dedispersed at the given DM.
    
        '''try subband'''
#        Nsub = Nsub
        s = []
        for nsub in range(Nsub):
            print('------------ nsub: ',str(nsub),'----------')

            folder = self.folder
            files = folder+'/Nsub'+str(Nsub)+'_'+str(nsub)+'*.vdif'
            files = sorted(glob.glob(files))#[fi:ff]
            fh = self.get_file_handle(files)

            sub_ftop = self.ftop.reshape(Nsub,int(self.nchan/Nsub))[nsub]
            sub_nchan = len(sub_ftop)
            sub_fref, sub_f0 = sub_ftop[0], sub_ftop[-1] 
            print('sub_fref, sub_f0', sub_fref, sub_f0)    
            wrap_time = (D * self.DM * (1/sub_fref**2 - 1/self.fref**2)).to(u.s)
            print('wrap_time',wrap_time)
            sub_wrap_time = (D * self.DM * (1/sub_f0**2 - 1/sub_fref**2)).to(u.s)
            sub_wrap_samples = (sub_wrap_time/self.dt).decompose().value
            wrap = int(np.ceil(sub_wrap_samples))
            print('wrap:',wrap)
        
            num_samples = int((sub_wrap_time+dd_duration).value/(self.dt).value)
            print('num_samples',num_samples)
            dd_time_samples = int((dd_duration).value/(self.dt).value)
            print('dd_time_samples',dd_time_samples)
            T0 = Time(timestamp, precision=9)+wrap_time
            print('timestamp',timestamp, 'T0',T0)
#            fh = self.get_file_handle()
            fh.seek(T0)
            print(fh.tell(unit='time'), T0,fh.tell(unit='time') == T0)
       
            t0=time.time()
            z = fh.read(num_samples).astype(np.complex64)
            print('z.shape',z.shape)
            print('fh.read: ', time.time()-t0)
    
            for channel in range(sub_nchan):
                if channel%100==0:
                    print('channel:',channel)
                z[..., channel] = self.coherent_dedispersion(z[..., channel], sub_fref, self.DM, channel+nsub*sub_nchan)

            s.append(z[:dd_time_samples,:,:])
            print('coh dd seconds: ', time.time()-t0) 
            print('wrap',wrap)
        s = np.asarray(s)   
        s = np.concatenate(s,axis=-1)
        print('final s shape', s.shape)

        if True: #saving DD timestream
            # z in the shape of (time, 2pol, 1024freq)
            intensity = np.nansum(abs(s.transpose(2,1,0))**2,axis=1) #(freq, time)

            # subtract noise
#            intensity-= intensity.mean(axis=1, keepdims=True)

            print('intensity.shape',intensity.shape)
            tbin, fbin = 50, 32
            rebin_I = self.rebin(intensity, fbin, tbin)

            np.save(cohdd_int_output+f"{self.psr_name}_{int(x.nchan/fbin)}c_{timestamp}_{num_samples}_rebinI.npy", rebin_I)

        return s # in the shape of (time, 2 pol, 1024 nchan)

    def get_phases(self, timestamp, num_samples, dt, ngate):
        """Returns pulse phase."""

        phasepol = self.polyco.phasepol(timestamp, rphase='fraction', 
                                        t0=timestamp, time_unit=u.second,
                                        convert=True)
        ph = phasepol(np.arange(num_samples) * dt.to(u.s).value)
        print('ph1',ph)
        ph -= np.floor(ph[0])
        print('ph2',ph)
        ph = np.remainder(ph * ngate, ngate).astype(np.int32)
        print('ph3',ph)
        return ph

    def gp_finder_method(self, z, gp_thres, gp_size):
        """Method to find giant pulses in signal."""

#        y = uniform_filter1d(z, gp_size, origin=-gp_size//2)
        y=z
        y = (y - y.mean()) / y.std()
        y /= y[abs(y) < 6].std()
        y /= y[abs(y) < 6].std()
        gp_index = np.argwhere(y > gp_thres).squeeze(-1)
        if gp_index.shape[0] > 0:
            gp_index = gp_index[np.logical_and(gp_index > gp_size,
                                               gp_index <
                                               (y.shape[0] - gp_size))]
            l0, l1 = 1, 0
            while l0 != l1:
                l0 = len(gp_index)
                for i, p in enumerate(gp_index):
                    gp_index[i] = (np.argmax(y[p-gp_size:p+gp_size]) + p
                                   - gp_size)
                gp_index = np.unique(gp_index)
                gp_index = gp_index[np.logical_and(gp_index > gp_size,
                                                   gp_index <
                                                   (y.shape[0] - gp_size))]
                l1 = len(gp_index)
        gp_sn = y[gp_index]
        return gp_index, gp_sn

    def get_times_list(self, num_samples):
        """Make time list."""

        first_file_num = True
        for file_num in self.config.file_nums:
            st = int(np.ceil(self.config.time_bounds[file_num][0].unix + 1))
            et = int((self.config.time_bounds[file_num][1] - (self.config.wrap
                 * self.config.dt)).unix - 1)
            file_length = (et - st) * u.s
            assert file_length > 0
            chuck_length = (num_samples - self.config.wrap) * self.config.dt
            num_chunks = int((file_length / chuck_length).decompose())
            times = np.linspace(st, et, num_chunks, endpoint=False)
            if first_file_num:
                tlist = np.array([Time(t, format='unix', precision=9) for t in times])
                for t in tlist:
                    t.format = 'isot'
                first_file_num = False
            else:
                tlist1 = np.array([Time(t, format='unix', precision=9) for t in times])
                for t in tlist1:
                    t.format = 'isot'
                tlist = np.concatenate((tlist, tlist1))
        return tlist

def gp_finder(pa, t, num_samples, gp_thres=5, gp_size=128):
    """Find giant pulses in signal."""

    print(f'{mstr} --Finding giant pulses--')
    gp_data = []
    gp_raw = []
    phasepol = pa.polyco.phasepol(t, rphase='fraction', t0=t,
                               time_unit=u.second, convert=True)

    print('str(t)',str(t))
    z_raw = pa.process_file(t) # the shape should be (time,2pol, 1024freq)
    z = abs(z_raw)**2

    print('gp_thres',gp_thres)
    print('gp_size',gp_size)

    '''subband search'''
    if False: # subband search
        zs = z.sum(axis=1)
        Nsub=8
        zs = zs.reshape(zs.shape[0],Nsub,int(1024/Nsub))
        zs = zs.sum(-1)

        sub_index, sub_sn = [],[]

        for nsub in range(zs.shape[-1]):
            sub2_index, sub2_sn = pa.gp_finder_method(zs[:,nsub], gp_thres=2, gp_size=512)
#    print('sub_sn.max()',sub_sn.max())
            sub2_index, sub2_sn = sub2_index[sub2_sn==sub2_sn.max()], sub2_sn[sub2_sn==sub2_sn.max()]
            print('sub2_index, sub2_sn',sub2_index, sub2_sn)

            sub_index.append(sub2_index[0])
            sub_sn.append(sub2_sn[0])
 
        print('zs.shape',zs.shape)

        print('sub_index, sub_sn',sub_index, sub_sn)
        sub_index, sub_sn = np.asarray(sub_index), np.asarray(sub_sn)
        gp_index, gp_sn = sub_index[sub_sn==sub_sn.max()], sub_sn[sub_sn==sub_sn.max()]

        print('SUBBAND search gp_index, gp_sn', gp_index, gp_sn)

    else: #wholeband
#        z = z.sum(-1).sum(-1)
        z = np.nansum(z.transpose(2,1,0),axis=1) #(freq, time)

        # subtract noise
        z -= z.mean(axis=1, keepdims=True)
        z = z.mean(axis=0)

        print('z.shape',z.shape)

        gp_index, gp_sn = pa.gp_finder_method(z, gp_thres, gp_size)
    print((f'{mstr} -- Time: {t.isot} '
               f'Found {len(gp_index)} giant pulses.'))
#    width = 5000
    if len(gp_index) > 0:
        t_arr = []; indx_arr = []; event_arr = []; sn_arr = []; gp_arr =[]; data_arr =[];
        for index, sn in zip(gp_index, gp_sn):
            print('sn: ',sn)
            gp = phasepol((index * NFFT*pa.dt).to(u.s).value) #% 1
#            print('gp:',gp)
            event_t = t + index * pa.dt
            gp_data.append((event_t, sn, gp))
            if index < width:
                gp_raw.append((z_raw[0:width]))
            elif index > (num_samples - width):
                gp_raw.append((z_raw[num_samples-width:(num_samples-1)]))
            else:
                gp_raw.append((z_raw[int(index - width/2):int(index + width/2)]))
            t_arr.append(t.value)
            indx_arr.append(index)
            event_arr.append(event_t)
            sn_arr.append(sn)
            gp_arr.append(gp)
        t_arr = np.asarray(t_arr)
        indx_arr = np.asarray(indx_arr)
        event_arr = np.asarray(event_arr)
        sn_arr = np.asarray(sn_arr)
        gp_arr = np.asarray(gp_arr)
        data_arr = np.asarray(gp_raw)

        if len(sn_arr)//10==0:

            file_name = triggers_output + 'gp_raw_t'+str(t)+'.npz'
            np.savez(file_name,
                     block_t=t_arr,
                     index=indx_arr,
                     event_t=event_arr,
                     snr=sn_arr,
                     phase=gp_arr,
                     data=data_arr)
        elif len(sn_arr)//10>0:

            for x in range(len(sn_arr)//10):

                file_name = triggers_output + 'gp_raw_t'+str(t)+'_'+str(x)+'.npz'
                np.savez(file_name,
                     block_t=t_arr[x*10:(x+1)*10],
                     index=indx_arr[x*10:(x+1)*10],
                     event_t=event_arr[x*10:(x+1)*10],
                     snr=sn_arr[x*10:(x+1)*10],
                     phase=gp_arr[x*10:(x+1)*10],
                     data=data_arr[x*10:(x+1)*10])
            remain_num = len(sn_arr)%10
            if remain_num !=0:

                file_name = triggers_output + 'gp_raw_t'+str(t)+'_'+str(len(sn_arr)//10)+'.npz'
                np.savez(file_name,
                     block_t=t_arr[-remain_num:],
                     index=indx_arr[-remain_num:],
                     event_t=event_arr[-remain_num:],
                     snr=sn_arr[-remain_num:],
                     cycle=gp_arr[-remain_num:],
                     data=data_arr[-remain_num:])

        print('\x1b[6;30;43m' + '*** saved gp_raw ***' + '\x1b[0m')
#    print(gp_data)
    return gp_data

x = AROPulsarAnalysis()
N = 2**22 #2**int(np.round(np.log(x.wrap)/np.log(2)))#2**19

# print(f'Making waterfall for {x.psr_name}.')
# z = make_waterfall(x, x.start_time, N)
# np.save(f"{x.psr_name}_waterfall_plus10min.npy", z)

ngate = 512
NFFT = 1 

x.wrap += (-x.wrap) % NFFT
block_length = dd_duration #((N - x.wrap) * x.dt).to(u.s)
max_time = ((x.stop_time - x.start_time) - x.wrap * x.dt).to(u.s)
print('max_time',max_time)
max_blocks = int(floor((max_time / block_length).decompose().value))
print('max_blocks',max_blocks)
num_blocks = max_blocks #260 
assert num_blocks <= max_blocks
timestamps = [x.start_time + i * block_length for i in range(num_blocks)]


if rank == 0:
    print(f"------------------------\n"
          f"Folding {x.psr_name} data.\n"
          f"Observation Details --\n"
          f"{x.start_time} -> {x.stop_time}\n"
          f"Total Duration (s): {max_time}\n"
          f"Block Length (s): {block_length.to(u.s)}\n"
          f"No. of blocks: {num_blocks} (Max: {max_blocks})\n"
#          f"Time to fold: {(num_blocks * block_length).to(u.s)}\n"
          f"------------------------", flush=True)

comm.Barrier()

time.sleep(rank)
t0=time.time()
for timestamp in timestamps[rank::size]:
    print(f'{mstr} {timestamp}')
#    pp, count = fold_band(x, timestamp, N, ngate, NFFT)
#    ppfull += pp
#    counts += count
    try:
        '''B0531+21: gp_size=512, B0329+54: gp_size= 4096'''
        gp_data = gp_finder(x, timestamp, N, gp_thres=5, gp_size=1024) 
    except e as Exception:
        print('error in this timestamp')
        print(str(e))
        pass

#    np.save('gp.npy', gp_data)

max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6 #KB to GB
#print('max_mem (GB): ',max_mem)
print('Done the whole search from ',str(ti),' to ',str(tf),' secs,\n  Nsub of ',str(Nsub),'\n  DM of ',str(dm),'\n  dd_duration (sec): ',str(dd_duration.value),'\n  Spending ',str(time.time()-t0),' secs,\n  Max mem (GB) of ',str(max_mem))


