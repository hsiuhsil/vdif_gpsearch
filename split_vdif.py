import numpy as np
import glob, time
from astropy.time import Time
import astropy.units as u, astropy.constants as c
import os, sys, time, glob, itertools

from baseband.helpers import sequentialfile as sf
from baseband import vdif
from pathlib import Path
import argparse
import mpi4py.rc
mpi4py.rc.threads = False

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


parser=argparse.ArgumentParser(description='starting the process')
parser.add_argument('-fi','--start-vdif', type=int, help='starting number of vdif files',required=True)
parser.add_argument('-ff','--end-vdif', type=int, help='ending number of vdif files',required=True)
parser.add_argument('-Nsub','--num-sub', type=int, default=16, help='the total number of subbands (max: 256)',required=True)
args = parser.parse_args()

fi = args.start_vdif
ff = args.end_vdif
Nsub=args.num_sub

raw_folder = '/scratch/p/pen/fleaf5/ARO/2012/Pdisk/20201211T202021Z_aro_vdif/*/*/'
#raw_files = sorted(glob.glob1(raw_folder,'*.vdif'))[fi:ff]
raw_files = [os.path.basename(vdif) for vdif in sorted(glob.glob(raw_folder+'*.vdif'))][fi:ff]

#save_folder = raw_folder+'Nsub_'+str(Nsub)+'/'
save_folder = '/scratch/p/pen/hsiuhsil/gp_search/20201211T202021Z_aro_vdif/'+'Nsub_'+str(Nsub)+'/'+str(fi)+'/'
Path(save_folder).mkdir(parents=True, exist_ok=True)

def main():

    t0=time.time()
    arr = np.arange(len(raw_files))#np.arange(2200,80000) #505387
    mpi_elements = np.array_split(arr, size)
    mpi_elements = comm.scatter(mpi_elements, root=0)

    mpi_process(mpi_elements)
    t1=time.time()
    print('---- ALL mpi_process DONE ----  '+str(t1-t0)+' sec')

def mpi_process(mpi_elements):

    for i in mpi_elements:
              
        raw_file=raw_files[i]
        for nsub in np.arange(Nsub):  
            split(nsub, raw_folder, raw_file)
            print('done '+str(nsub))
        

def split(nsub, raw_folder, raw_file):

    new_file = save_folder+'Nsub'+str(Nsub)+'_'+str(nsub)+'_'+raw_file
    print('new_file',new_file)

    # 1024 bytes payload + 32 bytes header = 1056 bytes = 264 words
    binary = np.fromfile(glob.glob(raw_folder+raw_file)[0], 'u4').reshape(-1, 264)

    header0 = vdif.VDIFHeader(binary[0, :8])
    header0.nchan = int(1024/Nsub)
    header0.samples_per_frame = 1

    #t1=time.time()
    #print('passing 1:', t1-t0)
    print('np.log2(Nsub).astype(int)',np.log2(Nsub).astype(int))
    lg2nchan = np.log2(header0.nchan).astype(int)

    assert header0.words[2] == (
            (32+header0.nchan) // 8  # frame (header+payload) length in 8-byte words
            + (lg2nchan << 24)    # lg2nchan
            + (1 << 29))   # VDIF version number
    binary[:, 2] = header0.words[2]
    # Only save header (8 words) plus first 128 channels = 32 words.
    h = binary[:, 0:8]
    start, end = 8+int(256/Nsub)*nsub,8+int(256/Nsub)*(nsub+1) 
    d = binary[:, start:end]
    (np.concatenate((h,d),axis=-1)).tofile(new_file)

    #t2=time.time()
    #print('passing 2:', t2-t1)



if __name__=='__main__':
    main()

