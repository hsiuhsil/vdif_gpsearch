#!/bin/bash 
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=11
#SBATCH --time=20:59:50
#SBATCH --job-name searching
#SBATCH --output=mpi_ex_%j.txt
#SBATCH --mail-type=ALL

#cd /home/p/pen/h`siuhsil/codes/
# EXECUTION COMMAND; -np = nodes*ppn

#mpirun -np 16 python findgp.py

cd /scratch/p/pen/hsiuhsil/gp_search

mpirun -np 198 python test_vdif.py -fi 0 -ff 10500 -ti 0 -tf 875 -ddt 1.0 -Nsub 8 -dm 348.82
mpirun -np 198 python test_vdif.py -fi 10000 -ff 20500 -ti 0 -tf 875 -ddt 1.0 -Nsub 8 -dm 348.82

mpirun -np 198 python test_vdif.py -fi 20000 -ff 30500 -ti 0 -tf 875 -ddt 1.0 -Nsub 8 -dm 348.82
mpirun -np 198 python test_vdif.py -fi 30000 -ff 40500 -ti 0 -tf 875 -ddt 1.0 -Nsub 8 -dm 348.82

mpirun -np 198 python test_vdif.py -fi 40000 -ff 50500 -ti 0 -tf 875 -ddt 1.0 -Nsub 8 -dm 348.82
mpirun -np 198 python test_vdif.py -fi 50000 -ff 60500 -ti 0 -tf 875 -ddt 1.0 -Nsub 8 -dm 348.82
mpirun -np 198 python test_vdif.py -fi 60000 -ff 70500 -ti 0 -tf 875 -ddt 1.0 -Nsub 8 -dm 348.82
mpirun -np 198 python test_vdif.py -fi 70000 -ff 80500 -ti 0 -tf 875 -ddt 1.0 -Nsub 8 -dm 348.82


#mpirun -np 1 python test_vdif.py -fi 310000 -ff 320500 -ti 0 -tf 12.966 -ddt 6 -Nsub 4 -dm 348.82


#  235 secs for Nsub of 4 by 40 cores
#  351 secs for Nsub of 8 by 40 cores
#  579 secs for Nsub of 16 by 40 cores
#  1147 secs for Nsub of 32 by 40 cores
#  2017 secs for Nsub of 64 by 40 cores
#mpirun -np 40 python split_vdif.py -fi 310000 -ff 320500 -Nsub 4

#mpirun -np 4 python test_vdif.py -fi 310000 -ff 320500 -ti 0 -tf 875 -ddt 6.0 -Nsub 1 -dm 348.82


#mpirun -np 1 python test_vdif.py -ti 0 -tf 3.121 -ddt 0.2 -Nsub 1 -dm 56.7546

#mpirun -np 1 python test_vdif.py -ti 0 -tf 12.966 -ddt 6.0 -Nsub 1 -dm 348.82
#mpirun -np 1 python test_vdif.py -ti 0 -tf 12.966 -ddt 6.0 -Nsub 2 -dm 348.82
#mpirun -np 1 python test_vdif.py -ti 0 -tf 14.966 -ddt 8.0 -Nsub 1 -dm 348.82
#mpirun -np 1 python test_vdif.py -ti 0 -tf 14.966 -ddt 8.0 -Nsub 2 -dm 348.82

#mpirun -np 1 python test_vdif.py -ti 0 -tf 1.32 -Nsub 1 -dm 56.7546
#mpirun -np 1 python test_vdif.py -ti 0 -tf 1.32 -Nsub 2 -dm 56.7546
#mpirun -np 1 python test_vdif.py -ti 0 -tf 1.32 -Nsub 4 -dm 56.7546
#mpirun -np 1 python test_vdif.py -ti 0 -tf 1.32 -Nsub 8 -dm 56.7546
#mpirun -np 1 python test_vdif.py -ti 0 -tf 1.32 -Nsub 16 -dm 56.7546
#mpirun -np 1 python test_vdif.py -ti 0 -tf 1.32 -Nsub 32 -dm 56.754
