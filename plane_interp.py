from __future__ import print_function, division

import mpi4py 
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import os, re, glob, subprocess, numpy as np
from utils import read_instants,averaging2,plane_generation,generate_metadata,meteo_fields,fields_h5
from geometry_utils import geometrical_magnitudes,save_scalarfield
import pyAlya

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# Folders & files

CASESTR  = 'c'
CASESDIR = '/gpfs/scratch/bsc21/bsc21742/FF4-EUROHPC/DATASETS/CASES/'
STL_BASENAME = 'geo'
POSTDIR  = '/gpfs/scratch/bsc21/bsc21742/NEURAL_NETWORKS/dataset-generator/output/'

'''
CASESTR  = 'cas'
CASESDIR = '/gpfs/scratch/bsc21/bsc21742/NEURAL_NETWORKS/test-data/'
STL_BASENAME = 'geo'
POSTDIR  = '/gpfs/scratch/bsc21/bsc21742/NEURAL_NETWORKS/dataset-generator/output/'
'''

# Parameters
compensation = 3.75
VARLIST = ['AVVEL', 'AVPRE']
#cases = np.array([00,1,2,4,5,6,7,8,10,12,14,16,17,18,19,20,21,32,23,26,27,30,32,36,37,42,56,59,60,69,75],dtype='int')
angles = np.array([0,60,120],dtype='int')
#zetas = np.array([1.5,4.5,7.5,10.5],dtype='double')
cases = np.array([69],dtype='int')
#angles = np.array([000],dtype='int')
zetas = np.array([1.5],dtype='double')


# Create postprocess folder
os.makedirs(POSTDIR,exist_ok=True)

# Generate plane mesh
int_mesh = plane_generation(710,1421,1421)
#int_mesh = plane_generation(710,121,121)
#int_mesh = plane_generation(500,101,101)
#int_mesh = plane_generation(600,1200,1200)
#int_mesh = plane_generation(1,51,51)
pyAlya.pprint(0,'plane mesh Generated',flush=True)

# Then we store the xyz coord of the target mesh
# which is the only variable we need for the interpolation
int_xyz = int_mesh.xyz.copy() # ensure we have a deep copy


# Second loop over cases and angles to read the
# dataset, rotate and interpolate over the given
# plane
for case in cases:
	for angle in angles:
		STLDIR = '{}{:02d}/'.format(CASESDIR,case)
		CASEDIR = '{}{:02d}/{:03d}/'.format(CASESDIR,case,angle)
		OUTDIR  = '{}{:02d}/{:03d}/'.format(POSTDIR,case,angle)

		pyAlya.pprint(0,'CASE DIR: ',CASEDIR,flush=True)
		pyAlya.pprint(0,'OUT DIR:',OUTDIR,flush=True)

		if mpi_rank == 0 and not os.path.exists(OUTDIR):
			os.makedirs(OUTDIR)

		# Read source mesh
		mesh = pyAlya.Mesh.read(CASESTR,basedir=CASEDIR)
		pyAlya.pprint(0,'Mesh case {:02d} read'.format(case),flush=True)

		# Read instants
		instants = read_instants(CASEDIR)
		pyAlya.pprint(0,'instants=',instants,flush=True)

		# Average field
		avgField, avgtime = averaging2(instants,mesh,VARLIST,CASEDIR,CASESTR)
		pyAlya.pprint(1,'Field angle {:03d} averaged'.format(angle),flush=True)
		pyAlya.pprint(0,'Total avg time=',avgtime,flush=True)

		# Multiply the averages with the compensation factor
		avgField *= compensation

		# Compute the gradients
		avgField = meteo_fields(avgField,mesh)

		# Rotate the fields and the mesh
		if angle:
			avgField.rotate([0, 0, angle], center=np.array([0, 0, 0], np.double))
			mesh.rotate([0, 0, angle], center=np.array([0, 0, 0], np.double))
			pyAlya.pprint(0,'Rotated the fields by -{:03d}'.format(angle),flush=True)

		# Loop for different zetas
		for zeta in zetas:

			metadata, basename = generate_metadata(avgtime,zeta,int_xyz.shape[0],case,angle,CASEDIR)
			pyAlya.pprint(0,'Metadata generated',flush=True)

			# Save mesh
			if pyAlya.utils.is_rank_or_serial(0):
				int_mesh.save(OUTDIR + basename+'-X.h5',mpio=False)
				int_mesh.save(OUTDIR + basename+'-Y.h5',mpio=False)


			STL_FILE  = '{}{}-{:02d}.STL'.format(STLDIR,STL_BASENAME,case)
			pyAlya.pprint(0,"reading ",STL_FILE)

			#geoFields = geometrical_magnitudes(STL_FILE,int_xyz,stl_angle=[0.0,0.0,angle],stl_displ=[0.25,0.25,0.0],stl_scale=500.0,dist_resolution=0.01)
			geoFields = geometrical_magnitudes(STL_FILE,int_xyz,stl_angle=[0.0,0.0,angle],dist_resolution=0.5)

			# Modify Z coordinate
			int_xyz[:,2] = zeta

			# Field interpolation and output
			mpi_comm.Barrier()
			int_fields = fields_h5(int_xyz,mesh,avgField,geoFields['MASK'],POSTDIR,metadata,basename)

			# Save Fields
			if pyAlya.utils.is_rank_or_serial(1):
				geoFields.save(OUTDIR + basename+'-X.h5',mpio=False,metadata=metadata)
				int_fields.save(OUTDIR + basename+'-Y.h5',mpio=False,metadata=metadata)
pyAlya.pprint(0,'Done.',flush=True)
pyAlya.cr_info()
