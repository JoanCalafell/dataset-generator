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
CASESDIR = '/gpfs/scratch/bsc21/bsc21742/Paper-Oklahoma/estudi-longitud/'
STL_BASENAME = 'geo'
POSTDIR  = '/gpfs/scratch/bsc21/bsc21742/NEURAL_NETWORKS/dataset/output/'

# Parameters
fact = 3
compensation = 3.75
VARLIST = ['AVVEL', 'AVPRE']
#cases = np.array([00,1,2,4,5,6,7,8,10,12,14,16,17,18,19,20,21,32,23,26,27,30,32,36,37,42,56,59,60,69,75],dtype='int')
#angles = np.array([0,60,120],dtype='int')
#zetas = np.array([1.5,4.5,7.5,10.5],dtype='double')
cases = np.array([00],dtype='int')
angles = np.array([120],dtype='int')
zetas = np.array([1.5],dtype='double')


# Create postprocess folder
os.makedirs(POSTDIR,exist_ok=True)

# Generate plane mesh
#int_mesh = plane_generation(710,1421,1421)
#int_mesh = plane_generation(710,121,121)
int_mesh = plane_generation(500,101,101)
#int_mesh = plane_generation(600,1200,1200)
pyAlya.pprint(0,'plane mesh Generated',flush=True)

'''
# First loop over cases and angles to store
# the mesh into an h5 file
for case in cases:
	for angle in angles:
		CASEDIR = '{}{:02d}/{:03d}/'.format(CASESDIR,case,angle)
		OUTDIR  = '{}{:02d}/{:03d}/'.format(POSTDIR,case,angle)
		if mpi_rank == 0 and not os.path.exists(OUTDIR):
			os.makedirs(OUTDIR)
		for zeta in zetas:
			_, basename = generate_metadata(1, zeta,1,case,angle,CASEDIR)

			# Save mesh
			if pyAlya.utils.is_rank_or_serial(0):
				int_mesh.save(OUTDIR + basename+'.h5',mpio=False)
'''
# Then we store the xyz coord of the target mesh
# which is the only variable we need for the interpolation
int_xyz = int_mesh.xyz.copy() # ensure we have a deep copy

# Delete the int_mesh structre that might be using
# quite some memory
del int_mesh

STL_FILE=CASESDIR+'00/geo-00.STL'
#STL_FILE='/gpfs/scratch/bsc21/bsc21742/NEURAL_NETWORKS/dataset/caso25.stl'
angle=0.0

geoFields = geometrical_magnitudes(STL_FILE,int_xyz,angle,stl_scale=1000.0)

mask_G=np.reshape(geoFields['MASK'],(101,101))
#height_G=np.reshape(height_G,(101,101))
#distance_G=np.reshape(distance_G,(101,101))
if mpi_rank==0: save_scalarfield(mask_G,"mask.png")
#if mpi_rank==0: save_scalarfield(height_G,"height.png")
#if mpi_rank==0: save_scalarfield(distance_G,"distance.png")



exit(0)

# Second loop over cases and angles to read the
# dataset, rotate and interpolate over the given
# plane
for case in cases:
	for angle in angles:
		CASEDIR = '{}{:02d}/{:03d}/'.format(CASESDIR,case,angle)
		OUTDIR  = '{}{:02d}/{:03d}/'.format(POSTDIR,case,angle)

		pyAlya.pprint(0,'CASE DIR: ',CASEDIR,flush=True)
		pyAlya.pprint(0,'OUT DIR',OUTDIR,flush=True)
		# creating symbolic links; only when the directories are private 
		PARDEST = CASEDIR + CASESTR + ".post.alyapar"	
		PARSOUR = OUTDIR  + CASESTR + ".post.alyapar"	
		pyAlya.pprint(0,'PARDEST=',PARDEST,flush=True)
		pyAlya.pprint(0,'PARSOUR=',PARSOUR,flush=True)
		if mpi_rank == 0 and not os.path.exists(OUTDIR):
			os.makedirs(OUTDIR)
			os.system("ln -fs %s %s"%(PARDEST, PARSOUR))


		STL_FILE  = '{}{}-{:02d}.STL'.format(CASEDIR,STL_BASENAME,case,angle)
		pyAlya.pprint(0,"reading ",STL_FILE)
		geoFields = geometrical_magnitudes(STL_FILE,angle)



		exit(0)

		# Read source mesh
		mesh = pyAlya.Mesh.read(CASESTR,basedir=CASEDIR)
		pyAlya.pprint(0,'Mesh case {:02d} read'.format(case),flush=True)

		# Read instants
		instants = read_instants(CASEDIR)
		pyAlya.pprint(0,'instants=',instants,flush=True)

		# Average field
		avgField, avgtime = averaging2(instants,mesh,VARLIST,CASEDIR,CASESTR)
		pyAlya.pprint(1,'Field angle {:03d} averaged'.format(angle),flush=True)
		pyAlya.pprint(0,'avg time=',avgtime,flush=True)

		# Multiply the averages with the compensation factor
		avgField *= compensation

		# Compute the gradients
		avgField = meteo_fields(avgField)

		# Rotate the fields and the mesh
		if angle:
			avgField.rotate([0, 0, angle], center=np.array([0, 0, 0], np.double))
			mesh.rotate([0, 0, angle], center=np.array([0, 0, 0], np.double))
			pyAlya.pprint(0,'Rotated the fields by -{:03d}'.format(angle),flush=True)

		# Loop for different zetas
		for zeta in zetas:

			# Generate metadata
			metadata, basename = generate_metadata(avgtime,zeta,int_xyz.shape[0],case,angle,CASEDIR)
			pyAlya.pprint(0,'Metadata generated',flush=True)

			# Modify Z coordinate
			int_xyz[:,2] = zeta

			# Field interpolation and output
			mpi_comm.Barrier()
			int_field = fields_h5(int_xyz,mesh,avgField,POSTDIR,metadata,basename)

			# Save Fields
			if pyAlya.utils.is_rank_or_serial(1):
				int_field.save(OUTDIR + basename+'.h5',mpio=False,metadata=metadata)
pyAlya.pprint(0,'Done.',flush=True)
pyAlya.cr_info()
