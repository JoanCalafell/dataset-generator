from __future__ import print_function, division

import mpi4py 
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import os, re, glob, subprocess, numpy as np
import pyAlya

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def read_instants(CASEDIR):
	steps = []
	for file in glob.glob(CASEDIR+"*AVPRE*post*"):
		steps.append(int(re.split('[- .]',file)[-4]))
	steps.sort()
	steps=steps[0:3]
	return steps

def averaging(instants,mesh,VARLIST,CASEDIR,CASESTR):

	# Read the first instant of the list
	_, header = pyAlya.Field.read(CASESTR,VARLIST,instants[0],mesh.xyz,basedir=CASEDIR)

	# Initialize field at zero
	avgField = pyAlya.Field(xyz = mesh.xyz, AVVEL = mesh.newArray(ndim=3))
	time     = header.time
	for instant in instants[1:]:
		field, header = pyAlya.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=CASEDIR)

#		if mpi_rank == 0: continue # Skip master

		# Compute time-weighted average (Welford's online algorithm)
		dt   = header.time - time # weight
		time = header.time        # sum_weight
		avgField["AVVEL"] += pyAlya.stats.addS1(avgField["AVVEL"], field["AVVEL"], w=dt/time)

	return avgField, time 

def averaging2(instants,mesh,VARLIST,CASEDIR,CASESTR):

	# Read the first instant of the list
	pyAlya.pprint(0,"CASESTR=",CASESTR,flush=True)
	pyAlya.pprint(0,"VARLIST=",VARLIST,flush=True)
	pyAlya.pprint(0,"CASEDIR=",CASEDIR,flush=True)
	pyAlya.pprint(0,"inst=",instants[0],flush=True)
	_, header = pyAlya.Field.read(CASESTR,VARLIST,instants[0],mesh.xyz,basedir=CASEDIR)

	# Initialize the average
	avgField = pyAlya.Field(xyz = mesh.xyz, AVVEL = mesh.newArray(ndim=3), AVPRE = mesh.newArray())
	time, total_time = header.time, header.time
	pyAlya.pprint(0,"init time=",time," init total time=",total_time,flush=True)
	for instant in instants[1:]:
		field, header = pyAlya.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=CASEDIR)

#		if mpi_rank == 0: continue # Skip master

		# Compute time-weighted average 
		dt   = header.time - time # weight
		time = header.time        # sum_weights
		for v in avgField.varnames:
			avgField[v]   += field[v] * dt
		total_time += dt
		pyAlya.pprint(0,"dt=",dt," time=",time," total time=",total_time,flush=True)
	return avgField / total_time, total_time

def plane_generation(Length,nx,ny):

	# Generate partition table
	ptable = pyAlya.PartitionTable.new(1,nelems=(nx-1)*(ny-1),npoints=nx*ny)

	# Generate points
	points = np.array([
		[-Length,-Length,0.0],
		[ Length,-Length,0.0],
		[ Length, Length,0.0],
		[-Length, Length,0.0]
		],dtype='double')

	# Generate plane mesh
	return pyAlya.Mesh.plane(points[0],points[1],points[3],nx,ny,ngauss=1,ptable=ptable,create_elemList=False)

def generate_metadata(avgtime,zeta,nnode,case,angle,CASEDIR):

	# CPU info
	pyAlya.pprint(0,"tornar a activar això per calcular CPUh consumides,flush=True")
	cpu_hours = 1
	
	n_cpu = mpi_size
	#tail = subprocess.run(['tail', '-50', '{}c.log'.format(CASEDIR)],stdout=subprocess.PIPE, universal_newlines=True)
	#cpu_time = float(tail.stdout.split('\n')[3].split()[3])
	#cpu_hours = n_cpu*cpu_time/3600

	# MetaData
	metadata = {'Urban geometry code:' : [case,'i'],
	'Height of the section with respect to the floor [m]:': [zeta,'f'],
	'Wind direction w.r.t. horizontal [deg]:' : [angle,'i'],
	'Number of plane mesh points:' : [nnode,'i'],
	'Micrometeorology time average span [s]:': [avgtime,'double'],
	'Micrometeorology simulation CPU cost [CPU-h]:':[cpu_hours,'f'] ,
	'Simulation CPU number:': [n_cpu,'i'],
	'Number of output scalar variables:': [3,'i'],
	'Number of output 3D vectorial variables:': [4,'i'],
	}
	basename = 'FF4-0012-iBAM-HPC-WIND-{:02d}-{:03d}-z{:04.1f}'.format(case,angle,zeta)
	return metadata, basename

def meteo_fields(avgField):

	# Compute the gradients of the velocity and the pressure
	avgField['GRAVZ'] = mesh.gradient(avgField['AVVEL'])[:,[0,2,4,5,8]]
	avgField['GRAPZ'] = mesh.gradient(avgField['AVPRE'])[:,2]
	return avgField

def fields_h5(int_xyz,mesh,avgField,POSTDIR,metadata,basename,nanval=0.0):

	# Field generation
	pyAlya.cr_start('fields_h5',0)

	# Interpolation
	pyAlya.pprint(0,'Initiating interpolation',flush=True)
	int_field = mesh.interpolate(int_xyz,avgField,method='FEM',fact=3.,ball_max_iter=5,global_max_iter=1)

	# Create fluid mask, fluid=True, building=False
	int_field['FLUID'] = np.logical_not(np.isnan(int_field['AVVEL'][:,0])).astype(np.int32)

	# Change NaNs to a value
	for v in int_field.varnames:
		if v == 'FLUID': continue # Skip the mask
		int_field[v][~int_field['FLUID']] = nanval
	pyAlya.pprint(0,'Interpolated',flush=True)
	pyAlya.cr_stop('fields_h5',0)
	return int_field

