from __future__ import print_function, division

import mpi4py 
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from stl import mesh
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pyAlya

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def solid_perimeter_generation(triangles) -> None:

        resolution=0.5 #new point spacing 
        perimeter=[]
        for tri in triangles:
                v1=np.zeros(3)
                v2=np.zeros(3)
                if tri[0][2]!=0: 
                        v1=tri[1]
                        v2=tri[2]
                elif tri[1][2]!=0:
                        v1=tri[0]
                        v2=tri[2]
                elif tri[2][2]!=0:
                        v1=tri[0]
                        v2=tri[1]
                
                nsteps=int(np.linalg.norm(v2-v1)/resolution)
                if nsteps>1: 
                        dLam=1.0/nsteps
                        Lam=0.0
                        perimeter.append(v1)
                        for i in range(nsteps):
                                Lam+=dLam
                                x=v1+Lam*(v2-v1)
                                perimeter.append(x)
                        perimeter.append(v2)
                else: 
                        perimeter.append(v1)
                        perimeter.append(v2)

        return np.array(perimeter)

def geometrical_data_extractor(target_mesh,horizontal_triangles,vertical_triangles):

        h_triangles=np.copy(horizontal_triangles)
        h_triangles[:,:,2]=0
        
        perimeter=solid_perimeter_generation(vertical_triangles)

        points=np.copy(target_mesh)
        points[:,2]=0.0

        size_G=points.shape[0]
        size_L=int(size_G/mpi_size)
        ini_idx=size_L*mpi_rank
        final_idx=size_L*(mpi_rank+1)-1
        if mpi_rank==(mpi_size-1): final_idx=size_G-1

        subset=points[ini_idx:final_idx+1]
        
        mask_L=np.zeros(subset.shape[0])
        height_L=np.zeros(subset.shape[0])
        distance_L=np.zeros(subset.shape[0])

        for idx,p in enumerate(subset):
                tri_idx=isIn(p,h_triangles)
                if tri_idx<0:
                        mask_L[idx]=1
                        height_L[idx]=0
                else:
                        mask_L[idx]=0
                        height_L[idx]=horizontal_triangles[tri_idx][0][2]

                if mask_L[idx]==1:
                        distance_L[idx]=wall_distance(p,perimeter)

        recv_buff_mask = mpi_comm.allgather(mask_L)
        recv_buff_height = mpi_comm.allgather(height_L)
        recv_buff_distance = mpi_comm.allgather(distance_L)
        
        mask_G=recv_buff_mask[0]
        height_G=recv_buff_height[0]
        distance_G=recv_buff_distance[0]
        for i in range(mpi_size-1):
                mask_G=np.concatenate((mask_G,recv_buff_mask[i+1]),axis=0)
                height_G=np.concatenate((height_G,recv_buff_height[i+1]),axis=0)
                distance_G=np.concatenate((distance_G,recv_buff_distance[i+1]),axis=0)

        '''                        
        mask_G=np.reshape(mask_G,(101,101))
        height_G=np.reshape(height_G,(101,101))
        distance_G=np.reshape(distance_G,(101,101))
        if mpi_rank==0: save_scalarfield(mask_G,"mask.png")
        if mpi_rank==0: save_scalarfield(height_G,"height.png")
        if mpi_rank==0: save_scalarfield(distance_G,"distance.png")
        '''

        fields = pyAlya.Field(xyz = points)

        fields['MASK'] = mask_G
        fields['HEGT'] = height_G
        fields['WDST'] = distance_G

        return fields

def wall_distance(point,perimeter):

        point_vec=np.tile(point,(perimeter.shape[0],1))

        dist=np.linalg.norm(perimeter-point_vec,axis=1)
        return np.amin(dist)

def isIn(point,triangles):

        point_vec=np.tile(point,(triangles.shape[0],1))

        v0 =triangles[:,0,:] 
        v1 =triangles[:,1,:] 
        v2 =triangles[:,2,:] 

        S=0.5*np.linalg.norm(np.cross(v1-v0,v2-v0,axis=1),axis=1)
        S1=0.5*np.linalg.norm(np.cross(v0-point_vec,v1-point_vec,axis=1),axis=1)
        S2=0.5*np.linalg.norm(np.cross(v1-point_vec,v2-point_vec,axis=1),axis=1)
        S3=0.5*np.linalg.norm(np.cross(v2-point_vec,v0-point_vec,axis=1),axis=1)

        isIn=abs((S1+S2+S3)-S)<0.001
        
        output=np.where(isIn==True)[0]

        return output[0] if len(output) > 0 else -1

def display_scalarfield(plane):

        plt.imshow(plane, cmap='plasma')
        plt.show()

def save_scalarfield(plane,filename):

        plt.imsave(filename, plane)



def display_points(points):
        x=points[:,0]
        y=points[:,1]
        z=points[:,2]

        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,z,s=1.0)
        plt.show()

def display_points_plane(points,plane):
        x=points[:,0]
        y=points[:,1]
        z=points[:,2]

        x1=plane[0].flatten()
        y1=plane[1].flatten()
        z1=np.zeros(len(x1))

        fig=plt.figure()
        ax = fig.add_subplot()
        ax.scatter(x,y,s=0.1)
        ax.scatter(x1,y1,s=1.0,c='#ff7f0e')
        plt.show()

def display_stl(mesh):
        # Create a new plot

        figure = plt.figure()
        axes = figure.add_subplot(projection='3d')
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

        # Auto scale to the mesh size
        scale = mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)
        # Show the plot to the screen
        plt.show()

def display_triangles(triangles):
        points=triangles.reshape(triangles.shape[0]*3,-1)
        display_points(points)


        
def rotate_stl(stl,angles,center=np.array([0, 0, 0])):
        
		alpha = math.pi*angles[2]/180.0
		beta  = math.pi*angles[1]/180.0
		gamma = math.pi*angles[0]/180.0
        
		R = np.ndarray(shape=(3,3))
        
		R[0][0] = math.cos(alpha)*math.cos(beta)
		R[1][0] = math.cos(alpha)*math.sin(beta)*math.sin(gamma)-math.sin(alpha)*math.cos(gamma)
		R[2][0] = math.cos(alpha)*math.sin(beta)*math.cos(gamma)+math.sin(alpha)*math.sin(gamma)
		R[0][1] = math.sin(alpha)*math.cos(beta)
		R[1][1] = math.sin(alpha)*math.sin(beta)*math.sin(gamma)+math.cos(alpha)*math.cos(gamma)
		R[2][1] = math.sin(alpha)*math.sin(beta)*math.cos(gamma)-math.cos(alpha)*math.sin(gamma)
		R[0][2] = -math.sin(beta)
		R[1][2] = math.cos(beta)*math.sin(gamma)
		R[2][2] = math.cos(beta)*math.cos(gamma)

		for tri, triangle in enumerate(stl):
			centers=np.tile(center,(3,1))
			stl[tri] = np.dot(triangle-centers,R)+centers
		return stl

def move_stl(stl,displacement=np.array([0, 0, 0])):
		for tri, triangle in enumerate(stl):
			displacements=np.tile(displacement,(3,1))
			stl[tri] = triangle+displacements
		
		return stl


def geometrical_magnitudes(STL_FILE,target_mesh,angle=0.0,stl_scale=1.0):

		my_mesh = mesh.Mesh.from_file(STL_FILE)

		triangles = stl_scale*my_mesh.vectors
		triangles = rotate_stl(triangles,[0, 0,angle])
		triangles = move_stl(triangles,[-650.0,-650.0,0.0])
		#display_triangles(triangles)

		horizontal_triangles=triangles[(triangles[:,0,2]==triangles[:,1,2]) & (triangles[:,0,2]==triangles[:,2,2]) & (triangles[:,0,2]!=0)]
		vertical_triangles=triangles[((triangles[:,0,2]==0) & (triangles[:,1,2]==0) & (triangles[:,2,2]!=0)) | \
               		                ((triangles[:,0,2]==0) & (triangles[:,1,2]!=0) & (triangles[:,2,2]==0)) | \
                        	        ((triangles[:,0,2]!=0) & (triangles[:,1,2]==0) & (triangles[:,2,2]==0))]

		#triangles=np.array([[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],[[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0]]])
		#display_triangles(triangles)

		fields=geometrical_data_extractor(target_mesh,horizontal_triangles,vertical_triangles)

		return fields
