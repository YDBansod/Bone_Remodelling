from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from fenics import *
import sys
#************************************CREATE MESH AND FUNCTION SACE************************************#      
#creating a square plate with quadrilateral mesh
N=40 
mesh=UnitSquareMesh.create(N,N,CellType.Type.quadrilateral)

V=VectorFunctionSpace(mesh,"P",1)  
V_ele=FunctionSpace(mesh,"DG",0) 

#************************************MODEL PARAMETERS************************************#
cnt_cell_converged=[]  
FName_str="Weinans_40_40/Density_"

rho0=0.8  # g/cm^3 
rho_val=[] 
for cell_s in cells(mesh): 
    rho_val.append(rho0)
    cnt_cell_converged.append(0)  
    
rho_min=0.01  # g/cm^3 
rho_max=1.74  # g/cm^3 

k=0.25 # J/g 
B=1    
nu=0.3  

M=100 
    
gamma=2.0

#time stepping variables
T=100  #final time
dt=1  
t=dt        

cnt_cells=mesh.num_cells()

#************************************DEFINE BOUNDARY CONDITIONS************************************#
tol=1E-14  

def bottom_fixed_boundary(x,on_boundary):
    return near(x[0],0,tol) and near(x[1],0,tol)

Fixed_left=Constant((0.,0.))    
bc_Fixed=DirichletBC(V,Fixed_left,bottom_fixed_boundary,method='pointwise')
 
def bottom_right_boundary(x,on_boundary):
    return near(x[1],0,tol) and x[0]>0

Roller_right=Constant(0)
bc_roller=DirichletBC(V.sub(1),Roller_right,bottom_right_boundary) #,method='pointwise')    

#defining the list of boundary conditions
bcs=[bc_Fixed,bc_roller]

#************************************MARK SUBDOMAIN FOR TOP EDGE************************************#
class Top(SubDomain):
    tol=1E-14
    def inside(self,x,on_boundary):
        return near(x[1],1,tol)

top=Top() #creating an obje   ct of the class Top

boundries=MeshFunction('size_t',mesh,1)
boundries.set_all(0)  #initialise all to zero index
top.mark(boundries,1)  #overwrite the indexing for the top boundary to 1
ds=Measure('ds', domain=mesh,subdomain_data=boundries)

F=Expression("m*x[0]+c",m=-10,c=10,degree=1)

#************************************COMPUTE MATERIAL PARAMETERS************************************#
def calculate_E(updated_rho_val):
    E_updated=Function(V_ele) 
    E_array=E_updated.vector().get_local()
    for i, rho in enumerate (updated_rho_val): 
        E_array[i]=M*pow(rho,gamma)
    E_updated.vector().set_local(E_array)
    return E_updated

E0=calculate_E(rho_val) 

#************************************COMPUTE THE LAME'S COEFFICIENTS************************************#
def calculate_Lame_coefficients(E_val):#E,nu
    mu_val=(E_val)/(2*(1+nu))  #calculate mu and append to the array
    lmbda_val=(E_val*nu)/((1+nu)*(1-2*(nu))) #calculate lmbda and append to the array 
    return mu_val,lmbda_val

mu, lmbda=calculate_Lame_coefficients(E0) #call since value used for t=0

#************************************DEFINE STRESS STRAIN TENSOR************************************# 
def epsilon(u):
    strain_u=0.5*(grad(u)+grad(u).T)
    return strain_u

def sigma(u,mu,lmbda):
    stress_u=lmbda*div(u)*Identity(d)+2*mu*epsilon(u)
    return stress_u 

#************************************DEFINE VARIATION FORMULATION************************************#
f=Constant((0,0))  

u=TrialFunction(V)    
v=TestFunction(V)

d=u.geometric_dimension()

a = 2*mu*inner(epsilon(u),epsilon(v))*dx + lmbda*dot(div(u),div(v))*dx
L=dot(f,v)*dx+v[1]*F*ds(1)

#************************************COMPUTE STRAIN ENERGY DENSITY (SED)************************************#
def calculate_SED(epsilon_val,sigma_val):                
    SED_val=0.5*inner(sigma_val,epsilon_val)    
    SED_plot=project(SED_val, V_ele)
    SED_values=SED_plot.vector().get_local() #nodal SED values
    return(SED_values,SED_plot)

#************************************CHECK FOR CONVERGENCE CRITERIA************************************#
def calculate_Density_change(rho_vals,SED):
    import numpy as np
    change_in_density=[]
    tol=1E-6 
    rho_plot=Function(V_ele)
    rho_array=rho_plot.vector().get_local() 
    stimulus_plot=Function(V_ele)
    stimulus=rho_plot.vector().get_local()   
    for i, SED_val in enumerate (SED): 
        if cnt_cell_converged[i]==0: 
            stimulus[i]=SED_val/rho_vals[i]         
            change_in_density.append(B*(stimulus[i]-k))                             
            rho_array[i]=rho_vals[i]+dt*change_in_density[i]
            
            #check for convergence of the cell
            if rho_array[i]<= rho_min:  
                cnt_cell_converged[i]=1
                rho_array[i]=rho_min
                
            elif rho_array[i]>=rho_max:  #bocome cortical
                cnt_cell_converged[i]=1
                rho_array[i]=rho_max
                
            elif near(change_in_density[i],0.0,tol): #in equilibrium
                cnt_cell_converged[i]=1                
        else: 
            change_in_density.append(0)
            rho_array[i]=rho_vals[i] 
            stimulus[i]=SED_val/rho_vals[i]
        
        rho_plot.vector().set_local(rho_array)   
    
    return rho_plot,rho_array,cnt_cell_converged  

#************************************TIME STEPPING LOOP************************************#
updated_rho_val=rho_val

FExt_str=".pvd"
u=Function(V) 
cnt_cells=mesh.num_cells()

while t<=T :       
    solve(a==L,u,bcs)
    
    epsilon_val=epsilon(u)    
    sigma_val=sigma(u,mu,lmbda)
    
    SED,SED_plt=calculate_SED(epsilon_val,sigma_val)
    
    updated_rho, updated_rho_val,cnt_cell_converged=calculate_Density_change(updated_rho_val,SED)      
    
    if t==T:
        fName=FName_str+str(t)+FExt_str
        File(fName)<<updated_rho
       
    E_updated_t=calculate_E(updated_rho_val) 
    E0.assign(E_updated_t)
    
    mu, lmbda=calculate_Lame_coefficients(E0)    
    if sum(cnt_cell_converged)==cnt_cells:
        t=T+1
    else:
        t=t+dt 
