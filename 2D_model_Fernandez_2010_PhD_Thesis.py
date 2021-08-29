from __future__ import print_function
from dolfin import *
import sys
   
N=20
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0,1.0), 20, 20,"crossed")

V=VectorFunctionSpace(mesh,"P",1)  
V_ele=FunctionSpace(mesh,"DG",0) 

FName_str="Fernandez_plate_model/Density_"  
FExt_str=".pvd"

cnt_cell_converged=[]  

rho0=0.8
rho_val=[] 
for cell_s in cells(mesh): 
    rho_val.append(rho0)
    cnt_cell_converged.append(0)  
    
rho_min=0.01
rho_max=1.740

k=0.25

nu=0.3  

M=10000.0
   
gamma=2.0 

B=1.0 

T=25.0  
dt=0.01  
t=dt        

cnt_cells=mesh.num_cells()

tol=1E-14  

def bottom_fixed_boundary(x,on_boundary):
    return near(x[0],0,tol) and near(x[1],0,tol)
Fixed_left=Constant((0.,0.))    
bc_Fixed=DirichletBC(V,Fixed_left,bottom_fixed_boundary,method='pointwise')
 
def bottom_right_boundary(x,on_boundary):
    return near(x[1],0.0,tol) and x[0]>0.0
Roller_right=Constant(0)
bc_roller=DirichletBC(V.sub(1),Roller_right,bottom_right_boundary) #,method='pointwise')    

bcs=[bc_Fixed,bc_roller]

class Top(SubDomain):
    tol=1E-14
    def inside(self,x,on_boundary):
        return near(x[1],1.0,tol)

top=Top() 

boundries=MeshFunction('size_t',mesh,1) 
boundries.set_all(0) 
top.mark(boundries,1) 
ds=Measure('ds', domain=mesh,subdomain_data=boundries)

F=Expression("m*x[0]+c",m=-10.0,c=10.0,degree=1)

def calculate_E(updated_rho_val):
    E_updated=Function(V_ele) 
    E_array=E_updated.vector().get_local()
    for i, rho in enumerate (updated_rho_val): 
        E_array[i]=M*pow(rho,gamma)
    E_updated.vector().set_local(E_array)
    return E_updated

E0=calculate_E(rho_val) 

def calculate_Lame_coefficients(E_val):
    mu_val=(E_val)/(2*(1+nu))  
    lmbda_val=(E_val*nu)/((1+nu)*(1-2*(nu))) 
    return mu_val,lmbda_val

mu, lmbda=calculate_Lame_coefficients(E0) 


def epsilon(u):
    strain_u=0.5*(nabla_grad(u)+nabla_grad(u).T)
    return strain_u

def sigma(u,mu,lmbda):
    stress_u=lmbda*div(u)*Identity(d)+2*mu*epsilon(u)
    return stress_u 

f=Constant((0.0,0.0))  

u=TrialFunction(V)    
v=TestFunction(V)

d=u.geometric_dimension()

a = 2*mu*inner(epsilon(u),epsilon(v))*dx + lmbda*dot(div(u),div(v))*dx
L=dot(f,v)*dx+v[1]*F*ds(1)

def calculate_SED(epsilon_val,sigma_val):                
    SED_val=0.5*inner(sigma_val,epsilon_val)    
    
    SED_plot=project(SED_val, V_ele)
    SED_values=SED_plot.vector().get_local()
     
    return(SED_values,SED_plot)

def calculate_Density_change(rho_vals,SED):
    import numpy as np
    change_in_density=[]
       
    rho_plot=Function(V_ele)
    rho_array=rho_plot.vector().get_local() 
    
    stimulus_plot=Function(V_ele)
    stimulus=stimulus_plot.vector().get_local() 
    
    for i, cells_i in enumerate (cells(mesh)):  
        
        if cnt_cell_converged[i]==0: 
            stimulus[i]=SED[i]/rho_vals[i]
                                                                          
            change_in_density.append(B*(stimulus[i]-k)) 
            
            rho_array[i]=rho_vals[i]+(dt*change_in_density[i])
        else: 
            change_in_density.append(0)
            rho_array[i]=rho_vals[i] 
        
        rho_plot.vector().set_local(rho_array)        
    return rho_plot,rho_array,cnt_cell_converged,change_in_density  

def check_convergence(density_values,change_in_density):
    
    tol=1E-14
    for i, density_val in enumerate (density_values):
        if density_val<=rho_min: 
            cnt_cell_converged[i]=1
            density_values[i]=rho_min
        elif density_val>=rho_max:  
            cnt_cell_converged[i]=1
            density_values[i]=rho_max
        elif near(change_in_density[i],0.0,tol):
            cnt_cell_converged[i]=1  
    return density_values

def create_rho_plot(updated_rho_array):
    V_ele=FunctionSpace(mesh,"DG",0)
    rho_ele=Function(V_ele) 
    rho_ele_array=rho_ele.vector().get_local()
    
    i=0
    for cell_s in cells(mesh):  
        rho_ele_array[i]=updated_rho_array[i]
        i=i+1
    rho_ele.vector().set_local(rho_ele_array)
    return rho_ele

updated_rho_val=rho_val

u=Function(V)  
cnt_freq=0.0

day=0 
while day<=T:    
    solve(a==L,u,bcs)
    cnt_freq=cnt_freq+0.01 
    
    #calculate stress-strain from nodal solution u
    epsilon_val=epsilon(u)    
    sigma_val=sigma(u,mu,lmbda)
    
    #calculate SED
    SED,SED_plt=calculate_SED(epsilon_val,sigma_val)
            
    #calculate updated density
    updated_rho, updated_rho_val,cnt_cell_converged,change=calculate_Density_change(updated_rho_val,SED)      
    
    #Calculate updated E
    E_updated_t=calculate_E(updated_rho_val) 
    E0.assign(E_updated_t)
        
    #calculate updated mu and lmbda    
    mu, lmbda=calculate_Lame_coefficients(E0)    
    
    if cnt_freq>=1:
        cnt_freq=0.0                
        day = day+1
                
        rho_after_convergence=check_convergence(updated_rho_val,change)

    if sum(cnt_cell_converged)==cnt_cells:
        t=T+1
        day=t
    elif day==T: 
        print("Specified days computed")
        t=T+1
        day=t
    else:
        t=t+dt 

FName_str_femur="Density_"
ele_rho=create_rho_plot(rho_after_convergence)

fName=FName_str+str(day)+FExt_str
File(fName)<<ele_rho   
