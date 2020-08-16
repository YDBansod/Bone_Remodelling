"""
@author: Yogesh Deepak Bansod
Implementing the electro-mechanical bone remodelling model based on Alvarado et al. 2012
"""
from fenics import *
import numpy as np

#************************************CREATE MESH AND FUNCTIONAL SACE************************************#      
N=80
mesh=UnitSquareMesh.create(N,N,CellType.Type.quadrilateral)

V_mech=VectorFunctionSpace(mesh,"CG",1)  
V_elemental=FunctionSpace(mesh,"DG",0)   
V_elect=FunctionSpace(mesh,"CG",1) 

#Output file path
Fname="Alvarado_2012/lmbda_density_plot_"
Fext=".pvd"

#************************************MODEL PARAMETERS************************************#
cnt_cell_converged=[]
lmbda_val=[] 
E0=6400             # N/cm2

for cell_s in cells(mesh): 
    cnt_cell_converged.append(0)
    lmbda_val.append(1.0)
    
lmbda_min=0.0125
lmbda_max=2.175 
U_ref_mech=0.08   # N/cm2 
nu=0.3

n=2.0
k_mech=0.6125

m=1.5486
k_elect=0.0  
eps_0=8.85E-14  #permitivity of free space F/cm

rho0=0.0008         # kg/cm^3
eps=1050*(rho0**m)  

voltage_1=Constant(100.0)  #voltage applied to right edge
voltage_2=Constant(0.0)    #voltage applied to bottom edge
U_ref_elect=0.08           # N/cm2 

#time discretization parmeters
Time=200
dt=0.1
t_cnt=dt

#************************************DEFINE BOUNDARY CONDITIONS************************************#
tol=1E-14

# Mechanical BC
def bottom_fixed_boundary(x,on_boundary):
    return near(x[0],0.0,tol) and near(x[1],0.0,tol)
Fixed_left=Constant((0.,0.)) 
bc_Fixed=DirichletBC(V_mech,Fixed_left,bottom_fixed_boundary,method='pointwise')

def bottom_right_boundary(x,on_boundary):
    return near(x[1],0.0,tol) and x[0]>0.0
Roller_right=Constant(0.0)
bc_Roller=DirichletBC(V_mech.sub(1),Roller_right,bottom_right_boundary,method='pointwise')   

bcs_mech=[bc_Fixed,bc_Roller]

#redefine the boundary integrals
class Top(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[1],1.0,tol) and on_boundary
top=Top()

class Left(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[0],0.0,tol) and on_boundary
left=Left()

class Bottom(SubDomain):
    def inside(self,x,on_boundary):
        return near(x[1],0.0,tol) and on_boundary
bottom=Bottom()

boundries=MeshFunction('size_t',mesh,1) 
boundries.set_all(0)  
top.mark(boundries,1)
left.mark(boundries,2)
bottom.mark(boundries,3)

ds=Measure('ds', domain=mesh,subdomain_data=boundries)

F=Expression("m*x[0]+c",m=-100.0,c=100.0,degree=1)  #N/cm2

# Electrical BC
bc_right_vlt=DirichletBC(V_elect,voltage_1,boundries,0)
bc_bottom_vlt=DirichletBC(V_elect,voltage_2,boundries,3)
bcs_elect=[bc_right_vlt,bc_bottom_vlt]

#Matrix of linear elasticity
def calculate_C0(E0_val,nu):
    T1 = E0_val/(1-nu**2)
    D = np.array([[T1, T1*nu, 0.0],
                  [T1*nu, T1, 0.0],
                  [0.0, 0.0, (T1*(1-nu))/2.0],
                 ])
    return D

#strain in vigot format
def epsilon(u):
    return as_vector([u[i].dx(i) for i in range(2)] +[u[i].dx(j) + u[j].dx(i) for (i,j) in [(0,1)]])

#Functions for calculating mechanical stimulus
def calculate_lmbda_plot(lmbda):
    lmbda_plot=Function(V_elemental) 
    lmbda_array=lmbda_plot.vector().get_local()
    for i, lm_val in enumerate (lmbda): 
        lmbda_array[i]=lm_val
    lmbda_plot.vector().set_local(lmbda_array)    
    return lmbda_plot

def calculate_U_mech(u,C0_mat):
    U_mech=0.5*inner(epsilon(u),C0_mat*epsilon(u))  
    U_mech_plot=project(U_mech,V_elemental) 
    return(U_mech_plot)

def change_in_lmbda_mech(u,C0,lmbda):
    U_mech_plt=calculate_U_mech(u,C0)
    U_mech_array=U_mech_plt.vector().get_local()
    lmbda_array=lmbda.vector().get_local()
    dlmbda_mech=[] 
    for i, lm_val in enumerate (lmbda_array):
        T1=k_mech
        T2=(lm_val**(n-1.0))
        T3=(U_mech_array[i]/U_ref_mech)
        mech_stimulus=T1*((T2*T3)-1.0)   
        dlmbda_mech.append(mech_stimulus)
    return dlmbda_mech

# Functions for calculating electrical stimulus    
def calculate_U_elect(phi):
    U_elect=0.5*eps_0*((-grad(phi))**2.0)  
    U_elect_nodal=project(U_elect,V_elect)
    U_elect_plot=project(U_elect_nodal,V_elemental)  
    return U_elect_plot

def change_in_lmbda_elect(phi,lmbda):
    temp_arr=[]
    U_elect_plt=calculate_U_elect(phi)  
    U_elect_array=U_elect_plt.vector().get_local()
    lmbda_arr=lmbda.vector().get_local()
    dlmbda_elect=[]
    for i, lm_val in enumerate (lmbda_arr):
        E1=k_elect
        E2=(eps*(lm_val**(m-1)))
        E3=(U_elect_array[i])/(U_ref_elect)
        elect_stimulus=E1*E2*E3
        dlmbda_elect.append(elect_stimulus)
    return dlmbda_elect 
    
# Calculate updated density
def calculate_new_lmbda(lmbda_plt, mechanical_stimulus,Elect_stimulus):
    new_lmbda=[]     
    lmbda_arr=lmbda_plt.vector().get_local()
    for i, lm_val in enumerate (lmbda_arr):
        if cnt_cell_converged[i]!=1:  
            dlm=mechanical_stimulus[i] + Elect_stimulus[i] 
            lmbda_updated=lmbda_arr[i] + (dt*dlm)
            new_lmbda.append(lmbda_updated)            
            tol=9.0E-9
            if lmbda_updated<=lmbda_min:
                new_lmbda[i]=lmbda_min
                cnt_cell_converged[i]=1
            elif lmbda_updated>=lmbda_max:
                new_lmbda[i]=lmbda_max
                cnt_cell_converged[i]=1
            elif near(dlm,0.0,tol):
                cnt_cell_converged[i]=1
        else: new_lmbda.append(lm_val)  
    updated_lm_plt=calculate_lmbda_plot(new_lmbda)     
    return updated_lm_plt
    
#***********************************************************************************************************#        
u=TrialFunction(V_mech)    
v1=TestFunction(V_mech)
v2=TestFunction(V_mech)

b=Constant((0,0))    

C0=calculate_C0(E0,nu)
C0_mat=as_matrix(C0)

lmbda_plot=calculate_lmbda_plot(lmbda_val)

#Variation formulation - mechanical
a_mech=inner(epsilon(v1),(lmbda_plot**n)*C0_mat*epsilon(u))*dx
L_mech=dot(b,v2)*dx + v2[1]*F*ds(1) 

u=Function(V_mech)

#variation formulation - electrical
phi=TrialFunction(V_elect)
w=TestFunction(V_elect)
n_elect=FacetNormal(mesh)

F_elect=((eps*(lmbda_plot**m))*inner(grad(w),grad(phi)))*dx  +\
        inner(grad(w),((eps_0*eps*(lmbda_plot**m))*(-grad(phi))))*dx 
           
a_elect=lhs(F_elect)  
L_elect=rhs(F_elect)

phi=Function(V_elect)

cnt_cells=mesh.num_cells()
while t_cnt<=Time:    
    solve(a_mech==L_mech,u,bcs_mech)
    Mech_stimulus = change_in_lmbda_mech(u,C0_mat,lmbda_plot)

    solve(a_elect==L_elect,phi,bcs_elect)
    Elect_stimulus = change_in_lmbda_elect(phi,lmbda_plot)

    updated_lm=calculate_new_lmbda(lmbda_plot, Mech_stimulus, Elect_stimulus)
       
    lmbda_plot.assign(updated_lm)
    
    if sum(cnt_cell_converged)==cnt_cells:
        t_cnt=Time+1
    else:
        t_cnt=t_cnt+dt

Fname1=Fname+str(t_cnt)+Fext
File(Fname1)<<lmbda_plot         