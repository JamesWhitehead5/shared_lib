"""
Python author: Luocheng Huang 
Matlab author: Shane Colburn
Version      : 1.0   
"""

from matlib import *
import h5py
pi = np.pi


# lumerical_propagator_angular.m

# Implements the Fresnel diffraction integral by the impulse response
# method
# Set the operating wavelengths
lambdas=[i *1e-09 for i in [460,540,700]]
# Specify the input filename
FOLDER=''

FILENAMES=('outputData_blue.mat','outputData_green.mat','outputData_red.mat')

monitor_z=6e-06

image_plane=4.1e-05

n_x=1068

n_y=1068
x_min=- 1.6e-05

x_max=1.6e-05
y_min=- 1.6e-05
y_max=1.6e-05
dx=(x_max - x_min) / n_x
dy=(y_max - y_min) / n_y
A=copy(x_max)
z_2D_plane=image_plane - monitor_z

z_start=6e-06 - monitor_z
z_final=0.00012 - monitor_z

on_axis_field_points=100

lf=length(FILENAMES)
lp=3

fig, axes = plt.subplots(nrows=lp, ncols=lf, figsize=(15, 12), dpi = 300)


for i in arange(0,lf-1).reshape(-1):
    f = h5py.File(FILENAMES[i], 'r')
    
    complx = lambda x: x[0] + x[1]*1j 
    complx = np.vectorize(complx)
    
    Ex=complx(np.array(f['E/E'])[0])
    Ey=complx(np.array(f['E/E'])[1])
    Ez=complx(np.array(f['E/E'])[2])
    
    xlist=linspace(x_min,x_max,n_x)
    ylist=linspace(x_min,x_max,n_y)
    Ex2=reshape(Ex,[n_x,n_y])
    Ey2=reshape(Ey,[n_x,n_y])
    Ez2=reshape(Ez,[n_x,n_y])
    
    initial=abs(Ex2) ** 2 + abs(Ey2) ** 2 + abs(Ez2) ** 2
    x_axis=linspace(dot(x_min,1000000.0),dot(x_max,1000000.0),n_x)
    y_axis=linspace(dot(y_min,1000000.0),dot(y_max,1000000.0),n_y)
    # lumerical_propagator_angular.m:47
    #imagesc(x_axis, y_axis, initial);
    #xlabel('x-axis (micron)');
    #ylabel('y-axis (micron)');
    #title('Monitor Plane Total Intensity');

    # Set the cartesian grid
    n=length(xlist)
    col=ones(n,1)
    xx=kron(xlist,col)
    yy=xx.T
    E_freq_x=fftshift(fft2(Ex2))
    E_freq_y=fftshift(fft2(Ey2))
    
    # Define reciprocal space cartesian grid
    k_xlist=dot(dot(2,pi),linspace(dot(- 0.5,n) / (dot(2,A)),dot(0.5,n) / (dot(2,A)),n))
    col=ones(n,1)
    k_x=kron(k_xlist,col)
    k_y=k_x.T
    k=dot(2,pi) / lambdas[i]
    k_z=sqrt(k ** 2 - k_x ** 2 - k_y ** 2+0j)
    H_freq=exp(dot(dot(1j,k_z),z_2D_plane))
    
    # Loop to find the field value along the propagation axis
    prop_axis_field=zeros(1,on_axis_field_points)
    z_set=linspace(z_start,z_final,on_axis_field_points)
    x0_plane=zeros(length(ylist),on_axis_field_points)
    y0_plane=zeros(length(xlist),on_axis_field_points)
    
    for j in arange(0,on_axis_field_points-1).reshape(-1):
        # Set the propagation distance
        z=z_set[j]
        
        # Reciprocal space propagator
        H_freq=exp(dot(dot(1j,k_z),z))
        
        # Perform the convolution in the Fourier domain
        Ix_out=abs(ifft2(multiply(E_freq_x,H_freq))) ** 2
        Iy_out=abs(ifft2(multiply(E_freq_y,H_freq))) ** 2
        I_tot_out = Ix_out + Iy_out
        
        # Pick off the vector to the form the x0_plane and y0_plane profile
        x0_plane[:,j]=abs(I_tot_out[floor((length(xlist)) / 2),:])
        y0_plane[:,j]=abs(I_tot_out[:,floor((length(xlist)) / 2)])
        
    H_freq=exp(dot(dot(1j,k_z),z_2D_plane))
    Ix_out=abs(ifft2(multiply(E_freq_x,H_freq))) ** 2
    Iy_out=abs(ifft2(multiply(E_freq_y,H_freq))) ** 2
    modE=Ix_out + Iy_out
    
    # start to plot!
    # z = F plane
    modE_norm=copy(modE)
    um = 1E6
    nm = 1E9
    pos = axes[0, i].imshow(modE_norm, extent=[x_min*um, x_max*um, y_min*um, y_max*um])
    axes[0, i].set_title('Intensity at '+str(lambdas[i]*nm)+'nm')
    axes[0, i].set_xlabel('x-axis (micron)')
    axes[0, i].set_ylabel('y-axis (micron)')
    fig.colorbar(pos, ax=axes[0, i])

    # yz plane
    modE_norm=copy(modE)
    pos = axes[1, i].imshow(x0_plane, interpolation='nearest', aspect='auto', 
                            extent=[(z_start + monitor_z)*um, 
                                    (z_final + monitor_z)*um, 
                                    x_min*um, x_max*um])
#    axes[1, i].set_title('Intensity at '+str(lambdas[i]*nm)+'nm')
    axes[1, i].set_xlabel('z-axis (micron)')
    axes[1, i].set_ylabel('x-axis (micron)')
    fig.colorbar(pos, ax=axes[1, i])    


    # yz plane
    modE_norm=copy(modE)
    pos = axes[2, i].imshow(y0_plane, interpolation='nearest', aspect='auto', 
                            extent=[(z_start + monitor_z)*um, 
                                    (z_final + monitor_z)*um, 
                                    y_min*um, y_max*um])
#    axes[2, i].set_title('Intensity at '+str(lambdas[i]*nm)+'nm')
    axes[2, i].set_xlabel('z-axis (micron)')
    axes[2, i].set_ylabel('y-axis (micron)')
    fig.colorbar(pos, ax=axes[2, i])    

#    print max_value_y,max_index_y=max(y0_plane(round(length(y0_plane) / 2),arange()),nargout=2)
#    print max_position_y=z_axis(max_index_y)
#    print max_value_x,max_index_x=max(x0_plane(round(length(x0_plane) / 2),arange()),nargout=2)
#    print max_position_x=z_axis(max_index_x)

    fig.savefig('propagation.png', dpi = 300)

