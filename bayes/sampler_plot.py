import matplotlib.pyplot as plt
import numpy as np

def sampler_plot(sampler,data,figdir = '../plots/',suffix='',name=None):
    x = data['x']
    y = data['y']
    z = data['z']
    x_err = data['x_err']
    y_err = data['y_err']
    z_err = data['z_err']

    plt.figure(figsize=(6,8))
    plt.subplot(321)
    plt.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]),\
               range=[0,2],bins=100)
    plt.xlabel(r'$R_{32}$')
    plt.subplot(322)
    plt.hist(np.tan(sampler.flatchain[:,1]),range=[0,2],bins=100)
    plt.xlabel(r'$R_{21}$')

    plt.subplot(323)
    plt.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
    plt.scatter(x,y,marker='.')
    plt.xlabel('CO(1-0)')
    plt.ylabel('CO(2-1)')
    testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
    xoff = 0 #np.median(sampler.flatchain[:,2])
    plt.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*(testx+xoff),color='r')

    plt.subplot(324)
    plt.errorbar(y,z,xerr=y_err,yerr=y_err,fmt=None,marker=None,mew=0)
    plt.scatter(y,z,marker='.')
    plt.xlabel('CO(2-1)')
    plt.ylabel('CO(3-2)')
    testx = np.linspace(np.nanmin(y),np.nanmax(y),10)
    plt.plot(testx,testx/np.tan(np.median(sampler.flatchain[:,0]))/\
               np.sin(np.median(sampler.flatchain[:,1])),color='r')
    
    plt.subplot(325)
    plt.hexbin(np.tan(sampler.flatchain[:,1]),
             1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]))
    plt.xlabel(r'$R_{21}$')
    plt.ylabel(r'$R_{32}$')

    plt.subplot(326)
    plt.hexbin(np.tan(sampler.flatchain[:,1]),sampler.flatchain[:,2])
    plt.xlabel(r'$R_{21}$')
    plt.ylabel(r'Offset')
    plt.tight_layout()
    plt.savefig(figdir+name+suffix+'.pdf',format='pdf',
              orientation='portrait')
    plt.close()
    plt.clf()

def sampler_plot2d(sampler,figdir = '../plots/',suffix='',
                   xoff=None,name=None,nLines=10):
    plt.figure(figsize=(6,6))
    plt.subplot(221)
    plt.hist(np.tan(sampler.flatchain[:,0]),range=[0,2],bins=100)
    plt.xlabel(r'$R_{21}$')

    plt.subplot(222)
    plt.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
    plt.scatter(x,y,marker='.')
    plt.xlabel('CO (lower)')
    plt.ylabel('CO (upper)')
    testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
    UseXoff = sampler.flatchain.shape[1]>6

    if xoff==None:
        if UseXoff:
            xoff = np.median(sampler.flatchain[:,1])
        else:
            xoff = 0

    sshape = sampler.flatchain.shape
    for line in np.arange(nLines):
        index = int(np.random.rand(1)*sshape[0])
        if UseXoff:
            xoff = sampler.flatchain[index,1]
        else:
            xoff=0
        plt.plot(testx,np.tan(sampler.flatchain[index,0])*(testx+xoff),
            alpha=0.3,color='gray')

    plt.plot(testx,np.tan(np.median(sampler.flatchain[:,0]))*
           (testx+xoff),color='r')
    plt.ylim((np.nanmin(y),np.nanmax(y)))
    plt.xlim((np.nanmin(x),np.nanmax(x)))
    
    sigma = (sampler.flatchain[:,1]**2+sampler.flatchain[:,2]**2)**0.5
    plt.subplot(223)
    plt.hexbin(np.tan(sampler.flatchain[:,0]),sigma)
    plt.xlabel(r'$R$')
    plt.ylabel('Scatter')

    plt.subplot(224)
    if UseXoff:
        plt.hist(sampler.flatchain[:,4])
    else:
        plt.hist(sampler.flatchain[:,3])
    plt.xlabel(r'$f_{bad}$')

    plt.savefig(figdir+name+suffix+'.pdf',format='pdf',
        orientation='portrait')

    plt.tight_layout()        
    plt.close()
    plt.clf()



def sampler_plot_mixture(sampler,data,figdir = '../plots/',suffix='',name=None,badprob=None):
    x = data['x']
    y = data['y']
    z = data['z']
    x_err = data['x_err']
    y_err = data['y_err']
    z_err = data['z_err']
    xoff=0
    if sampler.flatchain.shape[1]==7:
        xoff = np.median(sampler.flatchain[:,2])

    plt.figure(figsize=(6,8))
    plt.subplot(321)
    plt.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]),\
               range=[0,2],bins=100)
    plt.xlabel(r'$R_{32}$')
    plt.subplot(322)
    plt.hist(np.tan(sampler.flatchain[:,1]),range=[0,2],bins=100)
    plt.xlabel(r'$R_{21}$')

    plt.subplot(323)
    plt.scatter(x,y,marker='o',c=badprob,edgecolors='none',zorder=100,cmap='copper_r')
    plt.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker='o',mew=0,ecolor='black',c=badprob,alpha=0.3)
    plt.xlabel('CO(1-0)')
    plt.ylabel('CO(2-1)')
    testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
    plt.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*(testx+xoff),color='r')

    plt.subplot(324)
    plt.errorbar(y,z,xerr=y_err,yerr=y_err,fmt=None,marker=None,mew=0,c=badprob,alpha=0.3,ecolor='black')
    plt.scatter(y,z,marker='o',c=badprob,edgecolors='none',zorder=100,cmap='copper_r'
                )
    plt.xlabel('CO(2-1)')
    plt.ylabel('CO(3-2)')
    testx = np.linspace(np.nanmin(y),np.nanmax(y),10)
    plt.plot(testx,testx/np.tan(np.median(sampler.flatchain[:,0]))/\
               np.sin(np.median(sampler.flatchain[:,1])),color='r')
    
    plt.subplot(325)
    plt.hexbin(np.tan(sampler.flatchain[:,1]),
             1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]),cmap='copper_r')
    plt.xlabel(r'$R_{21}$')
    plt.ylabel(r'$R_{32}$')

    plt.subplot(326)
    plt.hexbin(np.tan(sampler.flatchain[:,1]),sampler.flatchain[:,2],cmap='copper_r')
    plt.xlabel(r'$R_{21}$')
    plt.ylabel(r'Offset')
    plt.tight_layout()
    plt.savefig(figdir+name+suffix+'.pdf',format='pdf',
              orientation='portrait')
    p.close()
    p.clf()


def sampler_plot2d_mixture(sampler,data,figdir = '../plots/',suffix='',name=None,badprob=None,nLines = 10):
    x = data['x']
    y = data['y']
    x_err = data['x_err']
    y_err = data['y_err']
    plt.copper()

    plt.figure(figsize=(6,6))
    plt.subplot(111)
    plt.scatter(x,y,marker='o',c=badprob,edgecolors='none',zorder=100,cmap='copper_r')
    plt.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker='o',mew=0,ecolor='black',c=badprob,alpha=0.3)
    plt.xlabel('CO(lower)')
    plt.ylabel('CO(upper)')

    testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
    if sampler.flatchain.shape[1] == 7:
        xoff = np.median(sampler.flatchain[:,1])
        UseXoff = True
    else:
        xoff = 0
        UseXoff = False
    plt.plot(testx,np.tan(np.median(sampler.flatchain[:,0]))*(testx+xoff),color='r')
#    p.tight_layout()

    for line in np.arange(nLines):
        index = int(np.random.rand(1)*sampler.flatchain.shape[0])
        if UseXoff:
            xoff = sampler.flatchain[index,1]
        else:
            xoff=0
        plt.plot(testx,np.tan(sampler.flatchain[index,0])*(testx+xoff),
            alpha=0.3,color='gray')


    plt.savefig(figdir+name+suffix+'.pdf',format='pdf',
              orientation='portrait')
    plt.close()
    plt.clf()
