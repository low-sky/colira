import matplotlib.pyplot as p
import numpy as np
def sampler_plot(sampler,data,figdir = '../plots/',suffix='',name=None):
    x = data['x']
    y = data['y']
    z = data['z']
    x_err = data['x_err']
    y_err = data['y_err']
    z_err = data['z_err']

    p.figure(figsize=(6,8))
    p.subplot(321)
    p.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]),\
               range=[0,2],bins=100)
    p.xlabel(r'$R_{32}$')
    p.subplot(322)
    p.hist(np.tan(sampler.flatchain[:,1]),range=[0,2],bins=100)
    p.xlabel(r'$R_{21}$')

    p.subplot(323)
    p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
    p.scatter(x,y,marker='.')
    p.xlabel('CO(1-0)')
    p.ylabel('CO(2-1)')
    testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
    xoff = 0 #np.median(sampler.flatchain[:,2])
    p.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*(testx+xoff),color='r')

    p.subplot(324)
    p.errorbar(y,z,xerr=y_err,yerr=y_err,fmt=None,marker=None,mew=0)
    p.scatter(y,z,marker='.')
    p.xlabel('CO(2-1)')
    p.ylabel('CO(3-2)')
    testx = np.linspace(np.nanmin(y),np.nanmax(y),10)
    p.plot(testx,testx/np.tan(np.median(sampler.flatchain[:,0]))/\
               np.sin(np.median(sampler.flatchain[:,1])),color='r')
    
    p.subplot(325)
    p.hexbin(np.tan(sampler.flatchain[:,1]),
             1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]))
    p.xlabel(r'$R_{21}$')
    p.ylabel(r'$R_{32}$')

    p.subplot(326)
    p.hexbin(np.tan(sampler.flatchain[:,1]),sampler.flatchain[:,2])
    p.xlabel(r'$R_{21}$')
    p.ylabel(r'Offset')
    p.tight_layout()
    p.savefig(figdir+name+suffix+'.pdf',format='pdf',
              orientation='portrait')
    p.close()
    p.clf()

def sampler_plot2d(sampler,figdir = '../plots/',suffix='',
                   xoff=None,name=None,nLines=10):
    p.figure(figsize=(6,6))
    p.subplot(221)
    p.hist(np.tan(sampler.flatchain[:,0]),range=[0,2],bins=100)
    p.xlabel(r'$R_{21}$')

    p.subplot(222)
    p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
    p.scatter(x,y,marker='.')
    p.xlabel('CO (lower)')
    p.ylabel('CO (upper)')
    testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
    UseXoff = sampler.flatchain.shape[1]>6

    if xoff==None:
        if UseXoff:
            xoff = np.median(sampler.flatchain[:,3])
        else:
            xoff = 0

    sshape = sampler.flatchain.shape
    for line in np.arange(nLines):
        index = int(np.random.rand(1)*sshape[0])
        if UseXoff:
            xoff = sampler.flatchain[index,3]
        else:
            xoff=0
        p.plot(testx,np.tan(sampler.flatchain[index,0])*(testx+xoff),
            alpha=0.3,color='gray')

    p.plot(testx,np.tan(np.median(sampler.flatchain[:,0]))*
           (testx+xoff),color='r')
    p.ylim((np.nanmin(y),np.nanmax(y)))
    p.xlim((np.nanmin(x),np.nanmax(x)))
    
    sigma = (sampler.flatchain[:,1]**2+sampler.flatchain[:,2]**2)**0.5
    p.subplot(223)
    p.hexbin(np.tan(sampler.flatchain[:,0]),sigma)
    p.xlabel(r'$R$')
    p.ylabel('Scatter')

    p.subplot(224)
    if UseXoff:
        p.hist(sampler.flatchain[:,4])
    else:
        p.hist(sampler.flatchain[:,3])
    p.xlabel(r'$f_{bad}$')

    p.savefig(figdir+name+suffix+'.pdf',format='pdf',
        orientation='portrait')

    p.tight_layout()        
    p.close()
    p.clf()

def sampler_plot_mixture(sampler,data,figdir = '../plots/',suffix='',name=None,badprob=None):
    x = data['x']
    y = data['y']
    z = data['z']
    x_err = data['x_err']
    y_err = data['y_err']
    z_err = data['z_err']
    plt.copper()

    p.figure(figsize=(6,8))
    p.subplot(321)
    p.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]),\
               range=[0,2],bins=100)
    p.xlabel(r'$R_{32}$')
    p.subplot(322)
    p.hist(np.tan(sampler.flatchain[:,1]),range=[0,2],bins=100)
    p.xlabel(r'$R_{21}$')

    p.subplot(323)
    p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0,color=badprob,cmap='copper')
    p.scatter(x,y,marker='.',color=badprob,cmap='copper')
    p.xlabel('CO(1-0)')
    p.ylabel('CO(2-1)')
    testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
    xoff = np.median(sampler.flatchain[:,2])
    p.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*(testx+xoff),color='r')

    p.subplot(324)
    p.errorbar(y,z,xerr=y_err,yerr=y_err,fmt=None,marker=None,mew=0,color=badprob,cmap='copper')
    p.scatter(y,z,marker='.',color=badprob,cmap='copper')
    p.xlabel('CO(2-1)')
    p.ylabel('CO(3-2)')
    testx = np.linspace(np.nanmin(y),np.nanmax(y),10)
    p.plot(testx,testx/np.tan(np.median(sampler.flatchain[:,0]))/\
               np.sin(np.median(sampler.flatchain[:,1])),color='r')
    
    p.subplot(325)
    p.hexbin(np.tan(sampler.flatchain[:,1]),
             1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]))
    p.xlabel(r'$R_{21}$')
    p.ylabel(r'$R_{32}$')

    p.subplot(326)
    p.hexbin(np.tan(sampler.flatchain[:,1]),sampler.flatchain[:,2])
    p.xlabel(r'$R_{21}$')
    p.ylabel(r'Offset')
    p.tight_layout()
    p.savefig(figdir+name+suffix+'.pdf',format='pdf',
              orientation='portrait')
    p.close()
    p.clf()
