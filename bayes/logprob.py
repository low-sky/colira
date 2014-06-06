import numpy as np
import scipy.stats as ss
import pdb

def lptest(x,y,z,x_err,y_err,z_err):
    theta,phi,scatter = p[0],p[1],p[2]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(phi-np.pi/4)>np.pi/4:
        return -np.inf
    if scatter<0.0:
        return -np.inf
    Gamma = x*np.sin(theta)*np.cos(phi)+\
        y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
    DeltaX2 = (x-Gamma*np.sin(theta)*np.cos(phi))**2
    DeltaY2 = (y-Gamma*np.sin(theta)*np.sin(phi))**2
    DeltaZ2 = (z-Gamma*np.cos(theta))**2
    Delta2 = DeltaX2+DeltaY2+DeltaZ2
    Sigma = DeltaX2/Delta2*x_err**2+\
        DeltaY2/Delta2*y_err**2+\
        DeltaZ2/Delta2*z_err**2+scatter
    lp = -0.5*np.nansum(Delta2/Sigma)-0.5*np.nansum(np.log(Sigma))

    return lp


def logprob3d(p,x,y,z,x_err,y_err,z_err):
    theta,phi,sigx,sigy,sigz = p[0],p[1],p[2],p[3],p[4]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(phi-np.pi/4)>np.pi/4:
        return -np.inf

    # Distance between ray at theta, phi and a point x,y,z
    #Gamma is the dot product of the data vector along the theoretical lin
    Gamma = x*np.sin(theta)*np.cos(phi)+\
        y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
    DeltaX2 = (x-Gamma*np.sin(theta)*np.cos(phi))**2
    DeltaY2 = (y-Gamma*np.sin(theta)*np.sin(phi))**2
    DeltaZ2 = (z-Gamma*np.cos(theta))**2
    Delta2 = DeltaX2+DeltaY2+DeltaZ2
    Sigma = DeltaX2/Delta2*(x_err**2+sigx**2)+\
        DeltaY2/Delta2*(y_err**2+sigy**2)+\
        DeltaZ2/Delta2*(z_err**2+sigz**2)
    lp = -0.5*np.nansum(Delta2/Sigma)+\
        np.nansum(ss.invgamma.logpdf((sigx/x_err)**2,1)+\
                      ss.invgamma.logpdf((sigy/y_err)**2,1)+\
                      ss.invgamma.logpdf((sigz/z_err)**2,1))

#-0.5*np.nansum(np.log(\
#            (x_err**2+sigx**2)+\
#                (y_err**2+sigy**2)+\
#                (z_err**2+sigz**2)))
    return lp

def logprob3d_xoff(p,x,y,z,x_err,y_err,z_err):
    theta,phi,xoffset = p[0],p[1],p[2]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(phi-np.pi/4)>np.pi/4:
        return -np.inf
    Gamma = (x+xoffset)*np.sin(theta)*np.cos(phi)+\
        y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
    DeltaX2 = (x+xoffset-Gamma*np.sin(theta)*np.cos(phi))**2
    DeltaY2 = (y-Gamma*np.sin(theta)*np.sin(phi))**2
    DeltaZ2 = (z-Gamma*np.cos(theta))**2
    Delta2 = DeltaX2+DeltaY2+DeltaZ2
    Sigma = DeltaX2/Delta2*x_err**2+\
        DeltaY2/Delta2*y_err**2+\
        DeltaZ2/Delta2*z_err**2
    lp = -0.5*np.nansum(Delta2/Sigma)\
        +0.5*np.nansum((xoffset/x_err)*(xoffset<0))
    return lp

def logprob3d_xoff_scatter(p,x,y,z,x_err,y_err,z_err):
    theta,phi,xoffset,sigx,sigy,sigz = p[0],p[1],p[2],p[3],p[4],p[5]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(phi-np.pi/4)>np.pi/4:
        return -np.inf
    Gamma = (x+xoffset)*np.sin(theta)*np.cos(phi)+\
        y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
    DeltaX2 = (x+xoffset-Gamma*np.sin(theta)*np.cos(phi))**2
    DeltaY2 = (y-Gamma*np.sin(theta)*np.sin(phi))**2
    DeltaZ2 = (z-Gamma*np.cos(theta))**2
    Delta2 = DeltaX2+DeltaY2+DeltaZ2
    Sigma = DeltaX2/Delta2*(x_err**2+sigx**2)+\
        DeltaY2/Delta2*(y_err**2+sigy**2)+\
        DeltaZ2/Delta2*(z_err**2+sigz**2)
    lp = -0.5*np.nansum(Delta2/Sigma)-\
        0.5*np.nansum(np.log(Sigma))\
        +0.5*np.nansum((xoffset/x_err)*(xoffset<0))
    return lp

def logprob2d(p,x,y,x_err,y_err):
    theta = p[0]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    Delta = (np.cos(theta)*y - np.sin(theta)*x)**2
    Sigma = (np.sin(theta))**2*x_err**2+(np.cos(theta))**2*y_err**2
    lp = -0.5*np.nansum(Delta/Sigma)
#\
#        -0.5*np.nansum(np.log(x_err**2+y_err**2+2*scatter))
    return lp

def logprob2d_posdef(p,x,y,x_err,y_err):
    theta = p[0]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    Delta = (np.cos(theta)*y - np.sin(theta)*x)**2
    ind = np.where(((x*np.cos(theta)**2+y*np.cos(theta)*np.sin(theta))<0) +\
                       ((x*np.cos(theta)*np.sin(theta)+y*np.sin(theta)**2)<0))
    Sigma = (np.sin(theta))**2*x_err**2+(np.cos(theta))**2*y_err**2
    Delta[ind]=(x**2+y**2)
    Sigma[ind]=(x_err**2+y_err**2)
    lp = -0.5*np.nansum(Delta/Sigma)\
        -0.5*np.nansum(Sigma)
    return lp

def logprob2d_xoff(p,x,y,x_err,y_err):
    theta,xoff = p[0],p[1]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    Delta = (np.cos(theta)*y - np.sin(theta)*(x+xoff))**2
    Sigma = (np.sin(theta))**2*x_err**2+(np.cos(theta))**2*y_err**2
    lp = -0.5*np.nansum(Delta/Sigma)+0.5*np.nansum(xoff/x_err*(xoff<0))
    return lp

def logprob2d_xoff_scatter(p,x,y,x_err,y_err):
    theta,scatterx,scattery,xoff = p[0],p[1],p[2],p[3]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    Delta = (np.cos(theta)*y - np.sin(theta)*(x+xoff))**2
    Sigma = (np.sin(theta))**2*(x_err**2+scatterx**2)+\
        (np.cos(theta))**2*(y_err**2+scattery**2)
    lp = -0.5*np.nansum(Delta/Sigma)+\
        np.nansum(ss.invgamma.logpdf(scatterx**2/(x_err**2),1))+\
        np.nansum(ss.invgamma.logpdf(scattery**2/(y_err**2),1))
    return lp

def logprob2d_scatter(p,x,y,x_err,y_err):
    theta,scatterx,scattery = p[0],p[1],p[2]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    Delta = (np.cos(theta)*y - np.sin(theta)*(x))**2
    Sigma = (np.sin(theta))**2*(x_err**2+scatterx**2)+\
        (np.cos(theta))**2*(y_err**2+scattery**2)
    lp = -0.5*np.nansum(Delta/Sigma)+\
        np.nansum(ss.invgamma.logpdf(scatterx**2/(x_err**2),1))+\
        np.nansum(ss.invgamma.logpdf(scattery**2/(y_err**2),1))

    return lp


def logprob2d_scatter_mixture(p,x,y,x_err,y_err):
    theta,scatter,badfrac,xbad,ybad,badsig= p[0],p[1],p[2],\
      p[3],p[4],p[5]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(badfrac-0.5)>0.5:
        return -np.inf
    xscale = np.nanmax(x)
    yscale = np.nanmax(y)
    xoff = 0
    Delta = (np.cos(theta)*y - np.sin(theta)*(x+xoff))**2
    Sigma = (np.sin(theta))**2*(x_err**2+scatter**2)+\
        (np.cos(theta))**2*(y_err**2+scatter**2)
    goodlp = -0.5*(Delta/Sigma)
    BadDelta = (y-ybad)**2+(x-xbad)**2
    badlp =-0.5*(BadDelta/(Sigma+badsig**2))
    lp = np.nansum(np.log(np.exp(goodlp)*(1-badfrac)+np.exp(badlp)*badfrac))\
        +ss.norm.logpdf(badfrac/0.001)+\
        np.sum(ss.invgamma.logpdf(scatter**2/(x_err**2+y_err**2),1))+\
        np.sum(ss.invgamma.logpdf(badsig**2/(xscale**2+yscale**2),1))+\
        np.sum(ss.norm.logpdf(xbad/xscale,1))+\
        np.sum(ss.norm.logpdf(ybad/yscale,1))+\
        ss.beta.logpdf(2*theta/np.pi,2,4)
# factor of 10 to make badsig really big.
    if np.isnan(lp):
        pdb.set_trace()
    return lp

def logprob2d_xoff_scatter_mixture(p,x,y,x_err,y_err):
    theta,xoff,scatter,badfrac,xbad,ybad,badsig= p[0],p[1],p[2],\
      p[3],p[4],p[5],p[6]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(badfrac-0.5)>0.5:
        return -np.inf
    datascale = np.percentile(x,90)
    Delta = (np.cos(theta)*y - np.sin(theta)*(x+xoff))**2
    Sigma = (np.sin(theta))**2*(x_err**2+scatter**2)+\
        (np.cos(theta))**2*(y_err**2+scatter**2)
    goodlp = -0.5*(Delta/Sigma)
    BadDelta = (y-ybad)**2+(x-xbad)**2
    badlp =-0.5*(BadDelta/(Sigma+badsig**2))

    lp = np.nansum(np.log(np.exp(goodlp)*(1-badfrac)+np.exp(badlp)*badfrac))\
        +ss.norm.logpdf(badfrac/0.005)+\
        np.sum(ss.invgamma.logpdf(scatter**2/(x_err**2+y_err**2),1))+\
        np.sum(ss.invgamma.logpdf(badsig**2/(x_err**2+y_err**2)/100,1))+\
        ss.beta.logpdf(2*theta/np.pi,2,4)
    if np.isnan(lp):
        pdb.set_trace()
    return lp

def logprob3d_xoff_scatter_mixture(p,x,y,z,x_err,y_err,z_err):
    theta,phi,xoff,scatter,badfrac,badsig,badmn = p[0],p[1],p[2],p[3],p[4],p[5],p[6]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(phi-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(badfrac-0.5) > 0.5:
        return -np.inf
    
    datascale = np.percentile(y,90)
       
    # Distance between ray at theta, phi and a point x,y,z
    #Gamma is the dot product of the data vector along the theoretical lin
    Gamma = (x+xoff)*np.sin(theta)*np.cos(phi)+\
        y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
    DeltaX2 = ((x+xoff)-Gamma*np.sin(theta)*np.cos(phi))**2
    DeltaY2 = (y-Gamma*np.sin(theta)*np.sin(phi))**2
    DeltaZ2 = (z-Gamma*np.cos(theta))**2
    Delta2 = DeltaX2+DeltaY2+DeltaZ2
    Sigma2 = DeltaX2/Delta2*(x_err**2+scatter**2)+\
        DeltaY2/Delta2*(y_err**2+scatter**2)+\
        DeltaZ2/Delta2*(z_err**2+scatter**2)
    goodlp = -0.5*(Delta2/Sigma2)
    BadDelta = (x-badmn*np.cos(phi)*np.sin(theta))**2+\
        (y-badmn*np.sin(phi)*np.sin(theta))**2+\
        (z-badmn*np.cos(theta))**2
    badlp =-0.5*(BadDelta/(Sigma2+badsig**2))
    lp = np.sum(np.log(np.exp(goodlp)*(1-badfrac)+np.exp(badlp)*badfrac))+\
        ss.norm.logpdf(badfrac/0.005)+\
        np.sum(ss.invgamma.logpdf(scatter**2/(x_err**2+y_err**2+z_err**2),1))+\
        np.sum(ss.invgamma.logpdf(badsig**2/(x_err**2+y_err**2+z_err**2)/100,1))+\
        ss.beta.logpdf(2*theta/np.pi,20,40)+\
        ss.beta.logpdf(2*phi/np.pi,20,40)


    # factor of 10 to make badsig really big.
    if np.isnan(lp):
        pdb.set_trace()
    return lp

def logprob3d_scatter_mixture(p,x,y,z,x_err,y_err,z_err):
    theta,phi,scatter,badfrac,badsig,badmn = p[0],p[1],p[2],p[3],p[4],p[5]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(phi-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(badfrac-0.5) > 0.5:
        return -np.inf
    xoff = 0.0
    datascale = np.percentile(y,90)
       
        # Distance between ray at theta, phi and a point x,y,z
    #Gamma is the dot product of the data vector along the theoretical lin
    Gamma = (x+xoff)*np.sin(theta)*np.cos(phi)+\
        y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
    DeltaX2 = ((x+xoff)-Gamma*np.sin(theta)*np.cos(phi))**2
    DeltaY2 = (y-Gamma*np.sin(theta)*np.sin(phi))**2
    DeltaZ2 = (z-Gamma*np.cos(theta))**2
    Delta2 = DeltaX2+DeltaY2+DeltaZ2
    Sigma2 = DeltaX2/Delta2*(x_err**2+scatter**2)+\
        DeltaY2/Delta2*(y_err**2+scatter**2)+\
        DeltaZ2/Delta2*(z_err**2+scatter**2)
    goodlp = -0.5*(Delta2/Sigma2)
    BadDelta = (x-badmn*np.cos(phi)*np.sin(theta))**2+\
        (y-badmn*np.sin(phi)*np.sin(theta))**2+\
        (z-badmn*np.cos(theta))**2
    badlp =-0.5*(BadDelta/(Sigma2+badsig**2))
    lp = np.sum(np.log(np.exp(goodlp)*(1-badfrac)+np.exp(badlp)*badfrac))+\
        ss.norm.logpdf(badfrac/0.001)+\
        np.sum(ss.invgamma.logpdf(scatter**2/(x_err**2+y_err**2+z_err**2),1))+\
        np.sum(ss.invgamma.logpdf(badsig**2/(x_err**2+y_err**2+z_err**2)/100,1))+\
        np.sum(ss.norm.logpdf(badmn/(2*datascale),1,3))+\
        ss.beta.logpdf(2*theta/np.pi,20,40)+\
        ss.beta.logpdf(2*phi/np.pi,20,40)
# factor of 10 to make badsig really big.
    if np.isnan(lp):
        pdb.set_trace()
    return lp


def logprob2d_kelly(p,x,y,x_err,y_err):
    sigy,sigx,rho,mux = p[0],p[1],p[2],p[3]
#    mux=0
    muy=rho*mux*sigy/sigx
    logprob = 0
    if sigy<0:
        return(-np.inf)
    if sigx<0:
        return(-np.inf)
    if rho<0:
        return(-np.inf)
    if rho>1: 
        return(-np.inf)
    if sigy>sigx:
        return(-np.inf)

    covmodel = np.matrix([[sigx**2,rho*sigx*sigy],[rho*sigx*sigy,sigy**2]])    
    meanmodel = np.matrix([[mux],[muy]])
    for i in np.arange(len(x)):
        covdata = np.matrix([[x_err[i]**2,0],[0,y_err[i]**2]])
        meandata = np.matrix([[x[i]],[y[i]]])
        try:
            norm = -0.5*np.log(np.linalg.det(covdata+covmodel))
            logexp = -0.5*(meandata-meanmodel).getT()*np.linalg.inv(covdata+covmodel)*(meandata-meanmodel)
            covmix = np.linalg.det(np.linalg.inv((covdata.getI()+covmodel.getI())))
            logsigsq = np.log(covmix)
            logprob = logprob+logexp+norm+logsigsq


#            meanmix = covmix*(covdata.getI())*meandata+covmix*(covmodel.getI())*meanmodel
#            expon = ((meandata-meanmix).getT())*covmix.getI()*(meandata-meanmix)
#            logprob = logprob-float(expon)-0.5*np.log(np.linalg.det(covdata+covmodel))
            if np.isnan(logprob):
                return -np.inf
        except (ValueError):
            return -np.inf
    return logprob
