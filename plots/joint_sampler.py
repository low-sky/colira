import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def sbscatter(sampler):
    chain = sampler.flatchain
    chain[:,2]=np.abs(chain[:,2])
    chain[:,4]=np.abs(chain[:,4])
    dd = pd.DataFrame(data=chain,
                      columns=['theta','phi','scatter','badfrac','badsig','badmn'])
    g = sns.PairGrid(dd)
    #g.map_upper(plt.scatter)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    plt.savefig('jtplt.pdf')
    
def sbratio(sampler):
    chain = sampler.flatchain
    chain[:,2]=np.abs(chain[:,2])
    chain[:,4]=np.abs(chain[:,4])
    dd = pd.DataFrame(data=chain,
                      columns=['theta','phi','scatter','badfrac','badsig','badmn'])
    with sns.axes_style("white"):
        sns.jointplot("theta", "phi", data, kind="kde");
