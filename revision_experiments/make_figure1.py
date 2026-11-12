import json,sys,os
RES=sys.argv[1] if len(sys.argv)>1 else 'results'   # folder holding exp1_*_results.json
OUT=sys.argv[2] if len(sys.argv)>2 else 'Figure1.png'
def load(name,lang):
    p=os.path.join(RES,f'exp1_{name}_{lang}_results.json')
    if not os.path.exists(p): return None,None
    m=json.load(open(p))['metrics']; return m['Hit@5'],m['MRR']
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
names=['BM25','BM25+Norm','LaBSE','mE5-base','mE5-large','BGE-M3']
keys=['bm25','bm25norm','labse','me5','me5large','bgem3']
hit={'ar':[],'ms':[]};mrr={'ar':[],'ms':[]}
for k in keys:
    for l in ('ar','ms'):
        h,m=load(k,l); hit[l].append(None if h is None else round(h,3)); mrr[l].append(None if m is None else round(m,3))
fig,axs=plt.subplots(1,2,figsize=(11.5,3.69),dpi=200)
w=.36;x=np.arange(6)
for ax,data,title,yl,ymax,step in [(axs[0],hit,'(a) Hit@5 (≥1 relevant in top 5)','Hit@5',.16,.02),(axs[1],mrr,'(b) MRR','MRR',.11,.02)]:
    ax.set_axisbelow(True);ax.yaxis.grid(True,color='#e6e6e6',lw=.8)
    for k,(lang,col,ec,off) in enumerate([('ar','#21708a','#21708a',-w/2),('ms','#a8d5e2','#5aa5bb',w/2)]):
        for i,v in enumerate(data[lang]):
            if v is None:
                ax.text(x[i]+off+.02,0.0015,'N/A',color='gray',fontsize=9,ha='left' if lang=='ms' else 'right');continue
            ax.bar(x[i]+off,v,w,color=col,edgecolor=ec,lw=1.2,label=('Arabic' if lang=='ar' else 'Malay') if i==0 or (i==1 and lang=='ms' and False) else None)
    # labels with collision handling
    for i in range(6):
        a,m=data['ar'][i],data['ms'][i]
        for lang,v,off in (('ar',a,-w/2),('ms',m,w/2)):
            if v is None: continue
            other=m if lang=='ar' else a
            dx=0;dy=ymax*.012
            if other is not None and abs(v-other)<ymax*.04:   # close neighbours: spread horizontally and stagger
                dx=-.10 if lang=='ar' else .10
                if v<=other: dy=ymax*.012
                else: dy=ymax*.012
            ax.text(x[i]+off+dx,v+dy,f'{v:.3f}',ha='center',va='bottom',fontsize=8.5)
    ax.set_xticks(x);ax.set_xticklabels(names,rotation=15,ha='right' if False else 'center')
    ax.set_ylim(0,ymax);ax.set_yticks(np.arange(0,ymax+1e-9,step));ax.set_ylabel(yl)
    ax.set_title(title,fontweight='bold',fontsize=12)
    for s in ('top','right'): pass
    h,l=ax.get_legend_handles_labels();seen={}
    for hh,ll in zip(h,l): seen.setdefault(ll,hh)
    ax.legend(seen.values(),seen.keys(),loc='upper left',fontsize=10)
plt.tight_layout();plt.savefig(OUT,dpi=200)
