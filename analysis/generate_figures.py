from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'figure_data'
OUT = ROOT / 'results' / 'figures'
PANELS = OUT / 'panels'
OUT.mkdir(parents=True, exist_ok=True)
PANELS.mkdir(parents=True, exist_ok=True)

TEAL   = '#1D9E75'
CORAL  = '#D85A30'
AMBER  = '#BA7517'
GRAY   = '#6B7280'
LGRAY  = '#D1D5DB'
RED    = '#A32D2D'
MAROON = '#A32D2D'

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        800,
    'savefig.dpi':       800,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.6,
    'grid.linewidth':    0.4,
    'grid.alpha':        0.4,
    'lines.linewidth':   1.4,
    'patch.linewidth':   0.6,
})


def save(fig, stem):
    png = PANELS / f'{stem}.png'
    pdf = PANELS / f'{stem}.pdf'
    fig.savefig(png, bbox_inches='tight', facecolor='white', dpi=800)
    fig.savefig(pdf, bbox_inches='tight', facecolor='white', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return png


def combine(panel_paths, rows, cols, out_stem, gap=36):
    ims = [Image.open(p).convert('RGB') for p in panel_paths]
    max_w = max(im.width for im in ims)
    max_h = max(im.height for im in ims)
    canvas = Image.new('RGB', (cols*max_w+(cols-1)*gap, rows*max_h+(rows-1)*gap), 'white')
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        x = c*(max_w+gap)+(max_w-im.width)//2
        y = r*(max_h+gap)+(max_h-im.height)//2
        canvas.paste(im, (x,y))
    canvas.save(OUT / f'{out_stem}.png', dpi=(800,800))
    canvas.save(OUT / f'{out_stem}.pdf', 'PDF', resolution=800, title=False, author=False, subject=False, keywords=False, creator=False, producer=False, creationDate=False, modDate=False)

# ---------------- Figure 2 ----------------
f2a = pd.read_csv(BASE/'fig2a_coverage_menus.csv')
f2b = pd.read_csv(BASE/'fig2b_cad_localization.csv')
f2c = pd.read_csv(BASE/'fig2c_silent_suppression.csv')
f2d = pd.read_csv(BASE/'fig2d_timeliness_contract_scatter.csv')

# 2a: dual menu bars with empirical intervals
fig = plt.figure(figsize=(5.25,3.85))
ax = fig.add_axes([0.13,0.19,0.83,0.70])
x = np.arange(2); w=.34
b = f2a['median_budget_coverage'].to_numpy(); j=f2a['median_joint_coverage'].to_numpy()
blo=f2a['median_budget_coverage_lower_2_5'].to_numpy(); bhi=f2a['median_budget_coverage_upper_97_5'].to_numpy()
jlo=f2a['median_joint_coverage_lower_2_5'].to_numpy(); jhi=f2a['median_joint_coverage_upper_97_5'].to_numpy()
bars1=ax.bar(x-w/2,b,w,color=TEAL,alpha=.82,edgecolor='white',linewidth=.5,
             yerr=np.vstack([b-blo,bhi-b]),capsize=3,label='Budget-only coverage')
bars2=ax.bar(x+w/2,j,w,color=CORAL,alpha=.82,edgecolor='white',linewidth=.5,
             yerr=np.vstack([j-jlo,jhi-j]),capsize=3,label='Joint coverage')
for bars,color,upper in [(bars1,TEAL,bhi),(bars2,CORAL,jhi)]:
    for bar,upper_bound in zip(bars,upper):
        h=bar.get_height(); label_y=max(h,upper_bound)+.025
        ax.text(bar.get_x()+bar.get_width()/2,label_y,f'{h:.3f}',ha='center',va='bottom',fontsize=8,color=color,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['Full fixed\nmenu (19)','Observed positive-budget\nmenu (12)'])
ax.set_ylabel('Matched-burden coverage'); ax.set_ylim(-.05,1.05)
ax.set_title('(a) Budget-only versus joint coverage')
ax.yaxis.grid(True); ax.legend(frameon=False,loc='upper left')
p2a=save(fig,'fig2a')

# 2b
fig=plt.figure(figsize=(5.25,3.85)); ax=fig.add_axes([.13,.19,.83,.70])
vals=f2b['median_CAD'].tolist(); labels=[f"{r['slice']}\n{r['window']}" for _,r in f2b.iterrows()]
cols=[CORAL,CORAL,LGRAY,LGRAY]
bars=ax.bar(range(4),vals,color=cols,alpha=.85,width=.55,edgecolor='white',linewidth=.5)
for bar,val,col in zip(bars,vals,cols):
    ax.text(bar.get_x()+bar.get_width()/2,(val+.008 if val>0 else .008),f'{val:.3f}',ha='center',va='bottom',fontsize=8,color=(CORAL if val>0 else GRAY),fontweight=('bold' if val>0 else 'normal'))
ax.set_xticks(range(4)); ax.set_xticklabels(labels,fontsize=7.5); ax.set_ylabel('CAD'); ax.set_ylim(0,.78)
ax.set_title('(b) CAD localization to attacked slice'); ax.yaxis.grid(True)
p2b=save(fig,'fig2b')

# 2c
fig=plt.figure(figsize=(5.25,3.85)); ax=fig.add_axes([.10,.22,.86,.62])
fracs=f2c['fraction'].tolist(); labels=f2c['category'].tolist(); colors=[LGRAY,TEAL,CORAL]
left=0
for frac,label,color in zip(fracs,labels,colors):
    ax.barh(0,frac,left=left,color=color,alpha=.85,height=.5,edgecolor='white',linewidth=.8)
    ax.text(left+frac/2,0,f'{frac*100:.1f}%',ha='center',va='center',fontsize=8,color='white',fontweight='bold')
    left+=frac
ax.set_xlim(0,1); ax.set_ylim(-.5,.5); ax.set_yticks([]); ax.set_xlabel('Pooled detector-configuration–trial fraction')
ax.set_title('(c) Silent suppression — max-strength Freeze')
patches=[mpatches.Patch(color=c,alpha=.85,label=l) for c,l in zip(colors,labels)]
ax.legend(handles=patches,loc='upper right',frameon=True,framealpha=.9,edgecolor='none',fontsize=7.2,handlelength=1.0,handleheight=.8,borderpad=.5)
p2c=save(fig,'fig2c')

# 2d with dashed rings
fig=plt.figure(figsize=(5.25,3.85)); ax=fig.add_axes([.13,.19,.83,.70])
ax.axhline(0,color=RED,linewidth=.9,linestyle=':',alpha=.6)
ax.text(.01,.012,'MB contract pass = 0',fontsize=7,color=RED,alpha=.75,va='bottom')
colors={'Fused Fisher':TEAL,'Coherence T²':AMBER,'CUSUM Factor LRT':RED,'Max Abs control':GRAY}
markers={'Fused Fisher':'o','Coherence T²':'o','CUSUM Factor LRT':'o','Max Abs control':'s'}
offsets={'Fused Fisher':(-.06,.028),'Coherence T²':(.02,.022),'CUSUM Factor LRT':(.02,-.042),'Max Abs control':(-.08,.022)}
for _,r in f2d.iterrows():
    name=r['display_row']; tk=r['matched_burden_timely_at_k']; cp=r['matched_burden_contract_pass_rate']; cj=r['coverage_joint']; color=colors[name]
    ax.scatter(tk,cp,s=65,color=color,marker=markers[name],zorder=5,edgecolors='white',linewidths=.8)
    if cj>0:
        ax.scatter(tk,cp,s=175,color='none',edgecolors=color,linewidths=1.3,linestyles='dashed',zorder=4,alpha=.65)
    dx,dy=offsets[name]; label='Max Abs (control)' if name=='Max Abs control' else name
    ax.text(tk+dx,cp+dy,label,fontsize=7.5,color=color,ha='center',va='bottom')
ax.set_xlim(-.05,1.05); ax.set_ylim(-.07,.52)
ax.set_xlabel('Matched-burden Timely@K'); ax.set_ylabel('Matched-burden contract-pass rate')
ax.set_title('(d) Timeliness versus contract success')
ax.xaxis.grid(True,alpha=.3); ax.yaxis.grid(True,alpha=.3)
ring=Line2D([0],[0],marker='o',color='none',markeredgecolor=GRAY,markeredgewidth=1.2,markersize=8,linestyle='dashed',label='Nonzero joint coverage')
ax.legend(handles=[ring],loc='upper left',frameon=False,fontsize=7.5)
p2d=save(fig,'fig2d')
combine([p2a,p2b,p2c,p2d],2,2,'fig2_main_cfpb')

# ---------------- Figure 3 ----------------
f3ab=pd.read_csv(BASE/'fig3ab_matched_burden_timeliness_surfaces.csv')
f3c=pd.read_csv(BASE/'fig3c_recalibration_summary.csv')

paths=[]
for fam,stem,xlabel,ylim in [('Freeze','fig3a','Freeze intensity (η)',(.0,.88)),('CorrMix','fig3b','CorrMix intensity (η)',(.0,.72))]:
    fig=plt.figure(figsize=(3.85,3.45)); ax=fig.add_axes([.17,.19,.79,.70])
    for label,state,color,ls,mk in [('Recal ON','Recalibration ON',TEAL,'-','o'),('Recal OFF','Recalibration OFF',CORAL,'--','s')]:
        s=f3ab[(f3ab['incident_family']==fam)&(f3ab['recalibration']==state)].sort_values('intensity')
        ax.plot(s['intensity'],s['median_MB_Timely_at_K'],color=color,linestyle=ls,marker=mk,markersize=4,label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel('Matched-burden Timely@K (median)')
    ax.set_title(f"({'a' if fam=='Freeze' else 'b'}) {fam} timeliness at matched burden")
    ax.set_ylim(*ylim); ax.legend(frameon=False); ax.yaxis.grid(True,alpha=.4)
    paths.append(save(fig,stem))

# 3c: recalibration summary with empirical intervals
fig=plt.figure(figsize=(4.25,3.45)); ax=fig.add_axes([.14,.22,.83,.66])
metrics=['Median budget coverage','Median all-trial contract pass','Observed nonzero-joint fraction']
labels=['Budget-only\ncoverage','All-trial contract\npass','Nonzero-joint\nfraction']
x=np.arange(3); w=.35
for state,offset,color,edge,label in [('Recalibration ON',-w/2,TEAL,'white','Recal ON'),('Recalibration OFF',w/2,LGRAY,GRAY,'Recal OFF')]:
    s=f3c[f3c['state']==state].set_index('metric').loc[metrics]
    vals=s['point'].to_numpy(); lo=s['lower'].to_numpy(); hi=s['upper'].to_numpy()
    bars=ax.bar(x+offset,vals,w,color=color,alpha=.85,edgecolor=edge,linewidth=.6,label=label,yerr=np.vstack([vals-lo,hi-vals]),capsize=3)
    for i,(bar,val) in enumerate(zip(bars,vals)):
        txt='8/19' if (state=='Recalibration ON' and i==2) else ('0/19' if (state=='Recalibration OFF' and i==2) else f'{val:.3f}')
        label_y=max(val,hi[i])+.018
        ax.text(bar.get_x()+bar.get_width()/2,label_y,txt,ha='center',va='bottom',fontsize=7.5,color=(TEAL if state=='Recalibration ON' else (RED if val==0 else GRAY)),fontweight=('bold' if state=='Recalibration ON' or val==0 else 'normal'))
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=7.5); ax.set_ylabel('Rate'); ax.set_ylim(0,.92)
ax.set_title('(c) Recalibration and observed feasibility')
ax.legend(frameon=False,loc='upper right'); ax.yaxis.grid(True,alpha=.4)
paths.append(save(fig,'fig3c'))
combine(paths,1,3,'fig3_threat_surfaces',gap=30)

# ---------------- Figure 4 ----------------
f4a=pd.read_csv(BASE/'fig4a_default_drift_stress.csv')
f4b=pd.read_csv(BASE/'fig4b_drift_placement.csv')
clean=float(f4a.loc[f4a.metric=='Clean pass rate','value'].iloc[0])
drift=float(f4a.loc[f4a.metric=='Drift pass rate','value'].iloc[0])
flip=float(f4a.loc[f4a.metric=='Flip 1→0 rate','value'].iloc[0])
evr=float(f4a.loc[f4a.metric=='Maximum event-burden ratio','value'].iloc[0])
tiw=float(f4a.loc[f4a.metric=='Maximum TIW-burden ratio','value'].iloc[0])

fig=plt.figure(figsize=(5.35,3.4)); ax=fig.add_axes([.10,.16,.67,.74]); ax2=ax.twinx(); W=.45
left=[(0,clean,LGRAY,GRAY,'Clean\npass'),(.55,drift,TEAL,TEAL,'Drift\npass'),(1.10,flip,CORAL,CORAL,'Flip\n₁→₀')]
for xc,val,fc,tc,_ in left:
    ax.bar(xc,val,W,color=fc,alpha=.88,edgecolor=('white' if fc!=LGRAY else GRAY),linewidth=.5)
    ax.text(xc,val+.013,f'{val:.3f}',ha='center',va='bottom',fontsize=8,color=tc,fontweight='bold')
ax.axvline(1.85,color=LGRAY,linewidth=1)
right=[(2.60,evr,AMBER,AMBER,'Max event\nratio'),(3.30,tiw,MAROON,MAROON,'Max TIW\nratio')]
for xc,val,fc,tc,_ in right:
    ax2.bar(xc,val,W,color=fc,alpha=.85,edgecolor='white',linewidth=.5)
    ax2.text(xc,val+.12,f'{val:.3f}×',ha='center',va='bottom',fontsize=8,color=tc,fontweight='bold')
ax2.set_ylim(0,6.5); ax2.set_ylabel('Worst-case burden ratio (×)',color=GRAY,fontsize=8); ax2.tick_params(axis='y',labelcolor=GRAY,labelsize=7.5); ax2.spines['right'].set_visible(True); ax2.spines['right'].set_linewidth(.6)
allb=left+right; ax.set_xticks([x[0] for x in allb]); ax.set_xticklabels([x[-1] for x in allb],fontsize=7.5)
ax.set_xlim(-.45,3.75); ax.set_ylim(0,.72); ax.set_ylabel('Pass / flip rate'); ax.set_title('(a) Default drift stress test (δ=1.0)'); ax.yaxis.grid(True,alpha=.35)
p4a=save(fig,'fig4a')

# Nonstacked placement panel, retaining palette and annotation style
fig=plt.figure(figsize=(5.35,3.4)); ax=fig.add_axes([.12,.16,.80,.74])
x=np.arange(3); w=.34
bp=ax.bar(x-w/2,f4b['pass_rate'],w,color=TEAL,alpha=.75,edgecolor='white',linewidth=.5,label='Drift pass rate')
bf=ax.bar(x+w/2,f4b['flip_1_to_0_rate'],w,color=CORAL,alpha=.90,edgecolor='white',linewidth=.5,label='Flip₁→₀ rate')
for i,(pr,fr,inc) in enumerate(zip(f4b['pass_rate'],f4b['flip_1_to_0_rate'],f4b['max_event_ratio_increase_vs_clean'])):
    ax.text(x[i]-w/2,pr+.012,f'{pr:.3f}',ha='center',va='bottom',fontsize=8,color='black',fontweight='bold')
    ax.text(x[i]+w/2,fr+.012,f'{fr:.3f}',ha='center',va='bottom',fontsize=8,color=CORAL,fontweight='bold')
    ax.text(x[i],max(pr,fr)+.058,f'+{inc:.3f}',ha='center',va='bottom',fontsize=7.5,color=GRAY)
ax.text(1,max(f4b['pass_rate'])+.112,'max event-ratio increase vs clean',ha='right',va='bottom',fontsize=7,color=GRAY,style='italic')
ax.set_xticks(x); ax.set_xticklabels([f'{int(y)}\ndrift start' for y in f4b['drift_start']],fontsize=8); ax.set_xlim(-.6,2.6); ax.set_ylabel('Rate'); ax.set_ylim(0,.75); ax.set_title('(b) Drift placement sensitivity (δ=1.0)'); ax.yaxis.grid(True,alpha=.4); ax.legend(frameon=False,loc='upper right',fontsize=7.5)
p4b=save(fig,'fig4b')
combine([p4a,p4b],1,2,'fig4_drift',gap=36)
shutil.rmtree(PANELS, ignore_errors=True)

print('Generated Figures 2-4')
