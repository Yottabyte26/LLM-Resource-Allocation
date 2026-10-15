"""Real conflict-diagnostic figures, separate from all locked part 1 figures."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_conflict_figures(folder,datasets,domains,domain_rows,pairs,thresholds,settings):
    """Draw all requested diagnostics, with plotting subsamples clearly labeled."""
    folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    files=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(folder/name,dpi=settings["figure_dpi"],bbox_inches="tight",facecolor="white");plt.close(fig);files.append(name)
    a1=datasets["A1"]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.hist(a1["K"],bins=60,color="#4777a8",alpha=.8)
    for name,color in [("P50","#666666"),("P90","#c07842"),("P95","#9b3b36")]:ax.axvline(thresholds[name],color=color,linestyle="--",label=f"A1 {name}={thresholds[name]:.4f}")
    ax.set(xlabel="Weighted disagreement K",ylabel="Record count",title="A1 within-record conflict distribution");ax.legend();save(fig,"01_A1_K_distribution.png")
    order=[r["domain"] for r in domain_rows]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.boxplot([a1["K"][domains==d] for d in order],tick_labels=order,showfliers=False,patch_artist=True,boxprops={"facecolor":"#d7e3ef"})
    ax.axhline(thresholds["P90"],linestyle="--",color="#c07842",label="A1 global P90")
    ax.set(ylabel="K",title="A1 domains: K (outliers hidden)");ax.tick_params(axis="x",rotation=20);ax.legend();save(fig,"02_domain_K_boxplots.png")
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(order))
    rate=np.array([r["high_conflict_rate"] for r in domain_rows])
    low=np.array([r["high_rate_wilson95_low"] for r in domain_rows]);high=np.array([r["high_rate_wilson95_high"] for r in domain_rows])
    ax.bar(x,100*rate,color="#4777a8");ax.errorbar(x,100*rate,yerr=100*np.array([rate-low,high-rate]),fmt="none",color="#333333",capsize=3)
    ax.set(xticks=x,xticklabels=order,ylabel="K > A1 global P90 (%)",title="A1 high-conflict rates (95% Wilson intervals)")
    ax.tick_params(axis="x",rotation=20);save(fig,"03_domain_high_conflict_rates.png")
    names=a1["names"];matrix=np.zeros((22,22))
    for r in pairs:
        j,k=names.index(r["metric_1"]),names.index(r["metric_2"]);matrix[j,k]=matrix[k,j]=r["severe_pair_conflict_rate"]
    fig,ax=plt.subplots(figsize=(14,12));im=ax.imshow(matrix,vmin=0,vmax=1,cmap="YlOrRd")
    ax.set(xticks=range(22),yticks=range(22),xticklabels=names,yticklabels=names,title="A1 pair disagreement: P(|z_j-z_k| >= 0.5)")
    ax.tick_params(axis="x",labelrotation=90,labelsize=8);ax.tick_params(axis="y",labelsize=8);fig.colorbar(im,ax=ax,shrink=.7,label="Severe pair rate");save(fig,"04_pair_conflict_heatmap.png")
    ranked=sorted(pairs,key=lambda r:(-r["severe_pair_conflict_rate"],-r["mean_abs_difference"]))[:20][::-1]
    fig,ax=plt.subplots(figsize=(14,9));ax.barh([f"{r['metric_1']} vs {r['metric_2']}" for r in ranked],[100*r["severe_pair_conflict_rate"] for r in ranked],color="#4777a8")
    ax.set(xlabel="Severe pair conflict rate (%)",title="A1 top 20 indicator pairs (rate, then mean difference)");save(fig,"05_top20_pair_conflicts.png")
    rng=np.random.default_rng(settings["random_state"]);index=np.sort(rng.choice(len(a1["Q"]),min(settings["scatter_max_points"],len(a1["Q"])),replace=False))
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(a1["Q"][index],a1["K"][index],s=4,alpha=.2,color="#4777a8")
    grid=np.linspace(0,1,300);ax.plot(grid,2*grid*(1-grid),color="#666666",linestyle="--",label="Mathematical bound K <= 2Q(1-Q)")
    ax.axhline(thresholds["P90"],color="#c07842",linestyle="--",label="A1 P90");ax.set(xlabel="Locked Q",ylabel="K",title=f"A1 Q vs K ({len(index):,} points shown)",xlim=(0,1),ylim=(0,.52));ax.legend(fontsize=8);save(fig,"06_Q_vs_K.png")
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(a1["K"][index],a1["Delta_Q"][index],s=4,alpha=.2,color="#4777a8")
    ax.axvline(thresholds["P90"],color="#c07842",linestyle="--",label="A1 P90");ax.set(xlabel="K",ylabel="|Q_CRITIC - Q_equal|",title=f"A1 disagreement vs weighting sensitivity ({len(index):,} shown)");ax.legend();save(fig,"07_K_vs_weighting_sensitivity.png")
    for dataset,domain,number in [("A2","arxiv",8),("A3","github",9)]:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        groups=[(a1["K"][domains==domain],f"A1-{domain}","#4777a8"),(datasets[dataset]["K"],f"{dataset}-full","#c07842"),(datasets[dataset]["K"][datasets[dataset]["nonoverlap_mask"]],f"{dataset}-nonoverlap","#41927d")]
        for values,label,color in groups:
            axes[0].hist(values,bins=np.linspace(0,.5,61),density=True,histtype="step",label=f"{label} (n={len(values):,})",color=color,linewidth=1.3)
            ordered=np.sort(values);axes[1].plot(ordered,np.arange(1,len(values)+1)/len(values),label=label,color=color,linewidth=1.3)
        axes[0].set(xlabel="K",ylabel="Density",title=f"{domain}: sample / full / nonoverlap")
        axes[1].set(xlabel="K",ylabel="Cumulative probability",title="Empirical CDF")
        for ax in axes:ax.axvline(thresholds["P90"],color="#777777",linestyle="--",linewidth=.8);ax.legend(fontsize=8)
        save(fig,f"0{number}_{domain}_sample_extended_K.png")
    return files
