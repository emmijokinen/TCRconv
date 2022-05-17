import numpy as np
import csv
import re
import ast
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
#from matplotlib.colors import ListedColormap
from matplotlib import colors as mpcolors
from Bio.Seq import translate
from re import search as re_search


def file2dict(filename,key_fields,store_fields,delimiter='\t'):
    """Read file to a dictionary.
    key_fields: fields to be used as keys
    store_fields: fields to be saved as a list
    delimiter: delimiter used in the given file."""
    dictionary={}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=delimiter)
        for row in reader:
            keys = [row[k] for k in key_fields]
            store= [row[s] for s in store_fields]

            sub_dict = dictionary
            for key in keys[:-1]:
                if key not in sub_dict:
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]
            key = keys[-1]
            if key not in sub_dict:
                sub_dict[key] = []
            sub_dict[key].append(store)
    return dictionary

def get_protseqs_ntseqs(chain='B'):
    """returns sequence dictioaries for genes: protseqsV, protseqsJ, nucseqsV, nucseqsJ"""
    seq_dicts=[]
    for gene,type in zip(['v','j','v','j'],['aa','aa','nt','nt']):
        name = 'data/'+'tr'+chain.lower()+gene+'s_'+type+'.tsv'
        sdict = file2dict(name,key_fields=['Allele'],store_fields=[type+'_seq'])
        for g in sdict:
            sdict[g]=sdict[g][0][0]
        seq_dicts.append(sdict)
    return seq_dicts

def notX(seq,X='N'):
    """ returns index of the first character in seq that is not # XXX:"""
    for i,c in enumerate(seq):
        if c != X:
            return i
    return i

def get_ntv(nt,cdr3,n_aa_cdr3=1,n_nt_cdr3=None):
    p=translate(nt)
    if n_nt_cdr3 is None:
        i2 = min( 3 * ( p.find(cdr3) + n_aa_cdr3 ), len(nt))
    else:
        i2 = min( 3 * ( p.find(cdr3) ) + n_nt_cdr3, len(nt))
    i1 = notX(nt,'N')
    ntv = nt[i1:i2]  # this isn't necessarily in-frame, subseq can still be searched for just fine.
    return ntv

def get_ntj(nt,cdr3,n_aa_cdr3=1,n_nt_cdr3=None):
    p=translate(nt)
    if n_nt_cdr3 is None:
        i1 = max(0, 3*(p.find(cdr3)+len(cdr3)-n_aa_cdr3))
    else:
        i1 = max(0, 3*(p.find(cdr3)+len(cdr3))-n_nt_cdr3)
    ntj = nt[i1:]
    return ntj

def findPossibleGenes(nt,nucseqs):
    """ nt: nucleotide sequence that should correspond to the region determined by the types of genes in nucseqs
        nucseqs: dictionary of genes (V/J) and their corresponding nt-seqs
        returns list of genes that could determine nt.
    """
    genelist=[]
    nt=nt.lower()
    for v in nucseqs:
        if nt in nucseqs[v]:
            genelist.append(v)
    return genelist

def determine_tcr_seq_vj(cdr3,V,J,protseqsV,protseqsJ,guess01=False):
    try:
        V = V.replace('TCR','TR')
        J = J.replace('TCR','TR')
        if guess01:
            if '*' not in V:
                V+='*01'
            if '*' not in J:
                J+='*01'
        pv = protseqsV[V]
        pj = protseqsJ[J]
        t = pv[:pv.rfind('C')]+  cdr3 + pj[re_search(r'[FW]G.G',pj).start()+1:]
        return t
    except:
        return ''

def determine_tcr_seq_nt(nt,cdr3,protseqsV,protseqsJ,nucseqsV,nucseqsJ,guess01=True,nntv=12,nntj=18):
    """Determine as much of the V(D)J sequence as possible when the nucleotide sequence and CDR3
       are given. Requires dictionaries of amino acid and nucleotide sequences corresponding to
       possible V- and J-genes. If guess01 is True, then if the gene can be determined for a
       sequence, but not its allele, the allele is set to be 01.
       Note: This function has only been tested with Adaptive Immunoseq and ImmuneCODE data.
       """

    # Don't accept non-canonical CDR3s or stop codons in the aa-seq
    if len(cdr3)==0 or cdr3[0]!='C' or cdr3[-1] not in 'FW' or '*' in translate(nt):
        return '','',''

    # ADD V
    vlist=[]
    while nntv>0 and len(vlist)==0: # Find V-genes matching the nt-seq
        ntv = get_ntv(nt,cdr3,n_nt_cdr3=nntv)
        vlist=findPossibleGenes(ntv,nucseqsV)
        nntv-=1

    t = ''
    if len(vlist)==0: # Nothing found, use just original
        t=translate(nt)[:-2] # Works with Adaptive data
        t = t[notX(t,'X'):] # Remove possible X:s from beginning
        vb = ''
    else:
        # If there is only one V-gene and allele or
        # if guess01 and there are multiple alleles but only one V-gene,
        # -> select first allele (usually 01)
        if len(vlist)==1 or ( guess01 and len(np.unique([v.split('*')[0] for v in vlist]))==1):
            p = protseqsV[vlist[0]]
            t = p[:p.rfind('C')]+ cdr3
            vb = vlist[0]
        else: # Utilize as much from the end of the aa-seq determined by the V-gene that is the same for all possible V-genes
            ps = []
            for v in vlist:
                p = protseqsV[v]
                p = p[:p.rfind('C')]
                ps.append(p)
            nup = len(np.unique(ps))

            minl = min([len(p) for p in ps])
            ps_ar = np.asarray([list(t[-minl:]) for t in ps])
            l=ps_ar.shape[1]
            for ip in range(l-1,-1,-1):
                if len(set(ps_ar[:,ip]))>1:
                    break
            p = ps[0][ip+1:]
            t = p + cdr3
            vb=''

    # ADD J
    jlist=[]
    while nntj>0 and len(jlist)==0: # Find J-genes matching the nt-seq
        ntj = get_ntj(nt,cdr3,n_nt_cdr3=nntj)
        jlist=findPossibleGenes(ntj,nucseqsJ)
        nntj-=1

    if len(jlist)==0: # Nothing found, use original
        t+=translate(nt)[-2:]
        jb=''
    else:
        # If there is only one J-gene and allele or
        # if guess01 and there are multiple alleles but only one J-gene,
        # -> select first allele (usually 01)
        if len(jlist)==1 or ( guess01 and len(np.unique([j.split('*')[0] for j in jlist]))==1 ):
            p = protseqsJ[jlist[0]]
            t += p[re_search(r'[FW]G.G',p).start()+1:]
            jb = jlist[0]
        else: # Utilize as much from the beginning of the aa-seq determined by the J-gene that is the same for all possible J-genes
            ps = []
            for j in jlist:
                p = protseqsJ[j]
                p = p[re_search(r'[FW]G.G',p).start()+1:]
                ps.append(p)

            minl = min([len(p) for p in ps])
            ps_ar = np.asarray([list(t[-minl:]) for t in ps])
            l=ps_ar.shape[1]
            for ip in range(l-1,-1,-1):
                if len(set(ps_ar[:,ip]))>1:
                    break
            p = ps[0][:ip]
            t += p
            jb=''

    return t, vb, jb

#####

def get_tcr_dict(filename,chain='B', min_tcrs_per_epi=50,skip_mi=True,species='HomoSapiens',min_score=0):
    # For TCRab data download only paired data from vdjdb
    # skip_mi: skip TCRs with missing information

    if chain=='AB':
        tcrs_vdj_all = file2dict(filename,['Species','complex.id','Epitope','Gene'],\
                                 ['CDR3','V','J','Reference','Meta','Score','MHC A','MHC B'])
        # tcr dictionary of paired chains. A & B are combined by complex.id
        td = {}
        for idx in tcrs_vdj_all[species]:
            for epitope in tcrs_vdj_all[species][idx]:
                if epitope not in td:
                    td[epitope]=[]
                td[epitope].append(tcrs_vdj_all[species][idx][epitope]['TRA'][0][0:3]\
                                   +tcrs_vdj_all[species][idx][epitope]['TRB'][0])
    else:
        tcrs_vdj_all = file2dict(filename,['Species','Epitope','Gene'],\
                                 ['CDR3','V','J','Reference','Meta','Score','MHC A','MHC B'])
        td = {}
        for epitope in tcrs_vdj_all[species]:
            if epitope not in td:
                    td[epitope]=[]
            td[epitope]=tcrs_vdj_all[species][epitope]['TR'+chain]

    num_entries = 7 if chain=='AB' else 4
    tcrs_vdj = {}
    num_epis, epis_u = [], []

    for epi in td:
        rows = []
        for row in td[epi]:
            # Check there's no missing info (or not required) and confidence score >= min_score
            if (not skip_mi or (skip_mi and np.all([row[i]!='' for i in range(num_entries)]) ) )\
                and int(row[num_entries+1])>=min_score:
                rows.append(row)

        tcrs = ['+'.join(row[:num_entries-1]) for row in rows]
        tcrs_u,Iu = np.unique(tcrs,return_index=True)

        n_s=len(tcrs_u)
        if n_s >= min_tcrs_per_epi:
            tcrs_vdj[epi]=[]
            for i in Iu:
                row=rows[i][:num_entries]
                meta = ast.literal_eval(rows[i][num_entries])
                row[num_entries-1] += '_'+meta['subject.id'] # reference + subject.id
                tcrs_vdj[epi].append(row)
            num_epis.append(n_s)
            epis_u.append(epi)

    print(len(epis_u), 'epitopes,', np.sum(num_epis), 'TCRs')
    return tcrs_vdj, epis_u, num_epis

def get_unique_tcr_dict(tcr_dict,chain='B',vbseqs=None,jbseqs=None,vaseqs=None,jaseqs=None,guess01=False):
    """Turn tcr_dict (from get_tcr_dict) to tcru_dict of unique TCRs V(D)J sequences"""

    tcru_dict={}
    for e in tcr_dict:
        for row in tcr_dict[e]:
            key =''
            bstart=0
            if 'A' in chain:
                cdr3a,va,ja = row[0],row[1],row[2]
                longA = determine_tcr_seq_vj(cdr3a,va,ja,vaseqs,jaseqs,guess01)
                #longA = get_long(va,ja,cdr3a,vaseqs,jaseqs,chain='A')
                key += longA + (len(chain)>1)*'+'
                bstart=3
            if 'B' in chain:
                cdr3b,vb,jb = row[bstart],row[bstart+1],row[bstart+2]
                longB = determine_tcr_seq_vj(cdr3b,vb,jb,vbseqs,jbseqs,guess01)
                #longB = get_long(vb,jb,cdr3b,vbseqs,jbseqs,chain='B')
                key += longB

            # If the tcr exicst in dict, concatenate new epitope to the epitopes the tcr can recognize
            if key in tcru_dict:
                if e not in tcru_dict[key][0]:
                    tcru_dict[key][0]+=' '+e
            else:
                new_row = [e]
                new_row +=row[bstart+3:bstart+4]+row[0:3]
                if 'A' in chain:
                    new_row += [longA]
                if 'B' in chain:
                    if bstart>0:
                        new_row += row[3:6]
                    new_row += [longB]
                tcru_dict[key]=new_row

    return tcru_dict


def plot_tcrs_per_epitope(epis_u,num_epis,dpi=150,sort=True,scale='log'):
    n=len(epis_u)
    if sort:
        I=np.flip(np.argsort(num_epis))
        epis_u=np.array(epis_u)[I]
        num_epis=np.array(num_epis)[I]
    plt.figure(figsize=(n*0.2+.5,2),dpi=150)
    plt.grid(axis='y',b=True,c='k')
    plt.grid(axis='y',b=True,which='minor',linewidth=0.5)
    plt.gca().set_yscale(scale)

    plt.bar(range(n),num_epis,lw=0,alpha=.8)
    plt.xticks(range(n),epis_u,rotation=90,fontsize=5)
    plt.xlim([-.75,n-.25])
    plt.ylabel('Number of TCRs/epitope')
    plt.xlabel('Epitope')
    plt.show()


def write_data_to_file(tcru_dict,filename,chain='B'):
    """Given dictionary of unique TCRs (from get_unique_tcr_dict), writes data to given file"""
    with open(filename,'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        # Header
        if chain == 'AB':
            writer.writerow('Epitope Subject CDR3A VA JA LongA CDR3B VB JB LongB'.split())
        elif chain == 'B':
            writer.writerow('Epitope Subject CDR3B VB JB LongB'.split())
        elif chain == 'A':
            writer.writerow('Epitope Subject CDR3A VA JA LongA'.split())
        # Data
        for tab in tcru_dict:
            writer.writerow(tcru_dict[tab])


def get_labels(epis,epis_u_fixed=None, fixedplus=False,delim=' '):
    """ epis: list or array of epitopes for which the labels will be extracted
        epis_u_fixed: list or array (of length l) of unique epitopes that will define the first l labels
        fixedplus: If True and epis_u_fixed is not None, define labels from additional epitopes in epis
            that don't occur in epis_u_fixed. These will be added to the end of labels in alphabetical order
        delim: delimiter that separates epitopes of cross-reactive TCRs
    """

    if epis_u_fixed is not None and ~fixedplus:
        epis_u = epis_u_fixed
    else:
        epis_u0 = np.unique(epis)
        epis_u=[]
        for epis_i in epis_u0:
            for e in epis_i.split(delim):
                epis_u.append(e)
        epis_u=np.unique(epis_u)
        if fixedplus:
            epis_u=np.concatenate((epis_u_fixed,list(filter(lambda epi: epi not in epis_u_fixed,epis_u))))

    labels_ar=np.zeros((len(epis),len(epis_u)),dtype=bool)
    for i,epis_i in enumerate(epis):
        for e in epis_i.split(' '):
            ind = np.nonzero(epis_u==e)[0]
            labels_ar[i][ind]=1

    return epis_u,labels_ar

def get_folds(labels,order=None,k=20,orderby='class_size',Itest=None,old_classes=None,update_order=True):
    """ Select stratified folds.
    If Itest is given, it can contain predetermined folds for some TCRs, others are marked by -1
    """

    n_tcrs, n_categories = labels.shape

    if order is None:
        if orderby=='xreact':
            numbers=[]
            for i in range(n_categories):
                numbers.append(np.sum(labels[labels[:,i],:i]) + np.sum(labels[labels[:,i],i+1:]))
            numbers = np.asarray(numbers)
            order=list(np.flip(np.argsort(numbers)))
        else:
            numbers=np.sum(labels,axis=0)
            order=list(np.argsort(numbers))
    if Itest is None:
        Itest=-1*np.ones((n_tcrs),dtype=int)
        Iused = np.zeros((n_tcrs),dtype=bool)
    else:
        Iused = Itest != -1
        print(order)
        order_old=order.copy()
        order=[]
        for i in range(len(order_old)):
            if order_old[i] in old_classes:
                order.append(order_old[i])
        for i in range(len(order_old)):
            if order_old[i] not in old_classes:
                order.append(order_old[i])
        print(order)


    # sort categories to get more balanced folds
    i=0
    while len(order)>0:
        # override previous order if there are classes from which some TCRs are already in use
        if update_order:
            nums_used = [np.sum(labels[Iused,i]) for i in order]
            i = np.argmax(nums_used)
            if nums_used[i]==0:
                i=0
        icat=order[i]
        order.pop(i)

        I = np.nonzero(labels[:,icat])[0] # All TCRs in this category
        #print('{:d}. total: {:d}, used: {:d} not used: {:d}'.format(icat,len(I),np.sum(Iused[I]),np.sum(~Iused[I])))
        if all(Iused[I]): # If there are no more TCRs left to assing to a fold, move on
            continue
        Ir = []
        ni=len(I)
        ni_left = ni

        # sort fold nums to get more balanced folds
        #fold_nums= np.flip(np.argsort([np.sum(Itest==i) for i in range(k)])) # start with the fold with currently most TCRs

        fold_nums = np.flip(np.argsort([np.sum(Iused[Itest==ik]) for ik in range(k)]))
        #print('Fold_nums:',fold_nums,'Fold_sizes:',[np.sum(Iused[Itest==ik]) for ik in fold_nums])
        for ik,ind in enumerate(fold_nums):
            num = int(ni_left/(k-ik)) # Number of TCRs that should be in fold ind
            num_already = np.sum(Itest[labels[:,icat]] == ind) # TCRs of this class already in fold ind
            num_add = num-num_already # number of TCRs to add in addition to the existing ones

            Ir+=[ind]*num_add # Define how many TCRs are added to each fold
            ni_left-=num

        Irs,Is,sds = 0,0,n_tcrs
        #assign to folds 50 times, select the one with less deviation in sum of assigned labels
        j=0
        while j<50 and sds>1:
            # order TCRs that will be assign to folds randomly and
            # select only TCRs of the selected category, that have not been assigned yet
            Irt = np.random.permutation(Ir)
            I = np.nonzero(np.logical_and(labels[:,icat],~Iused))[0]

            tmp = [np.sum(labels[I[Irt==ik]]) for ik in range(k)]
            sd = np.std(tmp)
            if sd < sds:
                sds = sd
                #print('{:.3f}'.format(sd),end=' ')
                Irs = Irt
                Is = I
            j+=1
        #print('')
        Ir=Irs
        I=Is

        for ik in range(k):
            Itest[I[Ir==ik]] = ik
        Iused[Itest!=-1]=True

    return Itest



def checkFoldBalance(labels,Itest,k=20):
    fold_sizes = np.asarray([np.sum(Itest==i) for i in range(k)])
    med = np.median(fold_sizes)
    print('Fold size, median: {:.1f}, max deviation: {:.1f}'.format(med,np.max(np.abs(fold_sizes-med))))

    ne = labels.shape[1]
    size_array= np.zeros((k,ne))
    for i in range(0,k):
        lab_i = labels[Itest==i]
        size_array[i,:]=np.asarray([np.sum(lab_i[:,i]) for i in range(ne)])

    med = np.median(size_array,axis=0)
    size_dif = size_array-np.tile(med,(k,1))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(ne*.3+.5+ 2, 2.5), dpi=200,
                       gridspec_kw ={'width_ratios': [ne*.3+.5, 2],'wspace': 0.1})


    im = ax[0].imshow(size_dif,interpolation='nearest',cmap=plt.get_cmap('RdBu',9))
    ax[0].set_xlabel('Classes')
    ax[0].set_xticks(range(ne))
    ax[0].set_xticklabels(range(ne),fontsize=7)
    ax[0].set_ylabel('fold')
    ax[0].set_title('Class size deviation from median')
    tic = MultipleLocator(1)
    plt.colorbar(im,pad=0.01,ticks=tic,ax=ax[0])

    ax[1].barh(range(k),fold_sizes,height=.7,color='k')
    ax[1].set_ylim([k-.5,-.5])
    ax[1].set_xlim([100*int(min(fold_sizes)/100),100*np.ceil(max(fold_sizes)/100)])
    ax[1].set_title('Fold sizes')
    plt.show()


def get_occmat(epis, epis_u, delim=' '):
    _,labels_ar = get_labels(epis,epis_u,delim=delim)
    occmat=np.zeros((len(epis_u),len(epis_u)))
    for labs in labels_ar:
        inds=np.nonzero(labs)[0]
        for i in inds:
            occmat[i,:]+=labs
    return occmat / np.transpose(np.sum(labels_ar,0,keepdims=True))


def plot_cross_reactivity(crmat,epis,fs=(12,12),bar_lim=1.5,dpi=300,diagblocks=None,xfix=0,yfix=0):
    nc=8
    n = crmat.shape[0]
    ma=np.max(crmat-np.eye(n))
    ymax = np.ceil(ma*10)/10
    ymax=0.6

    cmapc = plt.get_cmap('magma_r',int(nc*ymax*10-1)).colors
    cmapc[:,3]=0.8
    colours=np.concatenate((np.concatenate(([[1,1,.99,.8]],cmapc)),[[0,0,0,1]]))
    cm2 = mpcolors.LinearSegmentedColormap.from_list('mymap', colours, N=len(colours))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.0

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.15]
    rect_histy = [left + width + spacing, bottom, 0.15, height]
    rect_cb = [left + width + 0.15 + spacing+0.01,bottom,0.05,height]

    # start with a square Figure
    fig = plt.figure(figsize=fs,dpi=dpi)

    ax = fig.add_axes(rect_scatter)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_cb = fig.add_axes(rect_cb)

    ax_histy.tick_params(axis="y", labelleft=False,direction='in')

    im = ax.imshow(crmat,cmap=cm2,vmin=0,vmax=ymax+0.1/8)
    if diagblocks is not None:
        d=diagblocks
        for i in range(len(diagblocks)-1):
            for x1,x2,y1,y2 in [[i,i,i,i+1],[i+1,i+1,i,i+1],[i,i+1,i,i],[i,i+1,i+1,i+1]]:
                ax.plot([d[x1]-.5+xfix,d[x2]-.5+xfix],[d[y1]-.5+yfix,d[y2]-.5+yfix],color='k',alpha=.3,lw=.3,zorder=3)

    ax.set_ylim(-.5,n-.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(epis,fontsize=10,va='baseline')
    ax.set_xticks(range(n))
    ax.set_xticklabels(epis,rotation=90,fontsize=10,ha='center')

    # Average number of specificites per TCR
    ax_histy.barh(range(n),np.sum(crmat,axis=1),color=[0.2,0.2,0.2])
    a = ax_histy.axvline(x=0,ymin=0,ymax=n,color='k',linewidth=0.75)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.grid(True,'major','x')

    if bar_lim==1.6:
        ax_histy.set_xticks([1,1.2,1.4,1.6])
        ax_histy.set_xticklabels(['1','1.2','1.4','1.6'])
    elif bar_lim==2:
        ax_histy.set_xticks([1,1.25,1.5,1.75,2])
        ax_histy.set_xticklabels(['1','','1.5','','2'])
    elif bar_lim==2.2:
        ax_histy.set_xticks([1,1.2,1.4,1.6,1.8,2,2.2])
        ax_histy.set_xticklabels(['1','1.2','1.4','1.6','1.8','2','2.2'])
    else:
        ax_histy.set_xticks([1,1.5,2,2.5])
        ax_histy.set_xticklabels(['1','1.5','2','2.5'])
    ax_histy.set_xlim(1)
    ax_histy.set_ylim(-.5,n-.5)

    ax_cb.set_frame_on(False)
    ax_cb.set_visible(False)
    cb = plt.colorbar(im,ax=ax_cb,fraction=1)
    cb.set_ticks(np.concatenate((np.arange(0,ymax+0.0001,0.1),[ymax+0.1/nc])))
    cb.set_ticklabels(['{:.1f}'.format(s) for s in np.concatenate((np.arange(0,ymax+0.0001,0.1),[1]))])
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if not ax_histy.yaxis_inverted():
        ax_histy.invert_yaxis()
    plt.show()
