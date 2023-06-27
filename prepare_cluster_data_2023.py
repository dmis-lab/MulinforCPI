from scipy.cluster.hierarchy import ward,fcluster, linkage, single
from scipy.spatial.distance import pdist
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist 
from tape import ProteinBertModel, TAPETokenizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
from sklearn.metrics import jaccard_score, silhouette_score , accuracy_score


tokenizer = TAPETokenizer(vocab='unirep')

def take_sequencefp(sequence):
    dummy_array = [0]*500
    arr = list(tokenizer.encode(list(sequence))) + dummy_array 
    while len(arr)>500:
        arr.pop(len(arr)-1)
    return np.zeros(500)+np.array(arr)

def get_fps(list_smiles):

    fps = []
    fps_mol = []
    for smile in list_smiles:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048,useChirality=True)
        fps_mol.append(fp)
        fp_vec = np.array(fp)
        fps.append(fp_vec)
    return fps, fps_mol


def kmeans(data,k=5, no_of_iterations=100):
    metric = 'euclidean' #'euclidean'
    pca = PCA(2)
    df = pca.fit_transform(data)

    idx = np.random.choice(len(df), k, replace=False)
    #Randomly choosing Centroids 
    centroids = df[idx, :] #Step 1
     
    #finding the distance between centroids and all the data points
    distances = cdist(df, centroids, metric) #Step 2
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
     
    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = df[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) #Updated Centroids     
        distances = cdist(df, centroids, metric)
        points = np.array([np.argmin(i) for i in distances])

    label = points
    #Visualize the results
 
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
    plt.legend()
    plt.show()

    score = silhouette_score(df, points, metric='euclidean')
    print(score)
    return points 

def cluster_protein(datam, dist_threshold = 0.5):
    pca = PCA(2)
    df = pca.fit_transform(data)
    distance_matrix = ward(pdist(df))
    a = fcluster(distance_matrix, t=0.9, criterion='distance')

def test_cluster(data):
    pca = PCA(2)
    df = pca.fit_transform(data)
    #Initialize the class object
    kmeans = KMeans(n_clusters= 4)
     
    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
     
    #Getting unique labels
    u_labels = np.unique(label)
     
    return label

def ClusterFps(fps, distThresh):
    #disThresh: the threshold for the distance following tanimoto similarity
    #fps: list of fingerprints
    # first generate the distance matrix:
    dists = []
    # dist is the part of the distance matrix below the diagonal as an array:
    # 1.0, 2.0, 2.1, 3.0, 3.1, 3.2 ...
    nfps = len(fps)
    matrix = []
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])
        matrix.append(sims)

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, distThresh, isDistData=True)

    return cs,dists,matrix
def save_csv(train_dataframe, test_dataframe, val_dataframe, headers, folder, task, fold):
    if not os.path.exists(folder): os.makedirs(folder)
    if not os.path.exists(os.path.join(folder,task)): os.makedirs(os.path.join(folder,task))

    train_dataframe[headers].to_csv(os.path.join(os.path.join(folder,task),'{}_{}_train.csv'.format(task,fold)), index = False)
    test_dataframe[headers].to_csv(os.path.join(os.path.join(folder,task),'{}_{}_test.csv'.format(task,fold)), index = False)
    val_dataframe[headers].to_csv(os.path.join(os.path.join(folder,task),'{}_{}_val.csv'.format(task,fold)), index = False)


def main(input_file,folder):
    data_frame = pd.read_csv(input_file)
    list_unique_smiles = list(set(list(data_frame['smiles'])))
    fps_list, fps_mol = get_fps(list_unique_smiles) 

    headers = [col for col in data_frame.columns]
    # making clusters for compounds (distance threshold = 1 - similarity)
    comp_clusters, dists, matrix = ClusterFps(fps_mol, 0.5)
    list_unique_prots = list(set(list(data_frame['sequence'])))
    sequence_fingerprint_train = [take_sequencefp(seq) for seq in list_unique_prots]

    # making clusters fo proteins
    prot_cluster = kmeans(sequence_fingerprint_train,k=5)

    five_fold = KFold(n_splits=5)
    fold = 0
    for compound_index, protein_index in \
        zip(five_fold.split(comp_clusters),five_fold.split(np.unique(prot_cluster))):
        compound_train_index, compound_test_index = compound_index[0],compound_index[1]
        protein_train_index, protein_test_index = protein_index[0],protein_index[1]
        
        # take real smile
        comp_train,comp_test, prot_train, prot_test = [],[],[],[]
        for i in compound_train_index:
            for index in range(len(comp_clusters[i])):
                comp_train.append(list_unique_smiles[comp_clusters[i][index]])
        for i in compound_test_index:
            for index in range(len(comp_clusters[i])):
                comp_test.append(list_unique_smiles[comp_clusters[i][index]])

        # take real protein sequence
        for i in protein_train_index:
            for index in range(len(prot_cluster)):
                if prot_cluster[index] == i:
                    prot_train.append(list_unique_prots[index])
        for i in protein_test_index:
            for index in range(len(prot_cluster)):
                if prot_cluster[index] == i:
                    prot_test.append(list_unique_prots[index])

        # novel_pair setting
        train_dataframe = pd.DataFrame() 
        test_dataframe = pd.DataFrame()
        # novel_pair
        for i in tqdm(range(len(data_frame))):
            if data_frame.iloc[i]['smiles'] in comp_train and data_frame.iloc[i]['sequence'] in prot_train:
                train_dataframe = train_dataframe.append(data_frame.iloc[i], ignore_index = True)
            if data_frame.iloc[i]['smiles'] in comp_test and data_frame.iloc[i]['sequence'] in prot_test:
                test_dataframe = test_dataframe.append(data_frame.iloc[i], ignore_index = True)
        val_dataframe = train_dataframe[headers].sample(frac = 0.2)
        train_datafame_after = train_dataframe.loc[train_dataframe.index.difference(val_dataframe.index), ]
        save_csv(train_datafame_after, test_dataframe, val_dataframe, headers, folder, 'novel_pair',fold)
       
        # newcomp
        train_dataframe = pd.DataFrame() 
        test_dataframe = pd.DataFrame()
        for i in tqdm(range(len(data_frame))):
            if data_frame.iloc[i]['smiles'] in comp_train:
                train_dataframe = train_dataframe.append(data_frame.iloc[i], ignore_index = True)
            else:
                test_dataframe = test_dataframe.append(data_frame.iloc[i], ignore_index = True)
        val_dataframe = train_dataframe[headers].sample(frac = 0.2)
        train_datafame_after = train_dataframe.loc[train_dataframe.index.difference(val_dataframe.index), ]
        save_csv(train_datafame_after, test_dataframe, val_dataframe, headers, folder, 'novel_comp',fold)        
        
        # newprot
        train_dataframe = pd.DataFrame() 
        test_dataframe = pd.DataFrame()
        for i in tqdm(range(len(data_frame))):
            if data_frame.iloc[i]['sequence'] in prot_train:
                train_dataframe = train_dataframe.append(data_frame.iloc[i], ignore_index = True)
            else:
                test_dataframe = test_dataframe.append(data_frame.iloc[i], ignore_index = True)        
        val_dataframe = train_dataframe[headers].sample(frac = 0.2)
        train_datafame_after = train_dataframe.loc[train_dataframe.index.difference(val_dataframe.index), ]
        save_csv(train_datafame_after, test_dataframe, val_dataframe, headers, folder, 'novel_prot', fold) 
        fold = fold + 1

def check_dup(folder):
    for fold in tqdm(range(5)):
        train_all_path = os.path.join(folder,r'novel_pair\novel_pair_{}_train.csv'.format(fold))

        val_path = os.path.join(folder,r'novel_pair\novel_pair_{}_val.csv'.format(fold))
        test_path = os.path.join(folder,r'novel_pair\novel_pair_{}_test.csv'.format(fold))
        
        df_train = pd.read_csv(train_all_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        df_train = pd.concat([df_train,df_val])

        sequence_test_list_uni = list(set(list(df_test[df_test.columns[1]])))
        sequence_train_list_uni = list(set(list(df_train[df_train.columns[1]])))

        smiles_test_list_uni = list(set(list(df_test[df_test.columns[0]])))
        smiles_train_list_uni = list(set(list(df_train[df_train.columns[0]])))

        sequence_fingerprint_train = [take_sequencefp(seq) for seq in sequence_train_list_uni]
        sequence_fingerprint_test = [take_sequencefp(seq) for seq in sequence_test_list_uni]

        _, morgan_fingerprint_train = get_fps(smiles_train_list_uni)
        _, morgan_fingerprint_test = get_fps(smiles_test_list_uni)

       
        jac_sim_seq = []
        jac_sim_smi = []
        i = 0
        for seqtest_fp in sequence_fingerprint_test:
            for seqtrain_fp in sequence_fingerprint_train:
                jac_sim_seq1 = accuracy_score(seqtest_fp, seqtrain_fp)
                jac_sim_seq.append(jac_sim_seq1)
        i = 0
        for i, morgantest_fp in enumerate(morgan_fingerprint_test):
            for morganfptrain_fp in morgan_fingerprint_train:  
                jac_sim_smi.append(DataStructs.TanimotoSimilarity(morgantest_fp, morganfptrain_fp))
        print('fold', fold)
        print(min(jac_sim_seq))
        print(max(jac_sim_seq))
        print(min(jac_sim_smi))
        print(max(jac_sim_smi))
        print('compound_prot fold : {} +_ {}'.format( np.mean(jac_sim_seq),np.std(jac_sim_seq)))
        print('compound_smi fold : {} +_ {}'.format(np.mean(jac_sim_smi),np.std(jac_sim_smi)))

def make_val_set_fromhard(input_file_train,input_file_test,folder,fold):
    train_dataframe = pd.read_csv(input_file_train)
    test_dataframe = pd.read_csv(input_file_test)
    headers = [col for col in test_dataframe.columns]
    val_dataframe = train_dataframe[headers].sample(frac = 0.2)
    train_datafame_after = train_dataframe.loc[train_dataframe.index.difference(val_dataframe.index), ]
    save_csv(train_datafame_after, test_dataframe, val_dataframe, headers, folder, 'novel_pair',fold)        

if __name__ == '__main__':
    out_put_folder = str(sys.argv[2])
    input_file = str(sys.argv[1])
    main(input_file,out_put_folder)
