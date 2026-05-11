# Modified by YPC on 2023.02.06
# Change variable names, add "Positions = newPositions" and change the dimension of newPositions matrix by "newPositions = np.array([(r.dot(pos.T)).T" in superimpose2mean() function.
# Add cum_explained_var_ratio by "cum_explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)" to getPCA() function.


import numpy as np
from MDAnalysis.analysis import align
from sklearn.decomposition import PCA

class TrajectoryPCA(object):

    def __init__(self, AtomGroup):
        self._AtomGroup = AtomGroup
        self._isSuperimpose = False
        self._isPCA = False
    
    def superimpose2mean(self, cutoff=1e-6, showProcess=True):
        
        if self._isSuperimpose:
            return

        self._isSuperimpose = True

        rmsd = 1
        roundNum = 0
        Positions = [self._AtomGroup.positions - self._AtomGroup.center_of_geometry() for _ in self._AtomGroup.universe.trajectory]
        MeanStructure = Positions[0]
        
        while rmsd > cutoff:
            roundNum+=1
            RMSD = []
            RotateMatrix = []

            for position in Positions:
                R, rmsd = align.rotation_matrix(position, MeanStructure)
                RotateMatrix.append(R)
                RMSD.append(rmsd)

                if roundNum ==1:
                    self._RMSD1 = np.array(RMSD)
                
            newPositions = np.array([(r.dot(pos.T)).T for r, pos in zip(RotateMatrix, Positions)])
            newMeanStructure = newPositions.mean(axis=0)
            _, rmsd = align.rotation_matrix(newMeanStructure, MeanStructure)
            MeanStructure = newMeanStructure
            Positions = newPositions
            
            if showProcess:
                print(str(roundNum)+': '+str(rmsd))
        
        self._RMSD = np.array(RMSD)
        self.MeanStructure = MeanStructure
        self._newPositions = newPositions

        return newMeanStructure

    def getPCA(self):

        if self._isPCA:
            return

        if not self._isSuperimpose:
            self.superimpose2mean()

        self._isPCA = True

        shape = self._newPositions.shape
        flatPos = np.array(self._newPositions).reshape(shape[0], shape[1]*shape[2])
        pca = PCA()
        pca.fit(flatPos)
        pca.transform(flatPos)
        self.eigen_vals = pca.explained_variance_
        self.eigen_vecs = pca.components_
        cum_explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        return cum_explained_var_ratio, pca.components_

    def project2PC12(self):

        if not self._isSuperimpose:
            self.superimpose2mean()

        shape = self._newPositions.shape
        flatPos = np.array(self._newPositions).reshape(shape[0], shape[1]*shape[2])
        pca_2 = PCA(n_components=2)
        pca_2.fit(flatPos)

        return pca_2.transform(flatPos), pca_2.components_

    def getRMSD(self, target='first'):

        if not self._isSuperimpose:
            self.superimpose2mean()
        
        if target == 'mean':
            return self._RMSD
        elif target == 'first':
            return self._RMSD1
        else:
            print('target should be either first or mean')
            return