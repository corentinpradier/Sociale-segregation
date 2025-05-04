import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from itertools import product
import random
from tqdm import tqdm


class Segregation:

    def __init__(self, rows: int = 30, cols: int = 30, vaccum: int = 10, tolerence: float = 0.5):
        self.rows = rows
        self.cols = cols
        self.vaccum = vaccum
        self.tolerence = tolerence

        self.total_number = rows * cols - vaccum
        self.mat_initial = np.zeros((rows, cols))



        self.pairs, self.blue_pairs, self.red_pairs, self.vac, self.mat_initial = self._initials_conditions(self.mat_initial, self.total_number)
        self.mat_final = self.mat_initial.copy()

    def _generate_unique_index_pairs(self, N):
        indices = list(product(range(self.rows), repeat=2))
        unique_pairs = random.sample(indices, N)
        return unique_pairs
    
    def _initials_conditions(self, mat, N):
        pairs = np.array(self._generate_unique_index_pairs(N))
        blue_pairs = pairs[:int(N/2),:]
        red_pairs = pairs[int(N/2):,:]

        mat[blue_pairs[:,0], blue_pairs[:,1]] = 1
        mat[red_pairs[:,0], red_pairs[:,1]] = -1
        vac1 = np.where(mat == 0)[0]
        vac2 = np.where(mat == 0)[1]
        vac = np.column_stack((vac1, vac2))

        return pairs, blue_pairs, red_pairs, vac, mat
    
    def _satisfaction_condition(self, Nd, Ns):
        return 1 if Nd <= self.tolerence * (Nd + Ns) else 0
    
    def one_step(self, t = 10_000):
        t_range = np.arange(0, t, 1)
        for time in tqdm(t_range, desc="Segregation in progress"):
            pairs = self.pairs.copy()
            vac = self.vac.copy()

            indices = list(range(len(pairs)))
            random.shuffle(indices)
            pairs = np.array([pairs[i] for i in indices])

            indices_vac = list(range(len(vac)))
            random.shuffle(indices_vac)
            vac = np.array([vac[i] for i in indices_vac])

            ind_random_pairs = np.random.randint(len(pairs))
            ind_random_vac = np.random.randint(len(vac))

            pp = pairs[ind_random_pairs]
            pv = vac[ind_random_vac]

            mat_border_cond = np.zeros((self.rows + 2, self.cols + 2))
            mat_border_cond[1:-1, 1:-1] = self.mat_final

            ind_p_1, ind_p_2 = pp
            ind_v_1, ind_v_2 = pv

            a = []
            for p in (pp, pv):
                if p[0] != 0 and p[0] != self.rows-1 and p[1] != 0 and p[1] != self.cols-1:
                    a += [
                        self.mat_final[p[0]-1, p[1]-1], self.mat_final[p[0]-1, p[1]],
                        self.mat_final[p[0]-1, p[1]+1], self.mat_final[p[0], p[1]+1],
                        self.mat_final[p[0]+1, p[1]+1], self.mat_final[p[0]+1, p[1]],
                        self.mat_final[p[0]+1, p[1]-1], self.mat_final[p[0], p[1]-1]
                    ]
                else:
                    a += [
                        mat_border_cond[p[0]-1, p[1]-1], mat_border_cond[p[0]-1, p[1]], 
                        mat_border_cond[p[0]-1, p[1]+1], mat_border_cond[p[0], p[1]+1], 
                        mat_border_cond[p[0]+1, p[1]+1], mat_border_cond[p[0]+1, p[1]], 
                        mat_border_cond[p[0]+1, p[1]-1], mat_border_cond[p[0], p[1]-1]
                    ]

            a1, a_v = a[:8], a[8:]
            Nd, Ns, Nv = 0, 0, 0
            for i in a1:
                if self.mat_final[ind_p_1, ind_p_2] == -1:
                    Ns += i == -1
                    Nd += i == 1
                    Nv += i == 0
                elif self.mat_final[ind_p_1, ind_p_2] == 1:
                    Ns += i == 1
                    Nd += i == -1
                    Nv += i == 0

            Nd_v, Ns_v, Nv_v = 0, 0, 0
            for i in a_v:
                if self.mat_final[ind_p_1, ind_p_2] == -1:
                    Ns_v += i == -1
                    Nd_v += i == 1
                    Nv_v += i == 0
                elif self.mat_final[ind_p_1, ind_p_2] == 1:
                    Ns_v += i == 1
                    Nd_v += i == -1
                    Nv_v += i == 0

            sat_cond_pair = self._satisfaction_condition(Nd, Ns)
            sat_cond_vac = self._satisfaction_condition(Nd_v, Ns_v)

            if sat_cond_vac >= sat_cond_pair:
                self.mat_final[ind_v_1, ind_v_2] = self.mat_final[ind_p_1, ind_p_2]
                self.mat_final[ind_p_1, ind_p_2] = 0

                pairs = np.delete(pairs, ind_random_pairs, axis=0)
                vac = np.delete(vac, ind_random_vac, axis=0)
                
                pairs = np.concatenate((pairs, np.array([pv])))
                vac = np.concatenate((vac, np.array([pp])))

            if time == int(t/2):
                self.mat_intermediate = self.mat_final.copy()

            self.pairs = pairs
            self.vac = vac

    def get_data(self, data_type="start"):
        if data_type == "start":
            return self.mat_initial
        elif data_type == "end":
            return self.mat_final
        else:
            raise ValueError("Invalid data_type. Use 'start' or 'end'.")
    
    def _plot_matrix(self, mat, title, axis, ax, norm, cmap):
        list_xticks = [i if i % 2 == 0 else '' for i in range(len(self.mat_initial[0]) + 1)]

        ax.grid(linewidth=1, color='white', axis="both")
        # ax.axis([0, self.rows, 0, self.cols])
        ax.imshow(mat, extent=[0, self.rows, self.cols, 0], cmap=cmap, norm=norm)
        # ax.xaxis.tick_top()
        # ax.xaxis.set_label_position('top')
        # ax.invert_yaxis()
        
        ax.set_xticks(np.arange(0, self.rows + 1, 1))
        ax.set_xticklabels(list_xticks)
        ax.set_yticks(np.arange(0, self.cols + 1, 1))
        ax.set_yticklabels(list_xticks)
        
        if axis:
            ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                        left=False, right=False, labeltop=False, labelleft=False)
            
            ax.grid(False)  # Supprimer la grille si nécessaire

        ax.set_title(title)

    def plot(self, axis=False, colors = ['red', 'white', 'blue'], titles=['Initial Matrix', 'Intermediate Matrix', 'Final Matrix']):
        values = [-1.5, -0.5, 0.5, 1.5]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(values, cmap.N)
    
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))  # Créer deux axes côte à côte
        self._plot_matrix(self.mat_initial, titles[0] if titles else None, axis, axs[0], norm, cmap)
        self._plot_matrix(self.mat_intermediate, titles[1] if titles else None, axis, axs[1], norm, cmap)
        self._plot_matrix(self.mat_final, titles[2] if titles else None, axis, axs[2], norm, cmap)
        plt.tight_layout()
        plt.show()
