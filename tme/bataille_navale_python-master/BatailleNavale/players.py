import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Union, Iterable, Callable, Optional, List
from random import randint, random
from BatailleNavale.bataille import Bataille

#Player de base implémentant toutes les fonctions utiles pour les sous-classe.
#Ce player est juste un joueur random qui ne retient pas les coups qu'il fait
class Player:
    def __init__(self, *args):
        if len(args) > 0 and isinstance(args[0], Bataille):
            self.grille = args[0]
        else:
            self.grille = Bataille(*args)
        self.coup = None
        self.result = None

    def next_play(self):
        return self.num_to_pos(randint(0,self.shape[0] * self.shape[1] - 1))

    def resolve(self):
        i = 0
        while not self.victoire():
            self.coup = self.next_play()
            self.result = self.joue(self.coup[0], self.coup[1])
            i += 1
        return i

    def resolve_and_reset(self):
        i = self.resolve()
        self.reset()
        return i

    def reset(self):
        self.grille.reset()
        self.coup = None
        self.result = None


    def num_to_pos(self, num):
        return (num % self.shape[0], num // self.shape[0])

    def ind_to_pos(self, num):
        return (num // self.shape[0], num % self.shape[0])


    def __getattr__(self, attr):
        return getattr(self.grille, attr)


#  RandomPlayer qui ne rejoue jamais les mêmes coups il gagne forcément en 100 coups ou moins
class RandomPlayer(Player):
    def __init__(self, *args):
        super().__init__(*args)
        self.limit = self.grille.shape[0] * self.grille.shape[1] - 1
        self.coup_restants = np.arange(self.limit + 1, dtype="int16")
        np.random.shuffle(self.coup_restants)
        self.cur = -1

    

    def next_play(self):
        self.cur += 1
        if self.cur > self.limit:
            return None
        return self.num_to_pos(self.coup_restants[self.cur])

    def reset(self):
        super().reset()
        np.random.shuffle(self.coup_restants)
        self.cur = -1


# HeuristicPlyaer, a le même comportement que random player, à la différence qu'il chasse les bateaux qu'il touche
class HeuristicPlayer(RandomPlayer):

    def __init__(self, *args):
        super().__init__(*args)
        self.prochains_coups = []
        self.coups_joues = set()

    def next_play(self):
        if self.coup != None:
            self.coups_joues.add(self.coup)
        if self.result != None and self.result[0] == True :
            if self.result[1] == None : 
                x, y = self.coup
                self.prochains_coups.append((x + 1, y))
                self.prochains_coups.append((x - 1, y))
                self.prochains_coups.append((x, y + 1))
                self.prochains_coups.append((x, y - 1))
        
        coup = None
        if len(self.prochains_coups) > 0:
            coup = self.prochains_coups.pop()
            while len(self.prochains_coups) > 0 and (coup in self.coups_joues or not self.grille.position_correcte(coup[0], coup[1], 0, 1)):
                coup = self.prochains_coups.pop()
        if coup == None or coup in self.coups_joues or not self.grille.position_correcte(coup[0], coup[1], 0, 1):
            coup = self.num_to_pos(randint(0,99))
            while coup in self.coups_joues:
                coup = self.num_to_pos(randint(0,99))
        return coup

    
    def reset(self):
        super().reset()
        self.prochains_coups = []
        self.coups_joues = set()



class ProbabilistePlayer(Player):

    def __init__(self, *args, mode=-5, logs=False):
        super().__init__(*args)
        self.custom_grid = Bataille(mode='Custom', shape=self.grille.shape)
        self.custom_grid.board = np.ones(self.grille.shape, dtype=np.int8)

        self.probas_ligne = np.zeros(self.grille.shape, dtype=np.int8)
        self.probas_colonne = np.zeros(self.grille.shape, dtype=np.int8)
        self.probas = np.zeros(self.grille.shape, dtype=np.int8)

        self.hunt_mode = False
        self.logs = logs
        if logs:
            self.touche = []
        
        self.mode = mode

        self.calculate_probas()
    
    def calculate_probas(self):
        self.probas_ligne[:,:] = 0
        self.probas_colonne[:,:] = 0

        if self.hunt_mode: 
            for taille in self.tailles_remaining:
                for x in range(self.shape[0] + 1 - taille):
                    for y in range(self.shape[1]):
                        score = np.sum(self.custom_grid.board[x:x+taille, y])
                        if score <= 0:
                            self.probas_ligne[x:x+taille, y] -= score

                for x in range(self.shape[0]):
                    for y in range(self.shape[1] + 1 - taille):
                        score = np.sum(self.custom_grid.board[x, y:y+taille])
                        if score <= 0:
                            self.probas_colonne[x, y:y+taille] -= score

        else:
            for taille in self.tailles_remaining:
                for x in range(self.shape[0] + 1 - taille):
                    for y in range(self.shape[1]):
                        score = np.sum(self.custom_grid.board[x:x+taille, y])
                        if score <= 5:
                            self.probas_ligne[x:x+taille, y] += 1

                for x in range(self.shape[0]):
                    for y in range(self.shape[1] + 1 - taille):
                        score = np.sum(self.custom_grid.board[x, y:y+taille])
                        if score <= 5:
                            self.probas_colonne[x, y:y+taille] += 1
        
        self.probas_ligne[self.custom_grid.board == self.mode] = 0
        self.probas_colonne[self.custom_grid.board == self.mode] = 0
        self.probas[:,:] = self.probas_ligne[:,:] + self.probas_colonne[:,:]

    def recalculate_probas(self):
        x0, y0 = self.coup

        self.probas_ligne[:,y0] = 0
        self.probas_colonne[x0,:] = 0

        if self.hunt_mode:
            for taille in self.tailles_remaining:
                for x in range(self.shape[0] + 1 - taille):
                    score = np.sum(self.custom_grid.board[x:x+taille, y0])
                    if score <= 0:
                        self.probas_ligne[x:x+taille, y0] -= score

                for y in range(self.shape[1] + 1 - taille):
                    score = np.sum(self.custom_grid.board[x0, y:y+taille])
                    if score <= 0:
                        self.probas_colonne[x0, y:y+taille] -= score

        else:
            for taille in self.tailles_remaining:
                for x in range(self.shape[0] + 1 - taille):
                    score = np.sum(self.custom_grid.board[x:x+taille, y0])
                    if score <= 5:
                        self.probas_ligne[x:x+taille, y0] += 1

                for y in range(self.shape[1] + 1 - taille):
                    score = np.sum(self.custom_grid.board[x0, y:y+taille])
                    if score <= 5:
                        self.probas_colonne[x0, y:y+taille] += 1
        
        self.probas_ligne[:,y0][self.custom_grid.board[:,y0] == self.mode] = 0
        self.probas_colonne[:,y0][self.custom_grid.board[:,y0] == self.mode] = 0

        self.probas_ligne[x0,:][self.custom_grid.board[x0,:] == self.mode] = 0
        self.probas_colonne[x0,:][self.custom_grid.board[x0,:] == self.mode] = 0

        self.probas[:,y0] = self.probas_colonne[:,y0] + self.probas_ligne[:,y0]
        self.probas[x0,:] = self.probas_colonne[x0,:] + self.probas_ligne[x0,:]

    def next_play(self):
        if self.result != None:
            reussite, bateau = self.result  
            if not reussite:
                self.custom_grid.board[self.coup] = 100
                self.recalculate_probas()
            elif bateau == None:
                self.custom_grid.board[self.coup] = self.mode
                if not self.hunt_mode:
                    self.hunt_mode = True
                    self.calculate_probas()
                else:
                    self.recalculate_probas()
            else:
                x, y, d, taille, _ = bateau
                if d == 0:
                    self.custom_grid.board[x:x+taille, y] = 100
                else:
                    self.custom_grid.board[x, y:y+taille] = 100
                if np.any(self.custom_grid.board == self.mode):
                    self.recalculate_probas()
                else:
                    self.hunt_mode = False
                    self.calculate_probas()

        if self.logs:
            if self.result != None:
                if self.result[0]:
                    self.touche.append(self.coup)
                    print(f'Hunt Mode : {self.hunt_mode}, Réussi : {self.coup}')
                else:
                    print(f'Hunt Mode : {self.hunt_mode}, Raté   : {self.coup}')
            result = np.max(self.probas) + 3
            for good_coup in self.touche:
                self.probas[good_coup] = result
            plt.imshow(self.probas.T)
            plt.show()
            self.probas= self.probas_ligne + self.probas_colonne
        return self.ind_to_pos(np.argmax(self.probas))

    def reset(self):
        super().reset()
        self.custom_grid.board[:,:] = 1
        self.hunt_mode = False
        self.calculate_probas()



class MonteCarloPlayerNaif(Player):

    def __init__(self, *args, mode=-4, moves=60000, tries=10, logs=False):
        super().__init__(*args)

        self.custom_grid = Bataille(mode='Custom', shape=self.grille.shape)
        self.custom_grid.board[:,:] = 1

        self.probas = np.zeros(self.grille.shape, dtype=np.int32)
        self.nb_touche = 0

        self.mode = mode
        self.moves = moves
        self.tries = tries
        self.logs = logs
        if logs:
            self.touche = []

    def generate_grid(self, nb_touche, bateaux, nb_move, nb_move_max):
        my_moves = 0
        if len(bateaux) == 1:
            move_available = nb_move_max / nb_move
            if nb_touche == 0:
                pos = self.custom_grid.get_possible_pos(bateaux[0], mode=5)

                if len(pos) == 0:
                    return 0

                if move_available >= len(pos):
                    move_available = len(pos)
                else:
                    np.random.shuffle(pos)
                    
                move_available = int(move_available)

                for i in range(move_available):
                    x, y, d = pos[i]

                    if d == 0:
                        self.probas[x:x+bateaux[0], y] += 1
                    else:
                        self.probas[x, y:y+bateaux[0]] += 1
                    my_moves += 1
                
                for x, y, d, taille, _ in self.custom_grid.boats:
                    if d == 0:
                        self.probas[x:x+taille, y] += my_moves
                    else:
                        self.probas[x, y:y+taille] += my_moves
            else:
                pos = self.custom_grid.get_possible_pos(bateaux[0], mode=0)

                if len(pos) == 0:
                    return 0

                if move_available >= len(pos):
                    move_available = len(pos)
                else:
                    np.random.shuffle(pos)

                move_available = int(move_available)

                for i in range(len(pos)):
                    if move_available == 0:
                        return my_moves
                    x, y, d = pos[i]
                    if d == 0:
                        if np.sum(self.custom_grid.board[x:x+bateaux[0], y] == self.mode) < nb_touche:
                            continue
                    else:
                        if np.sum(self.custom_grid.board[x, y:y+bateaux[0]] == self.mode) < nb_touche:
                            continue
                    if d == 0:
                        self.probas[x:x+bateaux[0], y] += 1
                    else:
                        self.probas[x, y:y+bateaux[0]] += 1

                    for x, y, d, taille, _ in self.custom_grid.boats:
                        if d == 0:
                            self.probas[x:x+taille, y] += 1
                        else:
                            self.probas[x, y:y+taille] += 1
                    move_available -= 1
                    my_moves += 1  

        else:
            move_available = int(pow(nb_move_max / nb_move, 1 / len(bateaux)))
            if nb_touche == 0:
                pos = self.custom_grid.get_possible_pos(bateaux[0], mode=5)

                if move_available >= len(pos):
                    move_available = len(pos)
                else:
                    np.random.shuffle(pos)

                if len(pos) == 0:
                    return 0

                move_available = int(move_available)

                for i in range(move_available):
                    x, y, d = pos[i]
                    self.custom_grid.placer(x, y, d, bateaux[0], check=False)
                    my_moves += self.generate_grid(0, bateaux[1:], nb_move * move_available, nb_move_max)
                    self.custom_grid.pop_boat(mode=1)
            else:
                pos = self.custom_grid.get_possible_pos(bateaux[0], mode=0)

                if len(pos) == 0:
                    return 0

                if move_available >= len(pos):
                    move_available = len(pos)
                else:
                    np.random.shuffle(pos)

                move_available = int(move_available)

                for i in range(move_available):
                    x, y, d = pos[i]
                    if d == 0:
                        nb_touche_less = np.sum(self.custom_grid.board[x:x+bateaux[0], y] == self.mode)
                        last_array = self.custom_grid.board[x:x+bateaux[0], y].copy()
                    else:
                        nb_touche_less = np.sum(self.custom_grid.board[x, y:y+bateaux[0]] == self.mode)
                        last_array = self.custom_grid.board[x, y:y+bateaux[0]].copy()
                    self.custom_grid.placer(x, y, d, bateaux[0], check=False)
                    my_moves += self.generate_grid(nb_touche - nb_touche_less, bateaux[1:], nb_move * move_available, nb_move_max)
                    self.custom_grid.pop_boat(mode=1)
                    if d == 0:
                        self.custom_grid.board[x:x+bateaux[0], y] = last_array
                    else:
                        self.custom_grid.board[x, y:y+bateaux[0]] = last_array

        return my_moves
        

    def next_play(self):
        if self.result != None:
            reussite, bateau = self.result  
            if not reussite:
                self.custom_grid.board[self.coup] = 100
            elif bateau == None:
                self.custom_grid.board[self.coup] = self.mode
                self.nb_touche += 1
            else:
                x, y, d, taille, _ = bateau
                if d == 0:
                    self.custom_grid.board[x:x+taille, y] = 100
                else:
                    self.custom_grid.board[x, y:y+taille] = 100
                self.nb_touche -= taille - 1

        self.probas[:,:] = 0
        tailles = self.tailles_remaining

        my_moves = 0

        for i in range(self.tries):
            np.random.shuffle(tailles)
            my_moves += self.generate_grid(self.nb_touche, tailles, 1, self.moves)

        self.probas[self.custom_grid.board == self.mode] = 0
        coup = self.ind_to_pos(np.argmax(self.probas))
        if self.logs:
            if self.result != None:
                if self.result[0]:
                    self.touche.append(self.coup)
                    print(f'Grilles générées : {my_moves}, Réussi : {self.coup}')
                else:
                    print(f'Grilles générées : {my_moves}, Raté   : {self.coup}')
            result = np.max(self.probas) + 3
            for good_coup in self.touche:
                self.probas[good_coup] = result
            plt.imshow(self.probas.T)
            plt.show()
        return coup

        

    def reset(self):
        super().reset()
        self.custom_grid.board[:,:] = 1
        self.nb_touche = 0






class USSPlayer(Player):
    def __init__(self, ps=0.5, probas=np.ones((10,10)), mode='TruthProba', logs=False):
        super().__init__(Bataille(mode='Custom', shape=probas.shape))
        self.probas = probas
        self.probas /= np.sum(self.probas)
        self.probas_tmp = np.copy(self.probas)
        self.ps = ps
        self.mode = mode
        self.logs = logs
        if mode == 'TruthProba':
            coup = random()
            for ind, val in enumerate(np.ravel(self.probas_tmp)):
                if coup < val:
                    self.grille.placer(self.ind_to_pos(ind)[0], self.ind_to_pos(ind)[1], 0, 1)
                    break
                else:
                    coup -= val
        else:
            self.grille.place_alea(1)

    
    def next_play(self):
        if self.coup != None:
            reussite, bateau = self.result
            if not reussite:
                pi_k = (1 - self.ps) / (1 / self.probas_tmp[self.coup] - self.ps)
                self.probas_tmp[self.coup] = pi_k * (1 - self.probas_tmp[self.coup]) / (1 - pi_k)
        
        return self.ind_to_pos(np.argmax(self.probas_tmp))

    def joue(self, x, y):
        if self.logs:
            print(f'Joué : {self.coup}')
            plt.imshow(self.probas_tmp)
            plt.show()
        if random() <= self.ps:
            return self.grille.joue(x, y)
        else:
            return (False, None)

    def reset(self):
        super().reset()
        self.probas_tmp[:] = self.probas[:]
        if self.mode == 'TruthProba':
            self.grille.pop_boat()
            coup = random()
            for ind, val in enumerate(np.ravel(self.probas_tmp)):
                if coup < val:
                    self.grille.placer(self.ind_to_pos(ind)[0], self.ind_to_pos(ind)[1], 0, 1)
                    break
                else:
                    coup -= val
    



class USSRandomPlayer(USSPlayer):
    def next_play(self):
        if self.coup != None:
            reussite, bateau = self.result
            if not reussite:
                pi_k = (1 - self.ps) / (1 / self.probas_tmp[self.coup] - self.ps)
                self.probas_tmp[self.coup] = pi_k * (1 - self.probas[self.coup]) / (1 - pi_k)
        
        coup = random() * np.sum(self.probas_tmp)
        for ind, val in enumerate(np.ravel(self.probas_tmp)):
            if coup < val:
                return self.ind_to_pos(ind)
            else:
                coup -= val





class USSOtherPlayer(USSPlayer):
    def next_play(self):
        if self.coup != None:
            reussite, bateau = self.result
            if not reussite:
                self.probas_tmp[self.coup] /= 2
        
        return self.ind_to_pos(np.argmax(self.probas_tmp))
