import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Union, Iterable, Callable, Optional, List
from random import randint
from .utils import moyenne 

class Bataille:
    def __init__(self, *args, mode="Default", shape: Tuple[int, int] = (10,10)):
        self.__id__ = 0
        self.shape = shape
        self.board = np.zeros(shape, dtype=np.int8)
        self.boats = []
        if len(args) > 0:
            self.genere_grille(*args)
        elif mode == "Default":
            self.genere_grille([2,3,3,4,5])

    def position_correcte(self, x: int, y: int, direction: int, boat: int) -> bool:
        if x < 0 or x >= self.shape[0] or y < 0 or y >= self.shape[1]:
            return False
        elif direction == 0:
            return x + boat <= self.shape[0]
        elif direction == 1:
            return y + boat <= self.shape[1]
        else :
            return False

    def peut_placer(self, x: int, y: int, direction: int, boat: int, mode=0) -> bool :
        if not self.position_correcte(x, y, direction, boat):
            return False
        if direction == 0:
            return np.sum(self.board[x:x + boat, y]) <= mode
        else:
            return np.sum(self.board[x, y:y + boat]) <= mode
    
    def placer(self, x: int, y: int, direction: int, boat: int, check=True) -> bool :
        """ 
        Position : (0 .. shape[0] - 1, 0 .. shape[1] - 1)
        direction : {0 (horizontal), 1 (vertical)}
        boat : {1, .., max(shape)}
        """
        if check:
            if not self.peut_placer(x, y, direction, boat):
                return False
        self.boats.append((x, y, direction, boat, boat))
        if direction == 0:
            self.board[x:x + boat, y] = self.__id__ + 100
        else :
            self.board[x, y:y + boat] = self.__id__ + 100
        self.__id__ += 1
        return True


    def place_alea(self, boat: int, mode='Loop'):
        if mode == 'Loop':
            x = -1
            y = -1
            d = -1
            while not self.peut_placer(x, y, d, boat):
                d = randint(0, 1)
                if d == 0:
                    x = randint(0, self.shape[0] - boat)
                    y = randint(0, self.shape[1] - 1)
                else :
                    x = randint(0, self.shape[0] - 1)
                    y = randint(0, self.shape[1] - boat)
                #print(f'x : {x}, y : {y}, d : {d}')
        else :
            pos = self.get_possible_pos(boat)
            x, y, d = pos[randint(0, len(pos) - 1)]

        self.placer(x, y, d, boat)

    def get_possible_pos(self, boat: int, mode: int = 0):
        pos = []
        for x in range(self.shape[0] + 1 - boat):
            for y in range(self.shape[1]):
                if self.peut_placer(x, y, 0, boat, mode=mode):
                    pos.append((x, y, 0))
        
        for x in range(self.shape[0]):
            for y in range(self.shape[1] + 1 - boat):
                if self.peut_placer(x, y, 1, boat, mode=mode):
                    pos.append((x, y, 1))
        return pos


    def affiche(self):
        plt.imshow(self.board.T)
        plt.show()

    def pop_boat(self, mode=0) -> int:
        self.__id__ -= 1
        boat = self.boats.pop()
        if boat != None:
            if boat[2] == 0:
                self.board[boat[0]:boat[0] + boat[3], boat[1]] = mode
            else :
                self.board[boat[0], boat[1]:boat[1] + boat[3]] = mode
            return boat[3]
        else:
            return 0


    def __eq__(self, other):
        if isinstance(other, Bataille):
            return set(self.boats) == set(other.boats)
        else :
            return np.all(self.board == other)

    @property
    def tailles(self):
        return [taille for (_,_,_,taille,_) in self.boats]

    @property
    def tailles_remaining(self):
        return [taille for (_,_,_,taille,hp) in self.boats if hp > 0]

    def nb_possibilite(self, taille: int):
        count = 0
        for x in range(self.shape[0] + 1 - taille):
            for y in range(self.shape[1]):
                if self.peut_placer(x, y, 0, taille):
                    count += 1

        for x in range(self.shape[0]):
            for y in range(self.shape[1] + 1 - taille):
                if self.peut_placer(x, y, 1, taille):
                    count += 1
                        
        return count

    def genere_grille(self, *args):
        for others in args:
            if isinstance(others, Iterable):
                for taille in others:
                    self.place_alea(taille)
            else :
                self.place_alea(others)

    def joue(self, x: int, y: int) -> Tuple[bool, Optional[Tuple[int, int, int, int, int]]]:
        boat_id = self.board[x, y]
        #print(f'Boat id : {boat_id}')
        if boat_id != 0:
            self.board[x, y] = 0
            (x, y, d, taille, hp) = self.boats[boat_id - 100]
            hp -= 1
            boat = (x, y, d, taille, hp)
            self.boats[boat_id - 100] = boat
            if hp == 0:
                return (True, boat)
            else:
                return (True, None)
        else:
            return (False, None)
        
    def victoire(self) -> bool:
        return all(hp == 0 for _,_,_,_,hp in self.boats)

    def remove_boats(self):
        while len(self.boats) > 0:
            self.pop_boat()

    def reset(self):
        tailles = self.tailles
        self.remove_boats()
        self.genere_grille(tailles)

    def __repr__(self):
        self.affiche()
        s = "Boats : " + str(self.boats)
        s += "\n\n"
        s += "Board :\n" + str(self.board.T)
        return s

############################ Fonctions sur l'analyse combinatoire ###############################

def trouver_grille_similaire(grille: Bataille) -> int:
    test = Bataille(grille.tailles)
    i = 1
    while test != grille:
        test.reset()
        i += 1
    return i

#renvoie le nombre de position possible sur une grille vide
def nb_pos_grille_vide(taille: int) -> int:
    return (11 - taille) * 20

#renvoie le nombre de grilles possibles avec l'hypothèse d'indépendance des bateaux entre eux (utilisé pour la majoration simple)
def nb_pos_liste_bateaux_independance(liste_taille: List[int]) -> int:
    pos = 1
    for taille in liste_taille:
        nb_pos_boat = nb_pos_grille_vide(taille)
        print(f'Taille : {taille}, nb_positions : {nb_pos_boat}')
        pos *= nb_pos_boat
    return pos


def nb_pos_liste_bateaux_dependance_brute_force(tailles: List[int], logs: bool = False) -> int:
    grid = Bataille(mode="Custom")
    if len(tailles) > 1:
        i = 0
        start = time.time()
        taille: int = tailles[0]
        next_tailles = tailles[1:]
        for x in range(grid.shape[0] + 1 - taille):
            for y in range(grid.shape[1]):
                if grid.placer(x, y, 0, taille):
                    i += nb_pos_liste_bateaux_dependance_brute_force_rec(next_tailles, grid)
                    grid.pop_boat()
        if logs:
            print(f'Time : {time.time() - start:.1f} s')
        return 2 * i

    else:
        return grid.nb_possibilite(tailles[0])
   

def nb_pos_liste_bateaux_dependance_brute_force_rec(tailles: List[int], grid: Bataille) -> int:
    if len(tailles) > 1:
        i = 0
        taille = tailles[0]
        next_tailles = tailles[1:]
        for x in range(grid.shape[0] + 1 - taille):
            for y in range(grid.shape[1]):
                if grid.placer(x, y, 0, taille):
                    i += nb_pos_liste_bateaux_dependance_brute_force_rec(next_tailles, grid)
                    grid.pop_boat()

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1] + 1 - taille):
                if grid.placer(x, y, 1, taille):
                    i += nb_pos_liste_bateaux_dependance_brute_force_rec(next_tailles, grid)
                    grid.pop_boat()
                
        return i
    else:
        return grid.nb_possibilite(tailles[0])

#renvoie la produit des nombres de possibilités à chaque ajout de bateau
def nb_pos_liste_bateaux_avec_dependance_non_lineaire(liste_taille) -> int:
    nb_pos = 1
    test = Bataille(mode="Custom")
    for taille in liste_taille:
        nb_pos *= test.nb_possibilite(taille)
        test.place_alea(taille)
    return nb_pos

#ajoute la liste des nombres de possibilités à chaque ajout de bateau dans le tableau passé en argument.
def nb_pos_liste_bateaux_dependance(liste_taille: List[int], grid: Bataille) -> np.ndarray:
    grid.remove_boats()
    values = np.zeros(len(liste_taille), np.float64)
    for i in range(len(liste_taille)):
        values[i] = grid.nb_possibilite(liste_taille[i])
        grid.place_alea(liste_taille[i])
    return values


def nb_pos_liste_bateaux_avec_dependance_lineaire(nb_iter: int, tailles: List[int], logs: bool = False):
    nb_pos = 1
    grid = Bataille(mode='Custom')
    mean, _ = moyenne(nb_pos_liste_bateaux_dependance, nb_iter, logs, tailles,grid)
    for i in mean:
        nb_pos *= i
    return nb_pos




def nb_pos_liste_bateaux_dependance_brute_force_puis_approx(tailles: List[int], profondeur_brute_force: int, logs: bool = False) -> int:
    grid = Bataille(mode="Custom")
    if len(tailles) > 1:
        i = 0
        start = time.time()
        taille: int = tailles[0]
        next_tailles = tailles[1:]
        for x in range(grid.shape[0] + 1 - taille):
            for y in range(grid.shape[1]):
                if grid.placer(x, y, 0, taille):
                    i += nb_pos_liste_bateaux_dependance_brute_force_puis_approx_rec(next_tailles, profondeur_brute_force - 1, grid)
                    grid.pop_boat()
        if logs:
            print(f'Time : {time.time() - start:.1f} s')
        return 2 * i

    else:
        return grid.nb_possibilite(tailles[0])
   

def nb_pos_liste_bateaux_dependance_brute_force_puis_approx_rec(tailles: List[int], profondeur_brute_force: int, grid: Bataille) -> int:
    if len(tailles) > 1:
        if profondeur_brute_force > 0:
            i = 0
            taille = tailles[0]
            next_tailles = tailles[1:]
            for x in range(grid.shape[0] + 1 - taille):
                for y in range(grid.shape[1]):
                    if grid.placer(x, y, 0, taille):
                        i += nb_pos_liste_bateaux_dependance_brute_force_puis_approx_rec(next_tailles, profondeur_brute_force - 1, grid)
                        grid.pop_boat()

            for x in range(grid.shape[0]):
                for y in range(grid.shape[1] + 1 - taille):
                    if grid.placer(x, y, 1, taille):
                        i += nb_pos_liste_bateaux_dependance_brute_force_puis_approx_rec(next_tailles, profondeur_brute_force - 1, grid)
                        grid.pop_boat()
                    
            return i
        else:
            i = grid.nb_possibilite(tailles[0])
            grid.place_alea(tailles[0])
            for taille in tailles[1:-1]:
                i *= grid.nb_possibilite(taille)
                grid.place_alea(taille)
            i *= grid.nb_possibilite(tailles[-1])
            for _ in tailles[0:-1]:
                grid.pop_boat()
            return i
    else :
        return grid.nb_possibilite(tailles[0])