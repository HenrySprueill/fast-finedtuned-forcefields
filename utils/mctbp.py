import sys
import numpy as np
import math
import pickle
import networkx as nx
import ase
from ase.optimize.sciopt import SciPyFminBFGS
from ase.io import read
from ase.db import connect
from ase.io.trajectory import Trajectory

#from utils import ase_to_graph, coarsen_graph, return_graph_data, calculate_similarity
sys.path.insert(0, '/qfs/projects/ecp_exalearn/designs/finetune_comparison/utils')
import water_cluster_moves as mcm
from model_to_ase import SchnetCalculator

sys.path.insert(0, '/people/herm379/exalearn/IPU_trained_models/pnnl_sandbox/schnet')
from model import SchNet


def run_sim(model, num_waters, steps, savedir, db_path, fmax=0.005, min_tol=1.8, max_disp=1, temp=0.8, use_eff_temp=True, use_dsi=True):
    """
    Run TBP-MC simulation
    args: num_waters (int): size of water cluster, steps (int): number of MC moves to make,
          savedir (str): path to directory to save output files, min_tol (float): minimum distance
          between atoms after MC move to make it an "acceptable" structure, 
    returns: None
    """
    energy_list = []
    db_end = connect(f'{savedir}/w{num_waters}_end_structures_mctbp.db')
    db_start = connect(f'{savedir}/w{num_waters}_start_structures_mctbp.db')
    db_traj = connect(f'{savedir}/w{num_waters}_opt_trajs_mctbp.db')
    with connect(f'{db_path}/W{num_waters}_geoms_all.db') as conn: #Get starting structure
        ind = np.random.randint(0, high=len(conn))
        row = conn.get(id=ind+1)
        mol = row.toatoms()
        cluster = mol.get_positions()
        opt_cluster, curr_e = optimize_schnet(num_waters, cluster.copy(), model, savedir, fmax=fmax)
        energy_list.append(curr_e)
        new_cluster = opt_cluster.copy()
        for i in range(steps):
            accept = False
            while accept == False:  #Get "acceptable" MC move
                mc_cluster, move_type = get_move(new_cluster.copy(), max_disp, num_waters)
                accept = check_atom_overlap(mc_cluster, min_tol=min_tol)
            
            opt_new_cluster, new_e = optimize_schnet(num_waters, mc_cluster.copy(), model, savedir, fmax=fmax)
        
            if use_eff_temp == True:
                try:
                    accept_move = move_acceptance(curr_e, new_e, new_cluster, opt_new_cluster, eff_temp, use_dsi=use_dsi)
                except UnboundLocalError:
                    accept_move = move_acceptance(curr_e, new_e, new_cluster, opt_new_cluster, temp, use_dsi=use_dsi)
                bin_count = get_histogram_count(energy_list)
                eff_temp = calc_eff_temp(temp, bin_count)
            else:
                accept_move = move_acceptance(curr_e, new_e, temp)

            if accept_move == True:
                accept_structure = check_optimized_structure(opt_new_cluster)
                if accept_structure == True:
                    energy_list.append(new_e)
                    new_cluster = opt_new_cluster
                    curr_e = new_e

            traj = Trajectory(f'{savedir}/w{num_waters}_test.traj', 'r')
            nnpes_energy = [a.get_potential_energy() for a in traj]
            nnpes_grad = [a.get_forces() for a in traj] #gradients store in data['forces']
            nnpes_structure = [a.get_positions() for a in traj]

            for j in range(len(nnpes_energy)):
                mol = ase.Atoms('OHH'*num_waters, nnpes_structure[j])
                db_traj.write(atoms=mol, data={'energy': nnpes_energy[j]*23.06054195, 'forces': nnpes_grad[j]*23.06054195})

            mol = ase.Atoms('OHH'*num_waters, nnpes_structure[0])
            db_start.write(atoms=mol, data={'energy': nnpes_energy[0]*23.06054195, 'forces': nnpes_grad[0]*23.06054195})

            mol = ase.Atoms('OHH'*num_waters, nnpes_structure[-1])
            db_end.write(atoms=mol, data={'energy': nnpes_energy[-1]*23.06054195, 'forces': nnpes_grad[-1]*23.06054195, 'steps': len(nnpes_energy), 'accept': accept_move})

    return None

def generate_mctrajs(model, atoms, savedir, fmax=0.005, min_tol=1.8, max_disp=1):
    """
    Make random MC move on atoms and optimize (record energies and gradients)
    args: num_waters (int): size of water cluster, steps (int): number of MC moves to make,
          savedir (str): path to directory to save output files, min_tol (float): minimum distance
          between atoms after MC move to make it an "acceptable" structure,
    returns: None
    """
    db_traj = connect(f'{savedir}/opt_trajs_mctbp.db')
    cluster = atoms.get_positions()
    num_waters = int(np.shape(cluster)[0]/3)

    opt_cluster, curr_e = optimize_schnet(num_waters, cluster.copy(), model, savedir, fmax=fmax)
    new_cluster = opt_cluster.copy()

    accept = False
    while accept == False:  #Get "acceptable" MC move
        mc_cluster, move_type = get_move(new_cluster.copy(), max_disp, num_waters)
        accept = check_atom_overlap(mc_cluster, min_tol=min_tol)

    opt_new_cluster, new_e = optimize_schnet(num_waters, mc_cluster.copy(), model, savedir, fmax=fmax)
    
    #traj = Trajectory(f'{savedir}/w{num_waters}_test.traj', 'r')
    #nnpes_energy = [a.get_potential_energy() for a in traj]
    #nnpes_grad = [a.get_forces() for a in traj] #gradients store in data['forces']
    #nnpes_structure = [a.get_positions() for a in traj]

    #for j in range(len(nnpes_energy)):
    #    mol = ase.Atoms('OHH'*num_waters, nnpes_structure[j])
    #    db_traj.write(atoms=mol, data={'energy': nnpes_energy[j]*23.06054195, 'forces': nnpes_grad[j]*23.06054195})

    # Need to multiply 'energy' and 'forces' by 23.06054195
    return [x for x in Trajectory(f'{savedir}/w{num_waters}_test.traj', mode='r')]


def optimize_schnet(num_waters, coords, best_model, savedir, fmax=0.001, mult=5):
    """
    Optimize new structure using NN PES and return optimized structure and energy
    args: number of water molecules (int), x,y,z coordinates (array), pt file of best trained model,
          optimization convergence (float)
    return: matrix containing coordinates optimized by NN PES, energy at minimum (eV)
    """
    atom_string = 'OHH'*num_waters
    atoms = ase.Atoms(atom_string, positions=coords)
    calc = SchnetCalculator(best_model, atoms)
    atoms.calc = calc
    dyn = SciPyFminBFGS(atoms, trajectory=f'{savedir}/w{num_waters}_test.traj', logfile=f'{savedir}/w{num_waters}_test.log')
    try:
        dyn.run(fmax=fmax, steps=100000000)
    except:
        try:
            dyn.run(fmax=fmax*mult, steps=100000000)
        except:
            try:
                dyn.run(fmax=fmax*mult*mult, steps=100000000)
            except:
                try:
                    dyn.run(fmax=fmax*mult*mult*mult, steps=100000000)
                except:
                    pass
    relaxed_energy = calc.get_potential_energy()
    
    return atoms.get_positions(), relaxed_energy

def get_move(new_cluster, max_disp, num_waters):
    """
    For each molecule, either:
    1. Translate in x-, y-, z-coords
    2. Rotate about H-O-H bisector
    3. Rotate about O-H bond
    4. Rotate about O-O axis (with any other molecule)
    """
    move_type = np.random.randint(0,4)
    if move_type == 0:   #Translate single molecule by (x,y,z)
        num_trans = np.random.randint(1, int(num_waters/3))
        ind_trans = np.random.randint(0, num_waters, size=num_trans)
        for i in range(len(ind_trans)):
            rand_molecule = ind_trans[i]
            x_val = np.random.uniform(0,max_disp)
            y_val = np.random.uniform(0,max_disp)
            z_val = np.random.uniform(0,max_disp)
            new_cluster[rand_molecule*3:rand_molecule*3+3, :] += np.array([x_val, y_val, z_val])
    elif move_type == 1:   #rotate about H-O-H bisector
        num_rots = np.random.randint(1, int(num_waters)/3)
        ind_rots = np.random.randint(0, num_waters, size=num_rots)
        for i in range(len(ind_rots)):
            mol_rot = ind_rots[i]
            theta_val = np.random.uniform(0, 2*np.pi)
            new_cluster[mol_rot*3:(mol_rot*3+3), :] = mcm.rotate_around_HOH_bisector_axis(new_cluster[mol_rot*3:(mol_rot*3+3), :], theta_val)
        pass
    elif move_type == 2:   #rotate about O-H bond
        ind_rot = np.random.randint(0, num_waters) #choose molecule to rotate
        theta_val = np.random.uniform(0,2*np.pi)
        monomer_geom = new_cluster[ind_rot*3:ind_rot*3+3, :]
        which_h = np.random.randint(0,2)  #choose which hydrogen
        new_cluster[ind_rot*3:ind_rot*3+3, :] = mcm.rotate_around_local_axis(monomer_geom, 0, which_h+1, theta_val)
    else:      #rotate about O-O axis
        ind_rots = np.random.randint(0, num_waters, size=2) #choose two oxygens to rotate about
        theta_val = np.random.uniform(0,2*np.pi)
        dimer_geom = np.concatenate((new_cluster[ind_rots[0]*3:(ind_rots[0]*3+3), :], new_cluster[ind_rots[1]*3:(ind_rots[1]*3+3), :]), axis=0)
        rotate_dimer = mcm.rotate_around_local_axis(dimer_geom, 0, 3, theta_val)
        new_cluster[ind_rots[0]*3:(ind_rots[0]*3+3), :] = rotate_dimer[0:3, :]
        new_cluster[ind_rots[1]*3:(ind_rots[1]*3+3), :] = rotate_dimer[3:, :]
        pass
    
    return new_cluster, move_type

def check_optimized_structure(opt_cluster):
    """
    Determine whether any atoms in the new configuration are too close to one another
    args: xyz-coordinates of system after making a move
    return: whether the new configuration is allowed (or if atoms are overlapping)
    """
    #Check intramolecular geometry (angle: 80-135, bond: 0.82-1.27)
    accept = True
    cluster = opt_cluster.copy()
    hoh_ang = np.zeros(int(np.size(cluster, axis=0)/3))
    oh_dist = np.zeros(int(np.size(cluster, axis=0)/3)*2)
    for i in range(int(np.size(cluster, axis=0)/3)):
        oh_vec1 = cluster[3*i, :]-cluster[3*i+1, :]
        oh_vec2 = cluster[3*i, :]-cluster[3*i+2, :]
        oh_dist[2*i] = np.linalg.norm(oh_vec1)
        oh_dist[2*i+1] = np.linalg.norm(oh_vec2)
        hoh_ang[i] = np.arccos(np.dot(oh_vec1, oh_vec2)/(oh_dist[2*i]*oh_dist[2*i+1]))
    hoh_ang *= (180/np.pi)
    if (0.82 > oh_dist).any() or (oh_dist > 1.27).any() or (80 > hoh_ang).any() or (hoh_ang > 135).any():
        accept = False
    
    return accept

def check_atom_overlap(new_config, min_tol=1.0):
    """
    Determine whether any atoms in the new configuration are too close to one another
    args: xyz-coordinates of system after making a move
    return: whether the new configuration is allowed (or if atoms are overlapping)
    """
    cluster = new_config.copy()
    dist_matrix = np.zeros((np.size(cluster, axis=0), np.size(cluster, axis=0)))
    for i in range(np.size(cluster, axis=0)):
        atom1 = cluster[i, :]
        diff = cluster - atom1
        dist_matrix[:, i] = np.sqrt(np.sum(np.square(diff), axis=1))
        dist_matrix[i, i] = 10
        if (i+1) % 3 == 0:
            dist_matrix[i-1, i] = 10
            dist_matrix[i-2, i] = 10
            dist_matrix[i-2, i-1] = 10
            dist_matrix[i, i-1] = 10
            dist_matrix[i, i-2] = 10
            dist_matrix[i-1, i-2] = 10
    if np.min(dist_matrix) >= min_tol:
        accept = True
    else:
        accept = False
        
    return accept

def move_acceptance(prev_e, curr_e, prev_struct=None, curr_struct=None, temp=0.8, use_dsi=True, dsi_threshold=0.5, norm_const=0.5):
    """
    Calculates whether a move was accepted based on the change in energy and the temp
    args: energy before move (float), energy after move (float), structures (np.array), temperature (float)
    return: whether the move is accepted (bool)
    """
    if curr_e <= prev_e:
        accept = True
    else:
        if use_dsi == True:
            dsi = get_dsi(prev_struct, curr_struct)
            if dsi <= dsi_threshold:
                accept = False
            else:
                coeff = math.exp(min(1, -(curr_e - prev_e - norm_const*dsi)/temp)) #missing k_b
                rand_num = np.random.uniform(0,1)
                if rand_num <= coeff:
                    accept = True
                else:
                    accept = False    
        else:
            coeff = math.exp(min(1, -(curr_e - prev_e)/temp)) #missing k_b
            rand_num = np.random.uniform(0,1)
            if rand_num <= coeff:
                accept = True
            else:
                accept = False
    return accept

def coords_to_xyz(coords_array):
    """
    Prints array of coordinates in standard xyz format (for visualization)
    args: numpy array of coordinates
    return: None
    """
    for i in range(int(np.size(coords_array, axis=0)/3)):
        print(f'O  {coords_array[i*3, 0]}  {coords_array[i*3, 1]}  {coords_array[i*3, 2]}')
        print(f'H  {coords_array[i*3+1, 0]}  {coords_array[i*3+1, 1]}  {coords_array[i*3+1, 2]}')
        print(f'H  {coords_array[i*3+2, 0]}  {coords_array[i*3+2, 1]}  {coords_array[i*3+2, 2]}')
        
    return None

def calc_eff_temp(t_init, bin_count, norm_coeff=0.015):
    """
    Calculate the "effective" temperature using initial temp and histogram of energies sampled
    args: initial temp (float), bin_count (int)
    return: effective temperature (float)
    """
    temp_eff = t_init + np.exp(norm_coeff*bin_count)
    return temp_eff

def get_histogram_count(energy_list, bin_width=0.5):
    """
    args: list of energies, width of bin
    return: bin count for that energy
    """
    bins = np.arange(min(energy_list), max(energy_list)+2*bin_width, bin_width)
    counts, bin_edges = np.histogram(energy_list, bins=bins)

    for i in range(len(bins)-1):
        if bins[i] <= energy_list[-1] < bins[i+1]:
            bin_idx = i

    return counts[bin_idx]

def get_dsi(prev_structure, curr_structure):
    """
    Get Dis-Similarity Index (DSI) for previous structure and current structure
    args: previously sampled structure and current structure (both numpy arrays)
    return: DSI (float)
    """
    prev_o = prev_structure[0:np.size(prev_structure, axis=0):3, :]
    curr_o = curr_structure[0:np.size(curr_structure, axis=0):3, :]
    
    nW = np.size(prev_o, axis=0)
    num_dists = int(nW*(nW-1)/2)
    
    pairwise_dist_prev = np.zeros(num_dists)
    pairwise_dist_curr = np.zeros(num_dists)
    
    counter = 0
    for i in range(0, nW-1):
        for j in range(i+1, nW):
            pairwise_dist_prev[counter] = np.linalg.norm(prev_o[i, :] - prev_o[j, :])
            pairwise_dist_curr[counter] = np.linalg.norm(curr_o[i, :] - curr_o[j, :])
            counter += 1
    
    dsi = np.linalg.norm(np.sort(pairwise_dist_prev)-np.sort(pairwise_dist_curr))
    
    return dsi

def xyz_to_numpy(filename, cluster_size):
    coords = np.zeros((cluster_size*3, 3))
    with open(filename, 'r') as f:
        count = 0
        for line in f:
            if line.startswith('O '):
                atom = line.strip().split()
                coords[count, 0] = float(atom[1])
                coords[count, 1] = float(atom[2])
                coords[count, 2] = float(atom[3])
                count += 1
            elif line.startswith('H '):
                atom = line.strip().split()
                coords[count, 0] = float(atom[1])
                coords[count, 1] = float(atom[2])
                coords[count, 2] = float(atom[3])
                count += 1
            else:
                pass
    return coords



