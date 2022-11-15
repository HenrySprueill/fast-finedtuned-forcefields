from ase import data
import networkx as nx
import numpy as np


def infer_water_cluster_bonds(atoms):
    """
    Infers the covalent and hydrogen bonds between oxygen and hydrogen atoms in a water cluster.
    Definition of a hydrogen bond obtained from https://aip.scitation.org/doi/10.1063/1.2742385
    Args:
        atoms (ase.Atoms): ASE atoms structure of the water cluster. Atoms list must be ordered
            such that the two covalently bound hydrogens directly follow their oxygen.
    Returns:
        cov_bonds ([(str, str, 'covalent')]): List of all covalent bonds
        h_bonds [(str, str, 'hydrogen')]: List of all hydrogen bonds
    """

    # Make sure the atoms are in the right order
    z = atoms.get_atomic_numbers()
    assert z[:3].tolist() == [8, 1, 1], "Atom list not in (O, H, H) format"
    coords = atoms.positions

    # Get the covalent bonds
    #  Note: Assumes that each O is followed by 2 covalently-bonded H atoms
    cov_bonds = [(i, i + 1, 'covalent') for i in range(0, len(atoms), 3)]
    cov_bonds.extend([(i, i + 2, 'covalent') for i in range(0, len(atoms), 3)])

    # Get the hydrogen bonds
    #  Start by getting the normal to each water molecule
    q_1_2 = []
    for i in range(0, len(atoms), 3):
        h1 = coords[i + 1, :]
        h2 = coords[i + 2, :]
        o = coords[i, :]
        q_1_2.append([h1 - o, h2 - o])
    v_list = [np.cross(q1, q2) for (q1, q2) in q_1_2]

    #  Determine which (O, H) pairs are bonded
    h_bonds = []
    for idx, v in enumerate(v_list):  # Loop over each water molecule
        for index, both_roh in enumerate(q_1_2):  # Loop over each hydrogen
            for h_index, roh in enumerate(both_roh):
                # Get the index of the H and O atoms being bonded
                indexO = 3 * idx
                indexH = 3 * index + h_index + 1

                # Get the coordinates of the two atoms
                h_hbond = coords[indexH, :]
                o_hbond = coords[indexO, :]

                # Compute whether they are bonded
                dist = np.linalg.norm(h_hbond - o_hbond)
                if (dist > 1) & (dist < 2.8):
                    angle = np.arccos(np.dot(roh, v) / (np.linalg.norm(roh) * np.linalg.norm(v))) * (180.0 / np.pi)
                    if angle > 90.0:
                        angle = 180.0 - angle
                    N = np.exp(-np.linalg.norm(dist) / 0.343) * (7.1 - (0.05 * angle) + (0.00021 * (angle ** 2)))
                    if N >= 0.0085:
                        h_bonds.append((indexO, indexH, 'hydrogen'))

    return cov_bonds, h_bonds


def create_graph(atoms):
    """
    Given a ASE atoms object, this function returns a graph structure with following properties.
        1) Each graph has two graph-level attributes: actual_energy and predicted_energy
        2) Each node represents an atom and has two attributes: label ('O'/'H' for oxygen and hydrogen) and 3-dimensional
           coordinates.
        3) Each edge represents a bond between two atoms and has two attributes: label (covalent or hydrogen) and distance.
    Args:
        atoms (Atoms): ASE atoms object
    Returns:
        (nx.Graph) Networkx representation of the water cluster
    """

    # Compute the bonds
    cov_bonds, h_bonds = infer_water_cluster_bonds(atoms)

    # Add nodes to the graph
    graph = nx.Graph()
    for i, (coord, Z) in enumerate(zip(atoms.positions, atoms.get_atomic_numbers())):
        graph.add_node(i, label=data.chemical_symbols[Z], coords=coord)

    # Add the edges
    edges = cov_bonds + h_bonds
    for a1, a2, btype in edges:
        distance = np.linalg.norm(atoms.positions[a1, :] - atoms.positions[a2, :])
        graph.add_edge(a1, a2, label=btype, weight=distance)
    return graph


def coarsen_graph(in_graph: nx.Graph) -> nx.DiGraph:
    """Create a graph with only one node per water molecule
    Args:
        in_graph: Input graph, which contains both hydrogens and oxygen
    Returns:
         A directed graph with only the oxygen atoms as nodes. Nodes are identified
         as whether that O atoms is part of a water molecule that donates a hydrogen
         bond to another molecule or whether it receives a hydrogen bond
    """

    # Initialize the output graph
    output = nx.DiGraph()

    # Collect information from the previous graph
    bonds = []
    for node, node_data in in_graph.nodes(data=True):
        if node_data['label'] == 'O':
            # If it is an oxygen, make it a node in the new graph
            output.add_node(node // 3, **node_data)  # Make the count from [0, N)
        elif node_data['label'] == 'H':
            # Check if this H participates in H bonding
            donor = acceptor = None  # Stores the donor and acceptor oxygen id
            for neigh in in_graph.neighbors(node):
                neigh_info = in_graph.get_edge_data(node, neigh)
                bond_type = neigh_info['label']
                if bond_type == 'covalent':
                    donor = neigh // 3, neigh_info['weight']
                else:
                    acceptor = neigh // 3, neigh_info['weight']
            if not (donor is None or acceptor is None):
                bonds.append((donor, acceptor))  # Store as donor->acceptor
        else:
            raise ValueError(f'Unrecognized type: {node_data["label"]}')

    # Assign bonds to each water molecule
    for (d, w_d), (a, w_a) in bonds:
        # Add the edges
        output.add_edge(d, a, label='donate', weight=w_d+w_a)
        #output.add_edge(a, d, label='accept', weight=w_d+w_a)

    return output


def get_structure_metrics(cluster):
    """
    Params:
        cluster: ASE Atoms object
    Returns:
        OH covalent bonds (A)
        HOH bond angles (deg)
        OH_hbond_dists (A)
        HOH_hbond_angles (deg)
        OHO_hbond_angles (deg)   # https://pubs.acs.org/doi/10.1021/acs.biochem.8b00217 FIG 9
        OO_hbond_dist (A)        # https://pubs.acs.org/doi/10.1021/acs.biochem.8b00217 FIG 9
    """
    labels = ['OH_cov', 'HOH_cov', 'OH_hbond', 'HOH_hbond', 'OHO_hbond', 'OO_hbond']

    """
    # get covalent bond distances and angles
    HOH_covalent_angles = ana.get_values(ana.get_angles('H', 'O', 'H', unique=True))
    """

    # convert to H-bonding graph
    G = create_graph(cluster)


    # get bond type (covalent or hydrogen) for each edge 
    bond_type=nx.get_edge_attributes(G,'label')

    # sort covalent and hydrogen bonds
    OH_cov_pairs = [k for k,v in bond_type.items() if v=='covalent']
    OH_hbond_pairs = [k for k,v in bond_type.items() if v=='hydrogen']

    # get OH_cov bond lengths
    OH_covalent_bonds = [G.edges[b]['weight'] for b in OH_cov_pairs]

    # get HOH_cov triplets
    HOH_covalent_triplets=np.reshape(OH_cov_pairs, (-1,4))
    HOH_covalent_triplets=HOH_covalent_triplets[:,1:]

    # get HOH_cov angles
    HOH_covalent_angles = [cluster.get_angle(a,b,c) for a,b,c in HOH_covalent_triplets]


    # stop if no H-bonds are found
    if len(OH_hbond_pairs) == 0:
        print('fully disconnected structure found')
        values = [OH_covalent_bonds[0], HOH_covalent_angles[0]]

        type_list=[]
        value_list=[]
        for i in range(len(values)):
            value_list.append(values[i])
            type_list.append([labels[i]]*len(values[i]))
        return pd.DataFrame({'type':flatten(type_list), 'value':flatten(value_list)})


    # compute H---O distances
    OH_hbond_dists = [G.edges[b]['weight'] for b in OH_hbond_pairs]

    # get atomic numbers of OH hbond pairs
    OH_hbond_Z = [itemgetter(i,j)(cluster.get_atomic_numbers()) for i,j in OH_hbond_pairs]

    O_id = np.where(np.stack(OH_hbond_Z)==8)[1]
    O_nodes = [pair[O_id[i]] for i,pair in enumerate(OH_hbond_pairs)]

    # match those O node indices with their H in OH_conv_pairs to get ther Hs
    HOH_hbond_angles=[]
    for i,O in enumerate(O_nodes):
        bonded_H = list(set(flatten(itemgetter(np.where(np.stack(OH_cov_pairs)==O)[0])(np.stack(OH_cov_pairs)))))
        bonded_H.remove(O)
        angles = []
        for k in bonded_H:

            try:
                # compute H---O-Hs angles for both H
                angles.append(cluster.get_angle(OH_hbond_pairs[i][0],OH_hbond_pairs[i][1],k))
            except:
                pass
        if len(angles)>0:
            # keep angle closest to 180deg
            angles = [a if a > 90 else 180-a for a in angles]
            HOH_hbond_angles+=angles

    H_id = np.where(np.stack(OH_hbond_Z)==1)[1]
    H_nodes = [pair[H_id[i]] for i,pair in enumerate(OH_hbond_pairs)]

    OHO_hbond_angles=[]
    OO_hbond_dist=[]
    for i,H in enumerate(H_nodes):
        bonded_O = list(set(flatten(itemgetter(np.where(np.stack(OH_cov_pairs)==H)[0])(np.stack(OH_cov_pairs)))))
        bonded_O.remove(H)
        try:
            # compute O-H---O angle
            angle = cluster.get_angle(OH_hbond_pairs[i][0],OH_hbond_pairs[i][1],bonded_O[0])
            # convert angle closest to 180deg
            angle = angle if angle > 90 else 180-angle
            OHO_hbond_angles.append(angle)

            Hbonded_O = list(OH_hbond_pairs[i])
            Hbonded_O.remove(H)
            OO_hbond_dist.append(cluster.get_distance(Hbonded_O[0],bonded_O[0]))
        except:
            pass

    values = [OH_covalent_bonds, HOH_covalent_angles, OH_hbond_dists, HOH_hbond_angles, OHO_hbond_angles, OO_hbond_dist]

    type_list=[]
    value_list=[]
    for i in range(len(labels)):
        value_list.append(values[i])
        type_list.append([labels[i]]*len(values[i]))

    # return df
    return pd.DataFrame({'type':flatten(type_list), 'value':flatten(value_list)})
