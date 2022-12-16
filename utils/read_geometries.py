#!/usr/bin/python

import numpy as np

def read_geoms(geom):
    '''
    Reads a file containing a large number of XYZ formatted files concatenated together and splits them
    into an array of arrays of vectors (MxNx3) where M is the number of geometries, N is the number of atoms,
    and 3 is from each x, y, z coordinate.
    '''
    allCoords = []
    atomLabels = []
    header = []
    with open(geom) as ifile:
        while True:
            atomLabels__ = []
            line = ifile.readline()
            if line.strip().isdigit():
                natoms = int(line)
                title = ifile.readline()
                header.append(str(natoms) + '\n' + title)
                coords = np.zeros([natoms, 3], dtype="float64")
                for x in coords:
                    line = ifile.readline().split()
                    atomLabels__.append(line[0])
                    x[:] = list(map(float, line[1:4]))
                allCoords.append(coords)
                atomLabels.append(atomLabels__)
            if not line:
                break
    return header, atomLabels, allCoords

def write_geoms(header, labels, coords, ofile=None):
    """
    args: header, labels, and coords as output by read_geoms
    return: no return

    Writes the the molecules to stdout in xyz format if no ofile is specified.
    Otherwise, write the geometries to ofile.
    """
    for i, head in enumerate(header):
        output = head
        for j, vec in enumerate(coords[i]):
            output += (labels[i][j] + " " +
                      np.array2string(np.array(vec), precision=14, separator=' ', suppress_small=True).strip('[]') + '\n')
        if ofile is None:
            print(output)
        elif i == 0:
            with open(ofile, 'w') as f:
                f.write(output)
        else:
            with open(ofile, 'a') as f:
                f.write(output)
