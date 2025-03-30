"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: fe.py
       consolidation script to run all components of fe simulation
"""

import os
import argparse
import _meshGen
import _meshData
import _fxBig
import _fx
import _plotStress

def main(n, a, r, b, p, depth):
    depth += 1
    print("\t" * depth + "!! MESHING !!") 
    _meshGen.msh_(n, r, b, depth)
    print("\t" * depth + "!! MESH DATA GEN !!") 
    _meshData.data_(n, b, depth)
    if b:
        print("\t" * depth + "!! BEGIN FE on BIG: " + emf + " !!")
        _fxBig.fx_(n, r, depth)
    else:
        print("\t" * depth + "!! BEGIN FE !!")
        if a == 0:
            emfs = [n]
        elif a == 1:
            emfs = [x for x in [str(y) for y in range(0, 36, 1)]]
        s, f = [], []
        for emf in emfs:
            print("\t" * depth + " ~> Test: {}".format(emf))
            file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_msh/EMGEO_" + str(r) + ".msh")
            try:
                _fx.fx_(emf, file, r, p, s, depth)
                s.append(emf)
            except:
                f.append(emf)
                continue
        print("\t" * depth + " ~> Pass: {}".format(s))
        print("\t" * depth + " ~> Fail: {}".format(f))
    print("\t" * depth + "!! STRESS-STRAIN PLOT GEN !!") 
    _plotStress.plot_(n, r, s, depth)

if __name__ == '__main__':
    depth = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--test_num",type=str)
    parser.add_argument("-a", "--all_test",type=int)
    parser.add_argument("-r", "--ref_level",type=int)
    parser.add_argument("-b", "--big_test",type=int)
    parser.add_argument("-p", "--def_level",type=int)
    parser.add_argument("-s", "--stretch_test",type=int)
    args = parser.parse_args()
    n = args.test_num
    a = args.all_test
    r = args.ref_level
    b = args.big_test
    p = args.def_level
    s = args.stretch_test
    main(n, a, r, b, p, s, depth)