from MJOLNIR import TasUBlibDEG
from MJOLNIR.Data.Sample import Sample
from pathlib import Path
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR import _tools
import os
import numpy as np
import MJOLNIR.TasUBlibDEG as TasUBlib
from MJOLNIR.Data.DataReader import DataReader
import pytest

# The old tests read camea2018 files that are not available (question 37 of the refactoring
# log): tests that work with any file read the 2024 sample file, the others build their
# sample from a cell and two reflections
SAMPLE_DATA = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not SAMPLE_DATA, reason='sampledata/camea2024*.hdf not available')

def hhl_sample():
    """Cubic sample (a = 5 Å) in the (H,H,0)-(0,0,L) plane, as camea2018n000178.hdf"""
    return DataReader.sample_from_reflections([5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                                              np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))



def test_sample_exceptions():
    try: # No parameters given
        s1 = Sample(a=None)
        assert False
    except:
        assert True

    try: # negative parameters given
        s1 = Sample(a=-1,b=1,c=1)
        assert False
    except:
        assert True

    try: # negative parameters given
        s1 = Sample(a=1,b=-1,c=1)
        assert False
    except:
        assert True
    try: # negative parameters given
        s1 = Sample(a=1,b=1,c=-1)
        assert False
    except:
        assert True

    try: # negative parameters given
        s1 = Sample(a=1,b=1,c=1,alpha=200)
        assert False
    except:
        assert True
    try: # negative parameters given
        s1 = Sample(a=1,b=1,c=1,beta=-10)
        assert False
    except:
        assert True
    try: # negative parameters given
        s1 = Sample(a=1,b=1,c=1,gamma=-10)
        assert False
    except:
        assert True

@needs_data
def test_Sample_conversions():
    sample = CameaDataReader(SAMPLE_DATA[0]).sample

    # Check that tr and inv_tr results in the same
    p0,p1 = 2.25,-1.23
    Qx,Qy = sample.tr(p0,p1)
    P0,P1 = sample.inv_tr(Qx,Qy)

    assert(np.all(np.isclose([p0,p1],[P0,P1])))

    # Check that format_coord prints something
    string = sample.format_coord(Qx,Qy) # format is h = ??, k = ??, l = ??
    hkl = [float(x.split('=')[-1]) for x in string.split(',')]
    print(string)
    print(hkl)

    QxQyFromSample = sample.calculateHKLToQxQy(*hkl)
    print(QxQyFromSample)
    assert(np.all(np.isclose(QxQyFromSample,[Qx,Qy],atol=1e-3)))


def test_parameters():
    s1 = Sample(1,2,3,90,90,120)
    pars = np.array([getattr(s1,x) for x in ['a','b','c','alpha','beta','gamma']])

    assert(np.all(np.isclose(pars,np.array([1,2,3,90,90,120]))))

def test_equality():
    s1 = Sample(1,2,3,90,90,120)
    s2 = Sample(1,2,3,90,90,120)
    assert(s1==s2)

def test_calculateProjections(): # TODO: Update test

    s1 = Sample(np.pi*2,np.pi*2,np.pi*2,90,90,60)

    s1.calculateProjections()

    theta = 2*np.pi/3 # 120 degrees between projection vectors
    AStar = np.linalg.norm(s1.reciprocalVectorA)
    BStar = np.linalg.norm(s1.reciprocalVectorB)
    CStar = np.linalg.norm(s1.reciprocalVectorC)
    ## Projection is given by [[a*,cos(theta)b*],[0,sin(tetha)b*]]
    #assert(np.all(np.isclose(s1.projectionMatrix,np.array([[AStar*1,BStar*np.cos(theta)],[0,BStar*np.sin(theta)]]))))
    #assert(np.all(np.isclose(s1.tr(1,1),np.array([AStar+np.cos(theta)*BStar,np.sin(theta)*BStar]))))

    #point = s1.tr(3.5,7.2)
    #reverse = s1.inv_tr(point[0],point[1])
    #assert(np.all(np.isclose(reverse,np.array([3.5,7.2]))))

    #string = s1.format_coord(point[0],point[1])
    #assert(string=='h = 3.500, k = 7.200, l = 0.000')

@needs_data
def test_DataFile_Sample_UB():
    df = CameaDataReader(SAMPLE_DATA[0])
    s = df.sample
    b1,b2,b3 = [np.linalg.norm(x) for x in [s.reciprocalVectorA,s.reciprocalVectorB,s.reciprocalVectorC]]
    a1,a2,a3 = [np.linalg.norm(x) for x in [s.realVectorA,s.realVectorB,s.realVectorC]]
    beta1,beta2,beta3 = _tools.vectorAngle(s.reciprocalVectorB,s.reciprocalVectorC),_tools.vectorAngle(s.reciprocalVectorC,s.reciprocalVectorA),_tools.vectorAngle(s.reciprocalVectorA,s.reciprocalVectorB)
    alpha1,alpha2,alpha3 = _tools.vectorAngle(s.realVectorB,s.realVectorC),_tools.vectorAngle(s.realVectorC,s.realVectorA),_tools.vectorAngle(s.realVectorA,s.realVectorB)

    cell = s.cell#[a1,a2,a3,b1,b2,b3,alpha1,alpha2,alpha3,beta1,beta2,beta3]
    r1 = s.plane_vector1
    r2 = s.plane_vector2

    UBCalc = TasUBlib.calcTasUBFromTwoReflections(cell,r1,r2)
    comparison = np.isclose(UBCalc,s.orientationMatrix,atol=1e-5) # Assert that they are equal to 1e-5 (Numerical error)
    print(np.round(UBCalc,5))
    print(np.round(s.orientationMatrix,5))
    assert(np.all(comparison))


@needs_data
def test_DataFile_Sample_Projection():
    df = CameaDataReader(SAMPLE_DATA[0]) # In A-B plane
    assert(np.all(np.isclose(df.sample.projectionVector1,np.array([1.0,0.0,0.0]))))
    assert(np.all(np.isclose(df.sample.projectionVector2,np.array([0.0,1.0,0.0]))))

    sample = hhl_sample() # in 1 1 0 and 0 0 1 plane
    assert(np.all(np.isclose(sample.projectionVector1,np.array([0.0,0.0,1.0]))))
    assert(np.all(np.isclose(sample.projectionVector2,np.array([1.0,1.0,0.0]))))



def test_Sample_CurratAxe():
    """Spurion positions checked against the scattering geometry rather than numbers from a
    2018 file: a spurion sits where the instrument measures with the nominal ki and kf at the
    angles of an elastic Bragg reflection with |k| = k(Ef) (monochromator, the sample sees
    Ef) or k(Ei) (analyser, the analyser passes Ei)"""
    sample = hhl_sample()

    Ei = [4,7,9]
    Ef = np.array([[4,4,4],[5,5,5]])
    Bragg = [[0,0,1],[1,1,1]]

    k = lambda E: np.sqrt(E)*TasUBlib.factorsqrtEK
    QBragg = 2*np.pi/5.0*np.linalg.norm(Bragg, axis=1) # Cubic, a = 5 Å

    def expected_length(k_bragg, ki, kf, Q):
        A4 = 2*np.arcsin(Q/(2*k_bragg)) # Scattering angle of the elastic reflection
        return np.sqrt(ki**2 + kf**2 - 2*ki*kf*np.cos(A4))

    for spurionType, bragg_energy in [('Monochromator', 'Ef'), ('ANAlYser', 'Ei')]:
        pos = sample.CurratAxe(Ei,Ef,Bragg,spurionType=spurionType)
        assert(pos.shape == (len(Bragg),np.array(Ei).size,Ef.size,3))
        assert(np.all(np.isclose(pos[:,:,:,2],0.0))) # All Qz are to be zero
        for b, Q in enumerate(QBragg):
            for i, ei in enumerate(Ei):
                for f, ef in enumerate(Ef.flatten()):
                    k_bragg = k(ef) if bragg_energy == 'Ef' else k(ei)
                    assert np.isclose(np.linalg.norm(pos[b,i,f,:2]),
                                      expected_length(k_bragg, k(ei), k(ef), Q)), (spurionType, b, i, f)
                    if ei == ef: # Elastic: the spurion is the Bragg peak itself
                        assert np.allclose(pos[b,i,f,:2], sample.calculateHKLToQxQy(*Bragg[b]))

    posMonoProjection = sample.CurratAxe(Ei,Ef,Bragg,spurionType='monoChrOmaTOR',Projection=True)
    posMonoHKL = sample.CurratAxe(Ei,Ef,Bragg,spurionType='monoChrOmaTOR',HKL=True)
    assert(posMonoProjection.shape == (len(Bragg),np.array(Ei).size,Ef.size,2))
    assert(posMonoHKL.shape == (len(Bragg),np.array(Ei).size,Ef.size,3))

    try:
        sample.CurratAxe(Ei,Ef,Bragg,spurionType='WRONG!!')
        assert False
    except AttributeError:
        assert True
