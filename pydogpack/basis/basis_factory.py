from pydogpack.basis import basis

import yaml

# basis_factory allow modules to create basis without importing basis module
# try to avoid circular dependencies

NODAL_STR = "nodal"
GAUSS_LOBATTO_STR = "gauss_lobatto"
GAUSS_LEGENDRE_STR = "gauss_legendre"
LEGENDRE_STR = "legendre"
FVBASIS_STR = "finite_volume"
CLASS_KEY = "basis_class"


def from_dict(dict_):
    basis_class = dict_[CLASS_KEY]
    if basis_class == NODAL_STR:
        return basis.NodalBasis.from_dict(dict_)
    elif basis_class == GAUSS_LOBATTO_STR:
        return basis.GaussLobattoNodalBasis.from_dict(dict_)
    elif basis_class == GAUSS_LEGENDRE_STR:
        return basis.GaussLegendreNodalBasis.from_dict(dict_)
    elif basis_class == LEGENDRE_STR:
        return basis.LegendreBasis.from_dict(dict_)
    elif basis_class == FVBASIS_STR:
        return basis.FVBasis.from_dict(dict_)
    else:
        raise Exception("Basis Class: " + basis_class + " is not a valid option")


def from_file(filename):
    with open(filename, "r") as file:
        dict_ = yaml.safe_load(file)
        return from_dict(dict_)
