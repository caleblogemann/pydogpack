from pydogpack.basis import basis

import yaml

# basis_factory allow modules to create basis without importing basis module
# try to avoid circular dependencies


def from_dict(dict_):
    basis_class = dict_[basis.CLASS_KEY]
    if basis_class == basis.NODAL_STR:
        return basis.NodalBasis.from_dict(dict_)
    elif basis_class == basis.GAUSS_LOBATTO_STR:
        return basis.GaussLobattoNodalBasis.from_dict(dict_)
    elif basis_class == basis.GAUSS_LEGENDRE_STR:
        return basis.GaussLegendreNodalBasis.from_dict(dict_)
    elif basis_class == basis.LEGENDRE_STR:
        return basis.LegendreBasis.from_dict(dict_)
    elif basis_class == basis.FVBASIS_STR:
        return basis.FVBasis.from_dict(dict_)
    else:
        raise Exception("Basis Class: " + basis_class + " is not a valid option")


def from_file(filename):
    with open(filename, "r") as file:
        dict_ = yaml.safe_load(file)
        return from_dict(dict_)
