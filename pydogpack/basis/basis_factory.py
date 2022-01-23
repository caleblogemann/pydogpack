from pydogpack.basis import basis
from pydogpack.utils import errors

import yaml

# basis_factory
# try to avoid circular dependencies
# solution need to create basis and basis needs to create solutions
# solution uses basis_factory and basis uses solution


def from_dict(dict_):
    basis_class = dict_[basis.CLASS_KEY]
    if basis_class == basis.NODAL_STR:
        return basis.NodalBasis1D.from_dict(dict_)
    elif basis_class == basis.GAUSS_LOBATTO_STR:
        return basis.GaussLobattoNodalBasis1D.from_dict(dict_)
    elif basis_class == basis.GAUSS_LEGENDRE_STR:
        return basis.GaussLegendreNodalBasis1D.from_dict(dict_)
    elif basis_class == basis.LEGENDRE_STR:
        return basis.LegendreBasis1D.from_dict(dict_)
    elif basis_class == basis.FINITE_VOLUME_BASIS_STR:
        return basis.FiniteVolumeBasis1D.from_dict(dict_)
    elif basis_class == basis.LEGENDRE_2D_CARTESIAN_STR:
        return basis.LegendreBasis2DCartesian.from_dict(dict_)
    elif basis_class == basis.MODAL_2D_TRIANGLE_STR:
        return basis.ModalBasis2DTriangle.from_dict(dict_)
    else:
        raise Exception("Basis Class: " + basis_class + " is not a valid option")


def from_file(filename):
    with open(filename, "r") as file:
        dict_ = yaml.safe_load(file)
        return from_dict(dict_)


def from_dogpack_params(params):
    basis_type = params["basis_type"]
    num_dims = params["num_dims"]
    mesh_type = params["mesh_type"]
    space_order = params["space_order"]
    if basis_type == "space-legendre":
        if num_dims == 1:
            return basis.LegendreBasis1D(space_order)
        elif num_dims == 2:
            if mesh_type == "cartesian":
                return basis.LegendreBasis2DCartesian(space_order)
            elif mesh_type == "unstructured":
                return basis.ModalBasis2DTriangle(space_order)
            else:
                raise errors.InvalidParameter("mesh_type", mesh_type)
        else:
            raise errors.InvalidParameter("num_dims", num_dims)
    else:
        raise errors.InvalidParameter("basis_type", basis_type)
