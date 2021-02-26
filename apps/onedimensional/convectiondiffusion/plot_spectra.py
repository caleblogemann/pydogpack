from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from apps.onedimensional.thinfilm import thin_film

import numpy as np
import yaml
import matplotlib.pyplot as plt
import pdb

if __name__ == "__main__":
    with open("spectra.yml", "r") as file:
        dict_ = yaml.safe_load(file)

    bc = boundary.Periodic()
    basis_class = basis.LegendreBasis
    for num_basis_cpts in range(1, 4):
        data = dict_[str(bc)][basis_class.string][num_basis_cpts]
        plot_data = dict()
        for num_elems in data.keys():
            real = data[num_elems]["real"]
            imag = data[num_elems]["imag"]
            if (np.max(real) > 0.0):
                print(np.max(real))
            eigvals = np.array(real) + 1j * np.array(imag)
            max_eigval = np.max(np.abs(eigvals))
            plot_data[num_elems] = max_eigval

        plt.plot(list(plot_data.keys()), list(plot_data.values()), 'o')
    plt.show()
