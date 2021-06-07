def grab_int(linestring):
    linelist = linestring.split()
    return int(linelist[0])


def grab_float(linestring):
    linelist = linestring.split()
    return float(linelist[0])


def grab_string(linestring):
    linelist = linestring.split()
    return linelist[0]


def parse_grid_section(config, ini_params):
    """Parse the grid section of the parameters file."""

    if (
        ini_params["mesh_type"] == "cartesian"
        or ini_params["mesh_type"] == "curvilinear"
    ):
        ini_params["mx"] = grab_int(config["mesh"]["mx"])
        ini_params["xlow"] = grab_float(config["mesh"]["xlow"])
        ini_params["xhigh"] = grab_float(config["mesh"]["xhigh"])

        if ini_params["num_dims"] > 1:
            ini_params["my"] = grab_int(config["mesh"]["my"])
            ini_params["ylow"] = grab_float(config["mesh"]["ylow"])
            ini_params["yhigh"] = grab_float(config["mesh"]["yhigh"])

            if ini_params["num_dims"] > 2:
                ini_params["mz"] = grab_int(config["mesh"]["mz"])
                ini_params["zlow"] = grab_float(config["mesh"]["zlow"])
                ini_params["zhigh"] = grab_float(config["mesh"]["zhigh"])


def parse_ini_parameters(output_directory, parameters_file, ini_params={}):
    """Parse a parameters.ini file.

    This function parses an input file parameters.ini and returns a dictionary
    containing key-value pairs for each required object in a DoGPack
    parameters.ini file.

    TODO - test for errors on checking these files - read in from the default
    parameters file if that object is missing.

    Input
    -----

        parameters_file - a string pointing to a valid parameters.ini file.

    Returns
    -------

        ini_params - a dictionary containing key-value pairs.  If ini_params already
        exists as a dictionary, this routine will add new keys.
    """

    import configparser

    config = configparser.ConfigParser()
    config.read(parameters_file)
    num_sections = len(config.sections())

    ini_params["num_dims"] = grab_int(config["dogParams"]["num_dims"])
    if ini_params["num_dims"] == 1:
        ini_params["mesh_type"] = "cartesian"
    else:
        ini_params["mesh_type"] = grab_string(config["dogParams"]["mesh_type"])
    ini_params["number_of_output_frames"] = grab_int(
        config["dogParams"]["number_of_output_frames"]
    )
    ini_params["final_time"] = grab_float(config["dogParams"]["final_time"])
    ini_params["dt_initial"] = grab_float(config["dogParams"]["dt_initial"])
    ini_params["dt_max"] = grab_float(config["dogParams"]["dt_max"])
    ini_params["max_number_of_time_steps"] = grab_int(
        config["dogParams"]["max_number_of_time_steps"]
    )
    ini_params["time_stepping_method"] = grab_string(
        config["dogParams"]["time_stepping_method"]
    )
    ini_params["limiter_method"] = grab_string(config["dogParams"]["limiter_method"])
    ini_params["basis_type"] = grab_string(config["dogParams"]["basis_type"])
    ini_params["space_order"] = grab_int(config["dogParams"]["space_order"])
    ini_params["time_order"] = grab_int(config["dogParams"]["time_order"])
    ini_params["use_limiter"] = grab_int(config["dogParams"]["use_limiter"])
    ini_params["verbosity"] = grab_int(config["dogParams"]["verbosity"])
    ini_params["source_term"] = grab_int(config["dogParams"]["source_term"])
    ini_params["num_eqns"] = grab_int(config["dogParams"]["num_eqns"])
    ini_params["restart_frame"] = grab_int(config["dogParams"]["restart_frame"])
    ini_params["datafmt"] = grab_int(config["dogParams"]["datafmt"])

    extra_config = configparser.ConfigParser()
    extra_config.read(output_directory + "/cfl.ini")
    cfl_max_str = grab_string(config["dogParams"]["cfl_max"])
    cfl_target_str = grab_string(config["dogParams"]["cfl_target"])
    if cfl_max_str == "auto":
        ini_params["cfl_max"] = grab_float(extra_config["dogParams"]["cfl_max"])
    else:
        ini_params["cfl_max"] = grab_float(config["dogParams"]["cfl_max"])
    if cfl_target_str == "auto":
        ini_params["cfl_target"] = grab_float(extra_config["dogParams"]["cfl_target"])
    else:
        ini_params["cfl_target"] = grab_float(config["dogParams"]["cfl_target"])

    # Quadrature order for the initial conditions.  (Default setting in the
    # code (See: DogParams.cpp) is to use the space_order if this is not specified)
    ini_params["ic_quad_order"] = config.get(
        "dogParams", "ic_quad_order", fallback=ini_params["space_order"]
    )

    if config.sections()[num_sections - 1] == "reset":
        ini_params["new_num_eqns"] = grab_int(config["reset"]["new_num_eqns"])
    else:
        ini_params["new_num_eqns"] = ini_params["num_eqns"]

    if config.has_section("mesh"):
        parse_grid_section(config, ini_params)

    import os
    import sys

    current_dir = os.path.abspath("./")
    local_lib_dir = os.path.abspath("../lib/")

    sys.path.append(current_dir)

    file_parse_app_lib = local_lib_dir + "/parse_ini_parameters_app_lib.py"
    file_parse_app_example = current_dir + "/parse_ini_parameters_app_example.py"

    check_parse_app_lib = os.path.exists(file_parse_app_lib)
    check_parse_app_example = os.path.exists(file_parse_app_example)

    if check_parse_app_lib:
        from parse_ini_parameters_app_lib import parse_ini_parameters_app_lib

        parse_ini_parameters_app_lib(config, ini_params)

    if check_parse_app_example:
        from parse_ini_parameters_app_example import parse_ini_parameters_app_example

        parse_ini_parameters_app_example(config, ini_params)

    return ini_params
