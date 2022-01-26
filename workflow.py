#pestpp-mou freyberg test
#constraints: none
#decision variables: row, col, and rate of injection well
#optimizing: minimize average drawdown, minimize injection, and maximize streamflow at end of simulation

import os
import shutil
import platform
import string
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import flopy
import pyemu

#this script assumes mf6 and pestpp are in your path


def eval_cum_extraction(d):
    cwd = os.getcwd()
    os.chdir(d)
    df = cum_extraction()
    os.chdir(cwd)
    return df

def cum_extraction():
    import flopy
    def process_lst_file():
        lst = flopy.utils.Mf6ListBudget("freyberg6.lst")
        inc, cum = lst.get_dataframes(start_datetime=None, diff=True)
        cum.columns = cum.columns.map(str.lower)
        cum.index.name = "time"
        cum.to_csv("cum.csv")
        return cum
    process_lst_file()
    cum = pd.read_csv('cum.csv')
    print(cum)
    with open("cum_extraction.csv",'w') as f:
        f.write("cum_extrxn, {0}\n".format(cum.iloc[-1,9]))

def add_injxn_wel():
    args = {}
    with open("injxn_wel.dat",'r') as f:
        for line in f:
            raw = line.strip().split()
            args[raw[0]] = float(raw[1])
    row = max(1,int(args["row"]))
    col = max(9,int(args["col"]))

    #make third year pumping with injection (add an injection well to the well files)
    wel_file_list = [f for f in os.listdir() if "wel_stress_period_data" in f]
    strt_inj = 13
    for f in wel_file_list:
        if int(f.split("_")[-1].split(".")[0]) > strt_inj:
            lines = open(os.path.join(f), 'r').readlines()
            lines[-1] = '1,{0},{1},{2}\n'.format(row,col,args['rate'])
            open(os.path.join(f), 'w').writelines(lines)

    wel_file = os.path.join("freyberg6.wel")
    lines = open(wel_file,'r').readlines()
    with open(wel_file,'w') as f:
        for line in lines:
            if "maxbound" in line.lower():
                line = "   maxbound  7\n"
            f.write(line)

def eval_add_injxn_wel(tmp_ws,row=10,col=10,rate=100,write_tpl=False):
    #make third year pumping with injection (add an injection well to the well files)
    wel_file_list = [f for f in os.listdir() if "wel_stress_period_data" in f]
    strt_inj = 13
    for f in wel_file_list:
        if int(f.split("_")[-1].split(".")[0]) > strt_inj:
            with open(os.path.join(f), 'a') as file:
                file.write('1, 10, 10, 100.00\n')

    with open(os.path.join(tmp_ws,"injxn_wel.dat"),'w') as f:
        f.write("row {0}\n".format(row))
        f.write("col {0}\n".format(col))
        f.write("rate {0}\n".format(rate))
    tpl_file = "injxn_wel.dat.tpl"
    if write_tpl:
        with open(os.path.join(tmp_ws, tpl_file), 'w') as f:
            f.write("ptf ~\n")
            f.write("row ~injxn_row~\n")
            f.write("col ~injxn_col~\n")
            f.write("rate   ~injxn_rate~\n")

    bd = os.getcwd()
    os.chdir(tmp_ws)
    add_injxn_wel()
    os.chdir(bd)
    # if not write_tpl:
    #     run_and_plot_results(test_d)
    return tpl_file

def vary_pumping():
    args = {}
    with open("pumping_rates.dat",'r') as f:
        for line in f:
            raw = line.strip().split()
            args[raw[0]] = float(raw[1])

    #make third year pumping with injection (add an injection well to the well files)
    wel_file_list = [f for f in os.listdir() if "wel_stress_period_data" in f]
    for file in wel_file_list:
        with open(file) as f:
            lines = f.readlines()
            i=0
            new_lines=[]
            for line in lines:
                i+=1
                if i == 7:
                    break
                elif len(line.strip()) > 0:
                    items = line.split(',')
                    items[-1] = args['rate{0}'.format(i)]
                    new_line = ",".join(map(str,items))+'\n'
                    new_lines.append(new_line)
            with open(file, 'w') as f:
                f.writelines(new_lines)

def eval_vary_pumping(tmp_ws,rate1=-100,rate2=-100,rate3=-100,rate4=-100,rate5=-100,rate6=-100,write_tpl=False):

    with open(os.path.join(tmp_ws,"pumping_rates.dat"),'w') as f:
        f.write("rate1 {0}\n".format(rate1))
        f.write("rate2 {0}\n".format(rate2))
        f.write("rate3 {0}\n".format(rate3))
        f.write("rate4 {0}\n".format(rate4))
        f.write("rate5 {0}\n".format(rate5))
        f.write("rate6 {0}\n".format(rate6))
    tpl_file = "pumping_rates.dat.tpl"
    if write_tpl:
        with open(os.path.join(tmp_ws, tpl_file), 'w') as f:
            f.write("ptf ~\n")
            f.write("rate1 ~rate1~\n")
            f.write("rate2 ~rate2~\n")
            f.write("rate3 ~rate3~\n")
            f.write("rate4 ~rate4~\n")
            f.write("rate5 ~rate5~\n")
            f.write("rate6 ~rate6~\n")

    bd = os.getcwd()
    os.chdir(tmp_ws)
    vary_pumping()
    os.chdir(bd)
    # if not write_tpl:
    #     run_and_plot_results(test_d)
    return tpl_file

def eval_drawdown(d):
    cwd = os.getcwd()
    os.chdir(d)
    df = mean_drawdown()
    os.chdir(cwd)
    return df

def mean_drawdown():
    dd = pd.read_csv('drawdown.csv')

    with open("drawdown.csv",'w') as f:
        f.write("dd_obs {0}\n".format(np.mean(dd.iloc[-1,1:].values)))

def eval_streamflow(d):
    cwd = os.getcwd()
    os.chdir(d)
    df = streamflow()
    os.chdir(cwd)
    return df

def streamflow():
    sfr = pd.read_csv('sfr.csv')

    with open("streamflow.csv",'w') as f:
        f.write("sfr_obs, {0}\n".format(sfr.iloc[-1,2]))

def setup_model(org_ws):

    np.random.seed(123456)

    # run surfact
    tmp_ws = org_ws + "_temp"
    if os.path.exists(tmp_ws):
        shutil.rmtree(tmp_ws)
    shutil.copytree(org_ws, tmp_ws)

    #make sure it runs, and load in ss output as new heads IC
    pyemu.os_utils.run("{0}".format(os.path.join("mf6")),cwd=tmp_ws)

    name = 'freyberg.nam'
    sim = flopy.mf6.MFSimulation.load(sim_name='mfsim.nam', sim_ws=tmp_ws)
    m = sim.get_model()
    hd_file = os.path.join(tmp_ws, 'freyberg6_freyberg.hds')
    hds = flopy.utils.HeadFile(hd_file)
    hds = hds.get_alldata()
    hds[hds == 1e+30] = -999.99000
    ic = m.get_package('ic')
    ic.strt = hds[0,0,:,:]
    sim.write_simulation()
    
    #make first sp be equal to average transient recharge
    rch_file_list = [f for f in os.listdir(tmp_ws) if "rch_recharge" in f]
    rch_files = {}
    for f in rch_file_list:
        arr = np.loadtxt(os.path.join(tmp_ws, f)) * 0.05
        np.savetxt(os.path.join(tmp_ws, f), arr, fmt="%15.6E")
        rch_files[int(f.split(".")[1].split("_")[-1])] = arr
    keys = list(rch_files.keys())
    keys.sort()
    # calc the first sp recharge as the mean of the others
    new_first = np.zeros_like(rch_files[keys[0]])
    for key in keys[1:]:
        new_first += rch_files[key]
    new_first /= float(len(rch_files) - 1)
    for f in rch_file_list:
        np.savetxt(os.path.join(tmp_ws, f), new_first, fmt="%15.6E")

    pyemu.os_utils.run("{0}".format(os.path.join("mf6")),cwd=tmp_ws)

    sr = pyemu.helpers.SpatialReference.from_namfile(
        os.path.join(tmp_ws, "freyberg6.nam"),
        delr=m.dis.delr.array, delc=m.dis.delc.array)

    grid_v = pyemu.geostats.ExpVario(contribution=1.0, a=500)
    grid_gs = pyemu.geostats.GeoStruct(variograms=grid_v)

    new_dir = tmp_ws.replace('_temp','_template')

    #build pstfrom
    pf = pyemu.utils.PstFrom(tmp_ws,new_dir,spatial_reference=sr,
                             remove_existing=True,zero_based=False,start_datetime="12-31-2017")

    pf.mod_sys_cmds.append("mf6")

    # pf.add_py_function("workflow.py","add_injxn_wel()",is_pre_cmd=True)

    # add injection dvs
    # tpl_file = eval_add_injxn_wel(new_dir, write_tpl=True)

    pf.add_py_function("workflow.py","vary_pumping()",is_pre_cmd=True)
    tpl_file = eval_vary_pumping(new_dir, write_tpl=True)
    
    # #add obs for drawdown
    # eval_drawdown(new_dir)
    # ins_file = os.path.join(new_dir,"drawdown.csv.ins")
    # with open(ins_file,'w') as f:
    #     f.write("pif  ~\n")
    #     f.write("l1 w !dd_obs!\n")
    # pf.add_observations_from_ins(ins_file,pst_path=".")
    # pf.add_py_function("workflow.py", "mean_drawdown()", is_pre_cmd=False)

    #add obs for cumulative extraction
    eval_cum_extraction(new_dir)
    ins_file = os.path.join(new_dir,"cum_extraction.csv.ins")
    with open(ins_file,'w') as f:
        f.write("pif  ~\n")
        f.write("l1 w !cum_extrxn!\n")
    pf.add_observations_from_ins(ins_file,pst_path=".")
    pf.add_py_function("workflow.py", "cum_extraction()", is_pre_cmd=False)

    #add obs for streamflow
    eval_streamflow(new_dir)
    ins_file = os.path.join(new_dir,"streamflow.csv.ins")
    with open(ins_file,'w') as f:
        f.write("pif  ~\n")
        f.write("l1 w !sfr_obs!\n")
    pf.add_observations_from_ins(ins_file,pst_path=".")
    pf.add_py_function("workflow.py", "streamflow()", is_pre_cmd=False)

    # use the idomain array for masking parameter locations
    try:
        ib = m.dis.idomain[0].array
    except:
        ib = m.dis.idomain[0]

    # define a dict that contains file name tags and lower/upper bound information
    tags = {"npf_k_": [0.2, 5.], "npf_k33_": [.2, 5], "sto_ss": [.5, 2], "sto_sy": [.8, 1.2],
            "rch_recharge": [.5, 1.5]}
    dts = pd.to_datetime("12-31-2017") + \
          pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]), unit="d")

    # loop over each tag, bound info pair
    for tag, bnd in tags.items():
        lb, ub = bnd[0], bnd[1]
        # find all array based files that have the tag in the name
        arr_files = [f for f in os.listdir(new_dir) if tag in f and f.endswith(".txt")]

        if len(arr_files) == 0:
            print("warning: no array files found for ", tag)
            continue

        # make sure each array file in nrow X ncol dimensions (not wrapped)
        for arr_file in arr_files:
            arr = np.loadtxt(os.path.join(new_dir, arr_file)).reshape(ib.shape)
            np.savetxt(os.path.join(new_dir, arr_file), arr, fmt="%15.6E")

        for arr_file in arr_files:
            pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base=arr_file.split('.')[1] + "_gr",
                              pargp=arr_file.split('.')[1] + "_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                              geostruct=grid_gs)

    # # get all the list-type files associated with the wel package
    # list_files = [f for f in os.listdir(org_ws) if "freyberg6.wel_stress_period_data_" in f and f.endswith(".txt")]
    # # for each wel-package list-type file
    # for list_file in list_files:
    #     kper = int(list_file.split(".")[1].split('_')[-1]) - 1
    #     print(dts[kper])
    #     # add spatially constant, but temporally correlated parameter
    #     pf.add_parameters(filenames=list_file, par_type="constant", par_name_base="twel_mlt_{0}".format(kper),
    #                       pargp="twel_mlt".format(kper), index_cols=[0, 1, 2], use_cols=[3],
    #                       upper_bound=5, lower_bound=0.2, datetime=dts[kper])

    pst = pf.build_pst('freyberg.pst')


    df = pst.add_parameters(os.path.join(new_dir, tpl_file), pst_path=".")
    with open(os.path.join(new_dir, "risk.dat.tpl"), 'w') as f:
        f.write("ptf ~\n")
        f.write("_risk_ ~   _risk_    ~\n")
    pst.add_parameters(os.path.join(new_dir, "risk.dat.tpl"), pst_path=".")

    par = pst.parameter_data
    par.loc["_risk_","pargp"] = "dv_pars"
    par.loc["_risk_","parlbnd"] = 0.001
    par.loc["_risk_", "parubnd"] = 0.99
    # start at lower bound for cases without risk obj
    par.loc["_risk_", "parval1"] = 0.001
    par.loc["_risk_", "partrans"] = "none"

    par.loc[df.parnme,"pargp"] = "dv_pars"
    par.loc["rate1","parval1"] = -100
    par.loc["rate1", "parlbnd"] = -500
    par.loc["rate1", "parubnd"] = 0
    par.loc["rate1", "partrans"] = "none"

    par.loc[df.parnme,"pargp"] = "dv_pars"
    par.loc["rate2","parval1"] = -100
    par.loc["rate2", "parlbnd"] = -500
    par.loc["rate2", "parubnd"] = 0
    par.loc["rate2", "partrans"] = "none"

    par.loc[df.parnme,"pargp"] = "dv_pars"
    par.loc["rate3","parval1"] = -100
    par.loc["rate3", "parlbnd"] = -500
    par.loc["rate3", "parubnd"] = 0
    par.loc["rate3", "partrans"] = "none"

    par.loc[df.parnme,"pargp"] = "dv_pars"
    par.loc["rate4","parval1"] = -100
    par.loc["rate4", "parlbnd"] = -500
    par.loc["rate4", "parubnd"] = 0
    par.loc["rate4", "partrans"] = "none"

    par.loc[df.parnme,"pargp"] = "dv_pars"
    par.loc["rate5","parval1"] = -100
    par.loc["rate5", "parlbnd"] = -500
    par.loc["rate5", "parubnd"] = 0
    par.loc["rate5", "partrans"] = "none"

    par.loc[df.parnme,"pargp"] = "dv_pars"
    par.loc["rate6","parval1"] = -100
    par.loc["rate6", "parlbnd"] = -500
    par.loc["rate6", "parubnd"] = 0
    par.loc["rate6", "partrans"] = "none"

    # par.loc["injxn_col", "parval1"] = 10
    # par.loc["injxn_col", "parubnd"] = 17
    # par.loc["injxn_col", "parlbnd"] = 9
    # par.loc["injxn_col", "partrans"] = "none"
    #
    # #dont let this go to zero
    # par.loc["injxn_rate", "parval1"] = 100.
    # par.loc["injxn_rate", "parubnd"] = 1000.0
    # par.loc["injxn_rate", "parlbnd"] = 0.001
    # par.loc["injxn_rate", "partrans"] = "none"

    pst.try_parse_name_metadata()
    obs = pst.observation_data
    # obs.loc["dd_obs", "obgnme"] = "less_than"
    obs.loc["sfr_obs", "obgnme"] = "greater_than"
    obs.loc["cum_extrxn", "obgnme"] = "less_than"


    pst.control_data.noptmax = 0

    # wpar = par.loc[par.pargp=="twel_mlt","parnme"]
    # print(wpar)
    # # wpar = wel_par.loc[wel_par.parval1<0,"parnme"]
    # par.loc[wpar, "partrans"] = "none"
    # par.loc[wpar, "pargp"] = "dv_pars"
    # par.loc[wpar, "parubnd"] = 0.0
    # par.loc[wpar, "parlbnd"] = -500.0
    # this one is the objective
    # pst.add_pi_equation(wpar.to_list(),pilbl="pump_rate",obs_group="less_than")
    #pst.add_pi_equation([stage_par], obs_group="greater_than", pilbl=stage_par)
    #pst.add_pi_equation(["ar_width"],obs_group="less_than",pilbl="ar_width")
    # pst.add_pi_equation(["injxn_rate"], obs_group="less_than",pilbl="injxn_rate")
    # pst.add_pi_equation(["drawdown"], obs_group="less_than",pilbl="drawdown")
    # pst.add_pi_equation(["strmflw"], obs_group="greater_than", pilbl="strmflw")
    pst.add_pi_equation(["_risk_"], obs_group="greater_than",pilbl="_risk_")
    pst.pestpp_options["mou_objectives"] = ["cum_extrxn","sfr_obs"]
    pst.pestpp_options["opt_dec_var_groups"] = "dv_pars"
    pst.pestpp_options["panther_echo"] = True
    pst.pestpp_options["mou_risk_objective"] = True
    pst.pestpp_options["mou_generator"] = "de"
    pst.pestpp_options["mou_population_size"] = 100

    pst.write(os.path.join(new_dir, "freyberg.pst"))
    pe = pf.draw(100, use_specsim=True)
    pe.to_binary(os.path.join(new_dir, "prior.jcb"))
    pe.to_csv('prior.csv')

    pyemu.os_utils.run("{0} freyberg.pst".format(os.path.join("pestpp-mou")), cwd=new_dir)

def run_mou(risk_obj=False,chance_points="single",risk=0.5,stack_size=100,
            num_workers=6,pop_size=100,tag="",recalc_every=100000,noptmax=100):
    t_d = os.path.join("simple_freyberg_template")
    pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
    pst.pestpp_options["opt_par_stack"] = "prior.jcb"
    pst.pestpp_options["mou_risk_objective"] = risk_obj
    pst.pestpp_options["opt_recalc_chance_every"] = recalc_every
    pst.pestpp_options["opt_chance_points"] = chance_points
    pst.pestpp_options["opt_risk"] = risk
    pst.pestpp_options["mou_population_size"] = pop_size
    pst.pestpp_options["opt_stack_size"] = stack_size
    if risk == 0.5:
        objs = pst.pestpp_options["mou_objectives"].split(",")
        if "_risk_" in objs:
            objs.remove("_risk_")
        pst.pestpp_options["mou_objectives"] = objs
    else:
        objs = pst.pestpp_options["mou_objectives"].split(",")
        if "_risk_" not in objs:
            objs.append("_risk_")
        pst.pestpp_options["mou_objectives"] = objs
    pst.control_data.noptmax = noptmax
    pst.write(os.path.join(t_d,"freyberg.pst"))

    m_d = os.path.join("..","..","..","Desktop","simple_freyberg_master")
    w_d = os.path.join("..","..","..","Desktop")
    if len(tag) > 0:
        m_d += "_" + tag
    pyemu.os_utils.start_workers(t_d,"pestpp-mou","freyberg.pst",
                                 num_workers=num_workers,master_dir=m_d,
                                 verbose=True, worker_root=w_d)

if __name__ == "__main__":
    # setup_model('simple_freyberg')
    run_mou()