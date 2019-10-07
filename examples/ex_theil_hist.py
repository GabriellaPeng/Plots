

Theil = {type_sim: {i: np.zeros([len(npoly)]) for i in ['Um', 'Us', 'Uc', 'mse']} for type_sim in
                    proc_sim}

for type_sim, sim_val in proc_sim.items():
    for p in range(len(npoly)):
       Theil[type_sim]['mse'][p],Theil[type_sim]['Um'][p],Theil[type_sim]['Us'][p], \
       Theil[type_sim]['Uc'][p] = theil_inequal(sim_val[:, p], obs_norm[vr][:, p])