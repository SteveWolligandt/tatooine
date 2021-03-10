import pandas
import matplotlib.pyplot as plt

fw_integration_test = pandas.read_csv('autonomous_particles_doublegyre_forward.csv', sep=',', decimal=".")
bw_integration_test = pandas.read_csv('autonomous_particles_doublegyre_backward.csv', sep=',', decimal=".")

tau_1 = bw_integration_test[bw_integration_test["tau"] == 1]
tau_3 = bw_integration_test[bw_integration_test["tau"] == 3]
tau_5 = bw_integration_test[bw_integration_test["tau"] == 5]

tau_1__0_1    = tau_1[tau_1["radius"] == 0.1]
ax = tau_1__0_1    = tau_1__0_1.rename(columns={"error":"tau=1; r=1e-1"})
tau_1__0_01   = tau_1[tau_1["radius"] == 0.01]
tau_1__0_01   = tau_1__0_01.rename(columns={"error":"tau=1; r=1e-2"})
tau_1__0_001  = tau_1[tau_1["radius"] == 0.001]
tau_1__0_001  = tau_1__0_001.rename(columns={"error":"tau=1; r=1e-3"})
tau_1__0_0001 = tau_1[tau_1["radius"] == 0.0001]
tau_1__0_0001    = tau_1__0_0001.rename(columns={"error":"tau=1; r=1e-4"})

tau_3__0_1    = tau_3[tau_3["radius"] == 0.1]
tau_3__0_1    = tau_3__0_1.rename(columns={"error":"tau=3; r=1e-1"})
tau_3__0_01   = tau_3[tau_3["radius"] == 0.01]
tau_3__0_01   = tau_3__0_01.rename(columns={"error":"tau=3; r=1e-2"})
tau_3__0_001  = tau_3[tau_3["radius"] == 0.001]
tau_3__0_001  = tau_3__0_001.rename(columns={"error":"tau=3; r=1e-3"})
tau_3__0_0001 = tau_3[tau_3["radius"] == 0.0001]
tau_3__0_0001 = tau_3__0_0001.rename(columns={"error":"tau=3; r=1e-4"})

tau_5__0_1    = tau_5[tau_5["radius"] == 0.1]
tau_5__0_1    = tau_5__0_1.rename(columns={"error":"tau=5; r=1e-1"})
tau_5__0_01   = tau_5[tau_5["radius"] == 0.01]
tau_5__0_01   = tau_5__0_01.rename(columns={"error":"tau=5; r=1e-2"})
tau_5__0_001  = tau_5[tau_5["radius"] == 0.001]
tau_5__0_001  = tau_5__0_001.rename(columns={"error":"tau=5; r=1e-3"})
tau_5__0_0001 = tau_5[tau_5["radius"] == 0.0001]
tau_5__0_0001 = tau_5__0_0001.rename(columns={"error":"tau=5; r=1e-4"})

radii = pandas.concat(
        [tau_1__0_1["tau=1; r=1e-1"],
         tau_3__0_1["tau=3; r=1e-1"],
         tau_5__0_1["tau=5; r=1e-1"],
         tau_1__0_01["tau=1; r=1e-2"],
         tau_3__0_01["tau=3; r=1e-2"],
         tau_5__0_01["tau=5; r=1e-2"],
         tau_1__0_001["tau=1; r=1e-3"],
         tau_3__0_001["tau=3; r=1e-3"],
         tau_5__0_001["tau=5; r=1e-3"],
         tau_1__0_0001["tau=1; r=1e-4"],
         tau_3__0_0001["tau=3; r=1e-4"],
         tau_5__0_0001["tau=5; r=1e-4"]],
        axis=1)
radii.plot.box()
plt.show()

taus = pandas.concat(
        [tau_1__0_1["tau=1; r=1e-1"],
         tau_1__0_01["tau=1; r=1e-2"],
         tau_1__0_001["tau=1; r=1e-3"],
         tau_1__0_0001["tau=1; r=1e-4"],
         tau_3__0_1["tau=3; r=1e-1"],
         tau_3__0_01["tau=3; r=1e-2"],
         tau_3__0_001["tau=3; r=1e-3"],
         tau_3__0_0001["tau=3; r=1e-4"],
         tau_5__0_1["tau=5; r=1e-1"],
         tau_5__0_01["tau=5; r=1e-2"],
         tau_5__0_001["tau=5; r=1e-3"],
         tau_5__0_0001["tau=5; r=1e-4"]],
        axis=1)
taus.plot.box()
plt.show()
