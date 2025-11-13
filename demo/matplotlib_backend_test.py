
import matplotlib
matplotlib.use("qtagg")
# matplotlib.use("tkagg")
print(matplotlib.get_backend())

import matplotlib.rcsetup as rcsetup

print(rcsetup.all_backends)

import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.title("Docker GUI Test (non-root user)")
plt.show()