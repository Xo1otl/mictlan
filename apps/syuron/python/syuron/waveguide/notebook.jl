using WGMODES
using Plots

lambda = 1.55
dx = dy = 0.1
x = y = -10:dx:10
index = [sqrt(x^2 + y^2) < 4 ? 1.46 : 1.45 for x = x, y = y]
epsilon = index .^ 2
guess = maximum(index)
nmodes = 3
boundary = "0000"
field = "scalar"

println("Calculating waveguide modes...")

phi, neff = WGMODES.svmodes(lambda, guess, nmodes, dx, dy, epsilon, boundary, field);

println("plotting...")

Plots.plot(
    Plots.heatmap(x, y, index,
        title="Waveguide Index Structure",
        aspectratio=1,
    ),
    Plots.heatmap(x, y, phi[:, :, 1],
        title="Waveguide Mode",
        aspectratio=1,
    ),
    Plots.heatmap(x, y, phi[:, :, 2],
        title="Waveguide Mode",
        aspectratio=1,
    ),
    Plots.heatmap(x, y, phi[:, :, 3],
        title="Waveguide Mode",
        aspectratio=1,
    ),
    size=(800, 800),
)
