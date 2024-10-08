module Waveguide

wavelength = 1.55

dx = 0.02
dy = 0.02

xaxis = -2:dx:2
yaxis = -1:dy:1

indexdist = [
    if y < 0
        1.5
    elseif y < 0.2 || (y < 0.5 && abs(x) < 1)
        3.45
    else
        1.0
    end for y in yaxis, x in xaxis
]

println("indexdist = ", indexdist)

end