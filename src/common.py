"""Units: geometric lengths in km, pressure/energy density in km^-2."""

G_CGS = 6.67430e-8
C_CGS = 2.99792458e10
MEV_FM3_TO_CGS = 1.602176634e33
MEV_FM3_TO_KM2 = MEV_FM3_TO_CGS * G_CGS / C_CGS**4 * 1e10
SOLAR_MASS_KM = 1.47664  # Same convention as PNJLs/tov_geo.jl.
