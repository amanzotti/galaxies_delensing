*** Full-sky, Cosmic-variance only chi-squared for the lensing-temperature power spectrum ***
April 25, 2012: E.Komatsu
Ref: Lewis, Challinor & Hanson, JCAP 03, 018 (2011), arXiv:1101.2234

Here we provide a program for computing the full-sky, cosmic-variance only chi-squared of the lensing-ISW power spectrum, including variance from the lensing signal itself.

Chi-squared is computed as

Chi^2 = sum_l [Q(l)]^2/{C(l)*(N(l)+Ctheta(l))+[Q(l)]^2}

where
- C(l) is the lensed CMB temperature power spectrum
- Q(l) is the lensing-temperature power spectrum, which includes all the linear effects (not just the late-time ISW)
- Ctheta(l) is the power spectrum of lensing potential

all of which are computed from CAMB, and

- N(l) is the noisebias of the reconstructed lensing potential (computed by "compute_noise" in this package)

The lensed CMB temperature power spectrum is contained in "wmap5baosn_max_likelihood_lensedCls.dat". The lensing-temperature power spectrum and the lensing potential power spectrum are contained in "wmap5baosn_max_likelihood_lenspotentialCls.dat". Each column gives:

(1st) multipole
(2nd) unlensed TT
(3rd) unlensed EE
(4th) unlensed BB
(5th) unlensed TE
(6th) lensing potential power spectrum, [l(l+1)]^2 C^theta_l/twopi [unitless]
(7th) lensing-temperature power spectrum, [l(l+1)]^1.5 Q(l)/twopi [uK]
(8th) lensing-E polarization power spectrum,  [l(l+1)]^1.5 <E*Theta>/twopi [uK]

These data were generated for the maximum likelihood parameters given in Table I of Komatsu et al.(2008) [WMAP 5-year interpretation paper] with "WMAP5+BAO+SN". The input file for CAMB is also provided (wmap5baosn_max_likelihood_params.ini).

- To compile and use the  program, edit Makefile and simply "make"
- It will generate executables called "compute_noisebias" and "compute_chi2_lensingiswpowerspectrum"
- Run "./compute_noisebias", which will generate a data file "multipole_noisebias.txt", which contains:
(1st colummn): multipole
(2nd column): N(l) [unitless]
- Run "./compute_chi2_lensingiswpowerspectrum", which will give you a value of chi^2. No files will be generated.

For your convenience, the noisebias data calculated up to lmax=1500, 2000, and 3000 using the WMAP5+BAO+SN parameters are included as:
- results/multipole_noisebias_lmax=1500.txt
- results/multipole_noisebias_lmax=2000.txt
- results/multipole_noisebias_lmax=3000.txt

The corresponding chi^2 values are recorded in "results/results.txt"
