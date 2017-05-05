FC	= gfortran
FFLAGS	= -O3 -fopenmp -ffast-math  -funroll-loops
OBJS	= wigner3j_sub.o
.SUFFIXES: .f90
.f90.o:
	$(FC) $(FFLAGS) -c $<
default: compute_noisebias compute_chi2_lensingiswpowerspectrum
compute_noisebias: $(OBJS) compute_noisebias.o
	$(FC) $(FFLAGS) -o $@ $(OBJS) $@.o $(LDFLAGS)
compute_chi2_lensingiswpowerspectrum: $(OBJS) compute_chi2_lensingiswpowerspectrum.o
	$(FC) $(FFLAGS) -o $@ $(OBJS) $@.o $(LDFLAGS)
clean:
	-rm -f *.o *~
tidy: clean
	-rm -f compute_noisebias
	-rm -f compute_chi2_lensingiswpowerspectrum
	-rm -f multipole_noisebias.txt
