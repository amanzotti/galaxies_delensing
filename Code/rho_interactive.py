'''
This take the power spectra cross with CMB lensign and auto of different surveys ans spits out their rho as in Shwerwin Smithful
'''

import numpy as np
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import ConfigParser
from bokeh.io import curdoc, vplot, output_notebook
from bokeh.layouts import row, widgetbox, column
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Slider, TextInput, RadioButtonGroup, Button
from bokeh.plotting import figure, output_file, show, save
from bokeh.models.widgets import CheckboxButtonGroup

cosmosis_dir = '/home/manzotti/cosmosis/'
inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'
n_slices = 2


Config_ini = ConfigParser.ConfigParser()
values = ConfigParser.ConfigParser()
Config_ini.read(inifile)
values_file = Config_ini.get('pipeline', 'values')
output_dir = Config_ini.get('test', 'save_dir')

values.read(cosmosis_dir + values_file)


lbins = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/ells__delens.txt')


ckk = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clk_delens.txt')

# noise_cl = np.loadtxt(
#     './compute_chi2_lensingiswpowerspectrum/multipole_noisebias.txt')
# # check the first ell is the same
# assert (ells_cmb[0] == noise_cl[0, 0])

# noise_cl[:, 1] = noise_cl[:, 1] * noise_cl[:, 0] ** 4 / 4.  # because the power C_kk is l^4/4 C_phi
# noise_fun = interp1d(noise_cl[:, 0], noise_cl[:, 1])
ckk_noise = np.zeros_like(ckk)
# ckk_noise = noise_fun(lbins)

# initialize
rho_comb = np.zeros((np.size(lbins)))
rho_cib_des = np.zeros((np.size(lbins)))
rho_des = np.zeros((np.size(lbins)))
rho_cib = np.zeros((np.size(lbins)))


labels = ['cib', 'des', 'ska10', 'ska5', 'ska1', 'ska01', 'lsst', 'euclid']

cl_cross_k = {}
cl_auto = {}
rho = {}
for label in labels:
  cl_cross_k[label] = np.loadtxt(cosmosis_dir + output_dir +
                                 '/limber_spectra/cl' + label + 'k_delens.txt')
  cl_auto[label] = np.loadtxt(cosmosis_dir + output_dir +
                              '/limber_spectra/cl' + label + label + '_delens.txt')
  rho[label] = cl_cross_k[label] / np.sqrt(ckk[:] * cl_auto[label])


# single survey save
for label in labels:
  np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_{}.txt'.format(label),
             np.vstack((lbins, rho[label])).T)


# Multiple surveys

labels = ['des', 'cib']

cgk = np.zeros((len(labels) + 1, np.size(lbins)))
cgg = np.zeros((len(labels) + 1, len(labels) + 1, np.size(lbins)))

for i in np.arange(0, n_slices):
  cgk[i, :] = np.loadtxt(cosmosis_dir + output_dir +
                         '/limber_spectra/cl' + labels[i] + 'k_delens.txt')

  for j in np.arange(i, n_slices):
    cgg[i, j, :] = np.loadtxt(cosmosis_dir + output_dir +
                              '/limber_spectra/cl' + labels[i] + labels[j] + '_delens.txt')
    cgg[j, i, :] = cgg[i, j, :]

output_file("./lines.html", title="line plot example")
# output_notebook()
radio_button_group = RadioButtonGroup(
    labels=["CIB", "DES"], active=0)



p = figure(title="Tracer corralation", x_axis_label=r'$\ell$', y_axis_label=r'$\rho$', plot_width=800, plot_height=400)
p.title.text = 'init'
source = ColumnDataSource(data=dict(x=lbins, y=rho['cib'], cib = rho['cib'], des = rho['des']))
r = p.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
p.xaxis.bounds = (0, 2000)
p.yaxis.bounds = (0, 1)

def update_data(attrname, old, new):

    # Get the current slider values
    print('in here')
    print(old,new)
    print(new==1)
    p.title.text = "active " +radio_button_group.active
    # Generate the new curve
    if new ==1:
        y = rho['des']
    else:
        y = rho['cib']
    r.data_source.data['y'] = y

callback = CustomJS(args=dict(source=source), code="""
        var data = source.get('data');
        data['x'] = data['x'];
        data['y'] = data['des'];
        source.trigger('change');
    """)

callback2 = CustomJS(args=dict(source=source), code="""
        var data = source.get('data');
        data['x'] = data['x'];
        data['y'] = data['cib'];
        source.trigger('change');
    """)

toggle1 = Button(label="CIB", callback=callback2, name='cib')
toggle2 = Button(label="DES", callback=callback, name='des')

layout = column(toggle1,toggle2, p)
save(layout)
# # add cmb lensing

# cgk[-1, :] = ckk
# cgg[-1, :, :] = cgk[:, :]
# cgg[:, -1, :] = cgg[-1, :, :]
# # add noise
# cgg[-1, -1, :] = ckk  # + ckk_noise


# for i, ell in enumerate(lbins):
#   rho_comb[i] = np.dot(cgk[:, i], np.dot(np.linalg.inv(cgg[:, :, i]), cgk[:, i])) / ckk[i]
#   rho_gals[i] = np.dot(cgk[:-1, i], np.dot(np.linalg.inv(cgg[:-1, :-1, i]), cgk[:-1, i])) / ckk[i]
