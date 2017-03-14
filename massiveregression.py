import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec  # for unequal plot boxes
from scipy.optimize import curve_fit
import pandas as pd
import sqlite3
import csv
# Create Python Data Frame to store data for publication


#Now, for each new page we want to create, we have to create a new Figure instance
df = pd.DataFrame(columns=('number','id','n', 'nerror', 'a','aerror','r0','r0error','dm halo mass 10^10 Msolar','progenitor','parenthalo','redshift','chi_squared','dark_mass_percent float','dark_density_half float','star_density_half float','gas_density_half float','gas_radii_ratio float','star_radii_ratio float','star_mass','gas_mass','dark_mass_halo'))

exampleFile00 = open('listofhalos.txt')
exampleFile0 = open('listofradii.txt')
exampleFile2 = open('mass_at_radii.txt')

exampleReader00 = csv.reader(exampleFile00)
exampleReader0 = csv.reader(exampleFile0)
exampleReader2 = csv.reader(exampleFile2)


y111 = list(exampleReader00)

r111 = list(exampleReader0)
# parenthalos is the final list

listdm = list(exampleReader2)

##### PARENT HALO FILE

parenthalos = []
parenthalo = open('parenthalos.txt','r')
for y in parenthalo.read().split('\n'):
	if y.isdigit():
		parenthalos.append(int(y))
print "printing parent halos"
print parenthalos



#### Mass percent file



dark_mass_percent = open('dark_mass_percent.txt')

dark_masses =  [elem.strip().split(';') for elem in dark_mass_percent]

dark_masser = list(dark_masses)
dark_masse = [val for sublist in dark_masser for val in sublist]
dark_mass_percent = [float(i) for i in dark_masse]
print "printing dark_mass_percent"
print dark_mass_percent


###### DM MASS FILE

dm_mass = open('darkmattermass.txt')

dm_masses =  [elem.strip().split(';') for elem in dm_mass]

dm_masser = list(dm_masses)
dm_masse = [val for sublist in dm_masser for val in sublist]
dm_mass = [float(i) for i in dm_masse]
print "printing dm_mass"
print dm_mass




#### PROGENITOR FILE


progenitor = []
progs = open('progenitor.txt','r')
for y in progs.read().split('\n'):
	if y.isdigit():
		progenitor.append(int(y))
print "printing progenitor"
print progenitor



#### Redshift File

redshift = 2.2204460492503099e-16	

#### avg dark half rad file



dark_mass_half_rad = open('dark_density_half.txt')

dark_masses_half =  [elem.strip().split(';') for elem in dark_mass_half_rad]

dark_masser_half = list(dark_masses_half)
dark_masse_half = [val for sublist in dark_masser_half for val in sublist]
dark_density_half = [float(i) for i in dark_masse_half]
print "printing dark_mass_density_half"
print dark_density_half



#### avg star half rad file



star_mass_half_rad = open('star_density_half.txt')

star_masses_half =  [elem.strip().split(';') for elem in star_mass_half_rad]

star_masser_half = list(star_masses_half)
star_masse_half = [val for sublist in star_masser_half for val in sublist]
star_density_half = [float(i) for i in star_masse_half]
print "printing star_density_half"
print star_density_half



#### avg gas half rad file


gas_mass_half_rad = open('gas_density_half.txt')

gas_masses_half =  [elem.strip().split(';') for elem in gas_mass_half_rad]

gas_masser_half = list(gas_masses_half)
gas_masse_half = [val for sublist in gas_masser_half for val in sublist]
gas_density_half = [float(i) for i in gas_masse_half]
print "printing gas_density_half"
print gas_density_half

####  gas rad ratio file

gas_rad_ratio = open('gas_radii_ratio.txt')

gas_rad =  [elem.strip().split(';') for elem in gas_rad_ratio]

gas_radder = list(gas_rad)
gas_radde = [val for sublist in gas_radder for val in sublist]
gas_radii_ratio = [float(i) for i in gas_radde]
print "printing gas_density_ratio"
print gas_radii_ratio


####  star rad ratio file

star_rad_ratio = open('star_radii_ratio.txt')

star_rad =  [elem.strip().split(';') for elem in star_rad_ratio]

star_radder = list(star_rad)
star_radde = [val for sublist in star_radder for val in sublist]
star_radii_ratio = [float(i) for i in star_radde]
print "printing star_density_ratio"
print star_radii_ratio


##### Halo File



halo_id = []
halos = open('halo_id.txt','r')
for y in halos.read().split('\n'):
	if y.isdigit():
		halo_id.append(int(y))
print "printing halo id"
print halo_id



#####Total starmass file
stars = open('total_star_mass.txt')

star =  [elem.strip().split(';') for elem in stars]

stars1 = list(star)
stars2 = [val for sublist in stars1 for val in sublist]
star_mass = [float(i) for i in stars2]
print "star mass"
print star_mass





#####Total gas mass file
gas = open('total_gas_mass.txt')

gas1 =  [elem.strip().split(';') for elem in gas]

gas2 = list(gas1)
gas3 = [val for sublist in gas2 for val in sublist]
gas_mass = [float(i) for i in gas3]
print "gas mass"
print gas_mass




#####Total Dm mass file


#####Total starmass file
darks = open('total_dm_mass.txt')

darks1 =  [elem.strip().split(';') for elem in darks]

darks2 = list(darks1)
darks3  = [val for sublist in darks2 for val in sublist]
dark_mass_halo = [float(i) for i in darks3]
print "dark mass halo"
print dark_mass_halo






g = len(r111)

#flattened = [val for sublist in list_of_lists for val in sublist]
# HEre  I use a list comprehension


y11 =[val for sublist in y111 for val in sublist]

r11 = [val for sublist in r111 for val in sublist]

list_dm = [val for sublist in listdm for val in sublist]
pdf_pages = PdfPages('MassiveDMHalos135.pdf')


nm = 0
 
for i in xrange(nm):
  # Create a figure instance (ie. a new page)
  fig = plot.figure(figsize=(8.27, 11.69), dpi=100)
 
while nm < g:

	exampleFile = open(y11[nm])
	exampleReader = csv.reader(exampleFile)
	y1 = list(exampleReader)


	exampleFile1 = open(r11[nm])
	exampleReader1 = csv.reader(exampleFile1)
	r1 = list(exampleReader1)


	exampleFile2 = open(list_dm[nm])
	exampleReader2 = csv.reader(exampleFile2)
	total_dm_mass1 = list(exampleReader2)

	x = 0

	y2 = []



	while x < len(y1):
		mew = ''.join(y1[x])
	
		y2.insert(x,mew)
		x = x + 1

	y3 = map(float,y2)
	
	x1 = 0

	r2 = []

	while x1 < len(r1):
		mew1 = ''.join(r1[x1])
	
		r2.insert(x1,mew1)
		x1 = x1 + 1

	total_dm_mass2 = []

	x2 = 0
	
	while x2 < len(total_dm_mass1):
		mew2 = ''.join(total_dm_mass1[x2])
	
		total_dm_mass2.insert(x2,mew2)
		x2 = x2 + 1
	
	total_dm_mass1 = map(float,total_dm_mass2)
	total_dm_mass = total_dm_mass1[:len(y3)]
	y3 = map(float,y2)

	r3 = map(float,r2)		
	print len(r3), len(y3)
	e1 = np.multiply(0.2, y3)

#### NON LINEAR REGRESSION PORTION OF THE PROGRAM



	def line(r, a, r0 , n):
		#r00 = (r/(r0*1000))**2
	    
	
		return a * r**-n * (1.0 + r/r0)**(-3.0 + n)
		#return a*np.exp(-(r/r0)**n)

	param_bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,2])
		
	popt, pcov = curve_fit(line, r3, y3, sigma = e1, p0=[10.,10.,2.],bounds=param_bounds)
	from scipy.stats import chisquare

	perr = np.sqrt(np.diag(pcov))
	print " standard deviation of the error is"
	chi_squared = chisquare(y3)
	chi_squared = chi_squared[0]
	print chi_squared
	print "chisquare =", chisquare(y3)	
	    
  # Plot whatever you wish to plot
 
  # Done with the page
  
 	
# Write the PDF document to the disk

# Here the data is inserted into the data frame,halo_id[nm],dm_mass[nm],redshift[nm],descendants[nm],progenitor[nm],parenthalos[nm]]

	
	df.loc[nm] = [nm,halo_id[nm], popt[2] ,pcov[2,2]**0.5,popt[0],pcov[0,0]**0.5,popt[1],pcov[1,1]**0.5,dm_mass[nm],progenitor[nm],parenthalos[nm],redshift,chi_squared,dark_mass_percent[nm], dark_density_half[nm], star_density_half[nm],gas_density_half[nm],gas_radii_ratio[nm],star_radii_ratio[nm],star_mass[nm],gas_mass[nm],dark_mass_halo[nm]]


	#df = df.round({'n': 7, 'nerror': 5, 'a':5 ,'aerror':5,'r0':5,'r0error':5,'dm halo mass':6,'progenitor':1,'parent':1,'redshift':5,'chi_square':4,'mass_percent':4})

	
	fig, ax1 = plt.subplots()
	
	axes = plt.gca()
	axes.set	
	rfine = np.linspace(1, 1200, 150)  # define values to plot the function for
	plt.errorbar(r3, y3, yerr=e1)
	ax1.loglog(rfine, line(rfine, popt[0], popt[1],popt[2]), 'r-')
	ax1.set_xlabel('Radius kpc')
# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('density', color='b')
	ax1.tick_params('y', colors='b')
	plt.xlabel('Radius kpc')
	plt.ylabel('Dark Matter Density 10^10 Msolar /kpc^3')
		
	ax2 = ax1.twinx()

	ax2.plot(r3,total_dm_mass, 'r.')
	ax2.set_ylabel('Dark Matter Mass 10^10 Msolar', color='r')
	ax2.tick_params('y', colors='r')	
	plt.title('Dark Matter Density of halo %s at Redshift  %s'%(halo_id[nm],redshift))

	
	
	pdf_pages.savefig(fig)	
	nm = nm + 1



########  Create data table 



####### End of data table created. Now close pdf pages.

pdf_pages.close()


#####  Here we create our table 	
import sqlite3
conn = sqlite3.connect('mostmassivedarkmatter.db')    #Here I will create a database of thousands of darkmatters

c = conn.cursor()   #my cursor

# Create table
c.execute('''CREATE TABLE massiveDM135
             (number int PRIMARY KEY, id int , n float, ner float, a float, aer float, r0 float, r0er float,dm_mass float, progenitor int, parent int,redshift float,chi_squared float, dm_mass_percent float, dark_density_half float, star_density_half float,gas_density_half float,gas_radii_ratio float,star_radii_ratio float,star_mass float, gas_mass float, dark_mass_halo float)''')


print df


list_of_darkhalos = map(list, df.values)

print list_of_darkhalos 


for x in list_of_darkhalos:
	
	halo_string = ', '.join('?' * len(x))
	query_string = 'INSERT INTO massiveDM135 VALUES (%s);' % halo_string
	c.execute(query_string, x)
	
	


# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
