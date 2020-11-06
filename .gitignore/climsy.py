from statsmodels.tsa.seasonal import STL
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as font_manager
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
import calendar
from cartopy.util import add_cyclic_point
import metpy.calc as mpcalc
from matplotlib.offsetbox import AnchoredText
from decimal import Decimal
import datetime

def get_dataset(path, source):
	'''
	Converts xarray.Dataset objects to dataProcessor.

	Call signature::

		Climsy.dataset(path, source)

	*Args:
	----------
	path : str, Path, file, list
	
		File directory.

		OBS.: Accepts two values, returning two dataProcessor objects.

		Example:

		dataset_1, dataset_2 = Climsy.dataset(['C:/firstfile.nc', 'C:/secondfile.nc'], source)

	source : str
	
		Data source.

		- 'ERA-5'
		- 'NCEP'

	Returns
	-------
	obj : <Climsy.dataProcessor>

	'''
	if type(path) is list and len(path) == 2:

		data_1 = xr.open_dataset(path[0])
		data_2 = xr.open_dataset(path[1])

		return dataProcessor(data_1, source) , dataProcessor(data_2, source)

	else:
		data = xr.open_dataset(path)

		return dataProcessor(data, source)


class dataProcessor:
	'''
	Formatting class from which Climsy performs meteorological and mathematical operations such as divergence, vorticity, wind magnitude and scaling (division or multiplication).
	The supported data sources may have different netCDF structures (dimension namonth, etc...). This class returns standardized datasets from both sources so it matches other classes' methods
	within this library.

	Should be instanced through Climsy's dataset() method.

	Call signature ::

		Climsy.dataProcessor(data, source)

	*Args:
	----------
	data : xarray.open_dataset(path)
		
		Requires xarray.open_dataset() method, receiving the file's path as parameter.

	source : str
		
		Data source.

		- 'ERA-5'
		- 'NCEP'

	Example:

		Climsy.dataProcessor(xarray.open_dataset('C:/file.nc'), 'NCEP')

	OBS.: We highly recommend the Climsy's dataset() method as instance. Otherwise, should be instanced through xarray's open_dataset() method.

	Attributes
	----------
	dataset: 'xarray.dataset'
		
		Xarray's dataset-like data favors array visualization.

	variables: 'xarray.core.utils.Frozen'
		
		Access to dataset's variables.

	source: str
		
		Dataset's data source. ('ERA-5' or 'NCEP')

	data_view: 'dataPlot' object
		
		Attribute that converts a 'dataCalc' object into a 'dataPlot' object.

		Example:
		
		Instantiating a 'dataPlot' object by **dataProcessor.data_view** is equal to do it by **Climsy.data_view(dataProcessor.dataset, dataProcessor.source)**

	'''
	def __init__(self, data, source):

		self.dataset = data
		self.variables = data.variables
		self.source = source

		if 'expver' in self.dataset.coords:
		# removes 'expver' dimension when source == 'ERA-5'
			self.dataset = self.dataset.isel(expver = 0)
			self.dataset = self.dataset.drop('expver')

		# standardizes latitude, longitude, mtnlwrf, U and V from ERA-5 so it matches NCEP (lat, lon, olr, uwnd, vwnd)
		if 'latitude' in self.variables:
			self.dataset = self.dataset.rename({'latitude':'lat'})

		if 'longitude' in self.variables:
			self.dataset = self.dataset.rename({'longitude':'lon'})

		if 'u' in self.variables:
			self.dataset = self.dataset.rename({'u':'uwnd'})

		if 'v' in self.variables:
			self.dataset = self.dataset.rename({'v':'vwnd'})

		if 'z' in self.variables:
			self.dataset = self.dataset.rename({'z':'hgt'})

		if 'mtnlwrf' in self.variables:
			self.dataset = self.dataset.rename({'mtnlwrf':'olr'})

		# Creating attributes in order to automate future procedures with those values
		# In order not to lose relevant values
		self.dataset.attrs['source'] = self.source

		if 'analysis' in self.dataset.attrs:
			if self.dataset.attrs['analysis'] != 'seasonality by year':
				self.dataset.attrs['period'] = str(pd.to_datetime(self.dataset.time.values[0]).date()) + ' to ' + str(pd.to_datetime(self.dataset.time.values[-1]).date())
		else:
			self.dataset.attrs['period'] = str(pd.to_datetime(self.dataset.time.values[0]).date()) + ' to ' + str(pd.to_datetime(self.dataset.time.values[-1]).date())

		try:
			self.dataset.attrs['lat'] = str(self.dataset.lat.values[0]) + ' to ' + str(self.dataset.lat.values[-1]) + ' degrees'
			self.dataset.attrs['lon'] = str(self.dataset.lon.values[0]) + ' to ' + str(self.dataset.lon.values[-1]) + ' degrees'
		except:
			pass

		try:
			self.dataset.attrs['level'] = int(self.dataset.level.values)
		except:
			pass

		if 'analysis' in self.dataset.attrs:
			
			if self.dataset.attrs['analysis'] == 'climatology' or self.dataset.attrs['analysis'] == 'averages' or self.dataset.attrs['analysis'] == 'anomaly' or self.dataset.attrs['analysis'] == 'climatological seasonality' or self.dataset.attrs['analysis'] == 'seasonal averages':
				self.dataset = self.dataset.drop(labels = 'time')

		self.data_view = data_view(self.dataset, source)


	def add_attribute(self, attrs_name, attrs_info):
		'''
		Adds an attribute to the dataset.
		Returns a 'dataProcessor' object. (The same object but with the new attributes)

		Call signature::

			self.add_attribute(attrs_name, attrs_info)

		Args
		----------
		attrs_name : str
			
			Atributte name

		attrs_info : str, int, float
			
			Information of the attribute created

		Returns
		-------
		obj : <Climsy.dataProcessor>

		'''
		new_attrs_dataset = self.dataset
		new_attrs_dataset.attrs[attrs_name] = attrs_info

		return dataProcessor(new_attrs_dataset, self.source)


	def formatter(self, opr = None, factor = None, data_2 = None, level = None, lat = None, lon = None, time = None, freq = None):
		'''
		Dataset selection over time, coordinates and geopotential levels as well as scaling through multiplication or division operations according to a factor given by the user. Relies on xarray's sel method to perform selection.
		Returns a 'dataCalc' object.

		Call signature::

			self.formatter(**opr = None,
			**factor = None,
			**data_2 = None,
			**level = None,
			**lat = None,
			**lon = None,
			**time = None,
			**freq = None)

		All function arguments are optional. If None given, returns the dataset itself as a dataCalc object.
		If desired operation requires multiple variables, such as the wind relative vorticity calculus, formatter
		function receives another dataProcessor object as an attribute of '**data2' or even a single dataset with both
		variables.

		Example:  wind magnitude given two different dataProcessor objects

			dataset_Vwind.formatter(data_2 = dataset_Uwind, opr = 'magnitude')

													---------------

		Example: wind magnitude given one dataProcessor object with different variables within.

			dataset_Uwind_and_Vwind.formatter(opr = 'magnitude')

													---------------

		**Kwargs
		----------
		data_2 : dataProcessor
			
			Second dataProcessor object when the desired operation requests multiple variables (magnitude, divergence, vorticity) when both variables are placed in different dataProcessor objects.

			Example:

			dataset_Vwind.formatter(data_2 = dataset_Uwind, opr = 'magnitude')

		opr : str
			
			Meteorological or scaling operation. Valid operations: 'magnitude', 'vorticity', 'divergence', 'division' or 'multiplication'

		factor: int or float
			
			If operation matches 'division' or 'multiplication', factor must be given.

		level : int
			
			Selects isobaric level.

		lat : list or int
			
			Selects y coordinate.

			Example:

			dataProcessor.formatter(lat = [10, -50])

		lon : list
			
			Selects x coordinate.

			Example:

			dataProcessor.formatter(lon = [-60, 10])

		time: list
			
			Selects single time value or time range.

			Example:

			dataProcessor.formatter(time = ['1979-01-01', '2010-12-01'])

		freq: {'monthly', 'seasonal', 'daily'}
			
			Selects the type of time analysis. Stands for **None** by default, which implies that dataset does not pass through any 'resample' method, keeping its original shape.

			Example:

			dataProcessor.formatter(freq='monthly') --> converting a daily dataset to a monthly one.

		Returns
		-------
		obj : <Climsy.dataCalc>

		'''
		# creating a list with datasets
		# if data_2 is provided, len(dataProcessors == 2)

		dataProcessors = []
		data_base = self.dataset
		dataProcessors.append(data_base)

		if data_2 != None:
			data_base_2 = data_2.dataset
			dataProcessors.append(data_base_2)

		# creating a list with datasets after slicing
		# if data_2 is provided, len(datas == 2)
		datas = []

		# Slicing level, lat, lon, time dimensions
		for data in dataProcessors:

			if level != None:
			# In case of a dataset downloaded for a single level directly from the source, the level's value is not provided in the dataset
			# For this case, the user must provide the level's value not for a slice method, but for the creation of a new attribute carrying it's value
			# That new attribute with the value of level is NECESSARY for future procedures inside this module
			# There will be a warning message informing the user about the need of providing the level´s value in case of a single level dataset

				if 'level' in self.dataset:
					data = data.sel(level = level)
				else:
					# insert warning message
					self.add_attribute('level', str(level))

			if lat != None:
				
				if type(lat) == list:
					data = data.sel(lat = slice(lat[0], lat[1]))
				else:
					data = data.sel(lat = lat)

			if lon != None:
				
				if self.source == 'NCEP':
				# NCEP's longitude values are arranged in a [0, 360 East] sequency
				# Adapting to ERA-5's longitude sequency [-90 East, 90 East]

					if type(lon) == list:
						if lon[0] > lon [1]:
							w = data.sel(lon = slice(lon[0], 360))
							e = data.sel(lon = slice(0, lon[1]))
							e['lon'] = e['lon'] + 360
							data = xr.merge([e,w])
						else:
							data = data.sel(lon = slice(lon[0], lon[1]))
					else:
						data = data.sel(lon = lon)
				else:
					
					if type(lon) == list:
						data = data.sel(lon = slice(lon[0], lon[1]))
					else:
						data = data.sel(lon = lon)

			if time != None:
				# the slice method can be applied to a single time or a time range
				if type(time) == list:
					data = data.sel(time = slice(time[0], time[1]))
				else:
					data = data.sel(time = time)

			if freq != None:
			# Resampling datasets into monthly, season or daily datasets according to 'freq' required
			# There are procedures to keep the attributes, both dataset_variables.attrs and dataset.attrs, from the original datasets --> attrs get lost after resampling

				if freq == 'monthly':
					# keeping the original dataset.attrs
					attrs = data.attrs

					# keeping the original dataset_variables.attrs inside a dict
					var_attrs = {}
					var_name = data.to_array().coords['variable'].values
					for var in var_name:
						var_attrs.update({str(var):data[str(var)].attrs})

					# Applying the resample
					data = data.resample(time = 'MS').mean()

					# keeping both dataset_variables.attrs and dataset.attrs
					data.attrs = attrs
					for key in var_attrs.keys():
						data[key].attrs = var_attrs[key]

				elif freq == 'daily':
					# keeping the original dataset.attrs
					attrs = data.attrs

					# storing the original dataset_variables.attrs inside a dict
					var_attrs = {}
					var_name = data.to_array().coords['variable'].values
					for var in var_name:
						var_attrs.update({str(var):data[str(var)].attrs})

					# Applying the resample
					data = data.resample(time = 'D').mean()

					# keeping both dataset_variables.attrs and dataset.attrs
					data.attrs = attrs
					for key in var_attrs.keys():
						data[key].attrs = var_attrs[key]

				elif freq == 'seasonal':
					# keeping the original dataset.attrs
					attrs = data.attrs

					# storing the original dataset_variables.attrs inside a dict
					var_attrs = {}
					var_name = data.to_array().coords['variable'].values
					for var in var_name:
						var_attrs.update({str(var):data[str(var)].attrs})

					# Applying the resample
					data = data.resample(time = 'QS-DEC').mean()

					# keeping both dataset_variables.attrs and dataset.attrs
					data.attrs = attrs
					for key in var_attrs.keys():
						data[key].attrs = var_attrs[key]

			datas.append(data)

		if opr == 'magnitude' or opr == 'mag':
		# magnitude formatter
		# creating dataset of magnitude
		# returns a dataCalc object

			if len(datas) == 1:
				data = datas[0]
				data['magnitude'] = (data.uwnd**2 + data.vwnd**2)**0.5
				magnitude = data.drop(['uwnd', 'vwnd'])
			else:
				if 'uwnd' in datas[0]:
					dataset = ((datas[0].uwnd)**2 + (datas[1].vwnd)**2)**0.5
					magnitude = dataset.to_dataset(name = 'magnitude')
				else:
					dataset = ((datas[1].uwnd)**2 + (datas[0].vwnd)**2)**0.5
					magnitude = dataset.to_dataset(name = 'magnitude')

			return set_calc(magnitude, self.source)

		elif opr == 'divergence' or opr == 'div':
		# divergence formatter using metpy
		# creating dataset of divergence
		# returns a dataCalc object

			if len(datas) == 1:
			# When both variables 'uwnd' and 'vwnd' are available in dataset
			# When **data_2 is not provided

				dataset = datas[0]
				d = []
				dx, dy = mpcalc.lat_lon_grid_deltas(dataset.variables['lon'][:], dataset.variables['lat'][:])

				for i, data in enumerate(dataset.variables['time'][:]):
					div = mpcalc.divergence(dataset.uwnd.isel(time = i),
															dataset.vwnd.isel(time = i),
															dx, dy, dim_order = 'yx')
					d.append(xr.DataArray(div.m,
								  dims = ['lat', 'lon'],
								  coords = {'lat': dataset.variables['lat'][:], 'lon': dataset.variables['lon'][:], 'time': dataset.variables['time'][:][i]}, name = 'div'))

				divergence = xr.concat(d, dim = 'time').to_dataset()
				divergence.attrs = dataset.attrs

			else:
			# One dataset for each variable 'uwnd' and 'vwnd'
			# **data_2 provided

				d = []
				dx, dy = mpcalc.lat_lon_grid_deltas(datas[0].variables['lon'][:], datas[0].variables['lat'][:])

				for i, data in enumerate(datas[0].variables['time'][:]):
					div = mpcalc.divergence(datas[0].uwnd.isel(time = i),
															datas[1].vwnd.isel(time = i),
															dx, dy, dim_order = 'yx')
					d.append(xr.DataArray(div.m,
								  dims = ['lat', 'lon'],
								  coords = {'lat': datas[0].variables['lat'][:], 'lon': datas[0].variables['lon'][:], 'time': datas[0].variables['time'][:][i]}, name = 'div'))

				divergence = xr.concat(d, dim = 'time').to_dataset()

			divergence.attrs['level'] = level

			return set_calc(divergence, self.source)

		elif opr == 'vorticity' or opr == 'vort':
		# vorticity formatter using metpy
		# creating dataset of vorticity
		# returns a dataCalc object

			if len(datas) == 1:
			# When both variables 'uwnd' and 'vwnd' are available in dataset.
			# When **data_2 is not provided

				dataset = datas[0]
				v = []
				dx, dy = mpcalc.lat_lon_grid_deltas(dataset.variables['lon'][:], dataset.variables['lat'][:])

				for i, data in enumerate(dataset.variables['time'][:]):
					vort = mpcalc.vorticity(dataset.uwnd.isel(time = i),
															dataset.vwnd.isel(time = i),
															dx, dy, dim_order = 'yx')
					v.append(xr.DataArray(vort.m,
								  dims = ['lat', 'lon'],
								  coords = {'lat': dataset.variables['lat'][:], 'lon': dataset.variables['lon'][:], 'time': dataset.variables['time'][:][i]}, name = 'vort'))

				vorticity = xr.concat(v, dim = 'time').to_dataset()
				vorticity.attrs = dataset.attrs

			else:
			# One dataset for each variable 'uwnd' and 'vwnd'
			# When **data_2 is provided

				v = []
				dx, dy = mpcalc.lat_lon_grid_deltas(datas[0].variables['lon'][:], datas[0].variables['lat'][:])

				for i, data in enumerate(datas[0].variables['time'][:]):
					vort = mpcalc.vorticity(datas[0].uwnd.isel(time = i),
															datas[1].vwnd.isel(time = i),
															dx, dy, dim_order = 'yx')
					v.append(xr.DataArray(vort.m,
								  dims = ['lat', 'lon'],
								  coords = {'lat': datas[0].variables['lat'][:], 'lon': datas[0].variables['lon'][:], 'time': datas[0].variables['time'][:][i]}, name = 'vort'))

				vorticity = xr.concat(v, dim = 'time').to_dataset()

			vorticity.attrs['level'] = level

			return set_calc(vorticity, self.source)

		elif opr == 'division' or opr == '/':
			# formatting dataset according to the operation and factor provided
			data = datas[0]
			new_data = data / factor

			return set_calc(new_data, self.source)

		elif opr == 'multiplication' or opr == 'x':
			# formatting dataset according to the operation and factor provided
			data = datas[0]
			new_data = data * factor

			return set_calc(new_data, self.source)

		else:
			# as any operation is provided, returns self.dataset
			data_base = datas[0]

			return set_calc(data_base, self.source)

def set_calc(data, source):
	# instantiating dataCalc object
	dataset = data

	return dataCalc(dataset, source)

class dataCalc(dataProcessor):  # dataCalc herits from dataProcessor, which means its objects have the same attributes.
	'''
 	Long term means, anomalies, seasonal average and area average are methods from dataCalc class.
	The 'dataCalc' is a child class of 'dataProcessor', in other words it herits all dataProcessor attributes.

	Call signature ::

		Climsy.dataCalc(dados, source)

	*Args:
	----------
	dados : xarray.open_dataset(path)
		
		Requires xarray's open_dataset method and take the file's path as argument

	source : str
		
		Data source.

		- 'ERA-5'
		- 'NCEP'

	Example:

		Climsy.dataProcessor(xarray.open_dataset('C:/file.nc'), 'NCEP')

	Attributes
	----------
	dataset: 'xarray.dataset'
		
		Xarray's dataset-like data favors array visualization.

	variables: 'xarray.core.utils.Frozen'
		
		Access to dataset's variables.

	source: str
		
		Data source. ('ERA-5' or 'NCEP')

	data_view: 'dataPlot' object
		
		Attribute that converts a 'dataCalc' object into a 'dataPlot' object.

		Example:

		Instantiating a 'dataPlot' object by **dataCalc.data_view** is equal to do it by **Climsy.data_view(dataCalc.dataset, dataCalc.source)**

	'''

	def slicer(self, series = 'all-time'):
		'''
		Receives a dataset to be time-sliced according to series parameter.

		Call signature::

			self.slicer(series)

		*Args:
		----------
		series: 'all-time', 'rainy season' or list

			Slicing pattern. 'all-time' returns the whole time series, while 'rainy season' returns time series sliced from october to april for each year (as described in brazilian weather). A list of integers should be given if the user wants to manually select a month or a list of months. For example, series = [1, 4, 6] returns the given dataset sliced at January, April and June.

		Returns
		----------
		obj : <Climsy.dataCalc>

		'''
		ds = self.dataset

		if series == 'all-time':
			data = ds
		elif series == 'rainy season':
			ends = ds.where(ds['time.month']>= 10, drop = True)
			begins = ds.where(ds['time.month']<4, drop = True)
			data = xr.concat([ends, begins], dim = 'time')
			data = data.sortby(data.time).dropna(dim = 'time')

		elif type(series) == list:

			monthly_data = []

			for month in series:
				sliced_data = ds.where(ds['time.month'] == month, drop = True)
				monthly_data.append(sliced_data)

			data = xr.concat(monthly_data, dim = 'time')
			data = data.sortby(data.time).dropna(dim = 'time')

		return dataCalc(data, self.source)


	def loess(self, type = 'trend', series = 'rainy season', seasonality=7, period=6, robust=False, seasonal_deg=1, trend_deg=1, low_pass_deg=1):
		"""
		Loess decomposition of a dataset, which can return its trend, seasonality, observed values or its noise.

		Call Signature::

			self.loess(**type, **series, **seasonality, **period, **robust, **seasonal_deg, **trend_deg, **low_pass_deg)

		**Kwargs:
		--------
		type: 'trend', 'seasonality', 'observed', 'residual'. Default: 'trend'
			
			Loess filters.

		series: 'rainy season', 'all-time' or list. Default: 'rainy season'
			
			How the timeseries should be sliced according to its time values. (see climsy.dataCalc.slicer)

		seasonality: odd int > 3
			
			Smooth factor according to data seasonality. (see statsmodels.tsa.seasonal.STL)

		period: int
			
			See statsmodels.tsa.seasonal.STL.

		robust: bool
			
			See statsmodels.tsa.seasonal.STL.

		seasonal_deg: int
			
			See statsmodels.tsa.seasonal.STL.

		trend_deg: int
			
			See statsmodels.tsa.seasonal.STL.

		low_pass_deg: int
			
			See statsmodels.tsa.seasonal.STL.
		"""
		#Função que recorta um dataframe de acordo com o recorte de série caso recorte = True e calcula a decomposição
		#de loess de acordo com o tipo de análise ('tendência', 'sazonalidade', 'residual', 'observado')

		ds = self.dataset
		dataset = ds.to_array().isel(variable = 0)
		var = str(dataset['variable'].values)

		#recorte dos dados e set da sazonalidade e período de acordo com o intervalo da série temporal resultante
		ds = set_calc(ds, self.source).slicer(series = series).dataset

		if 'level' in ds:
			ds = ds.drop('level')

		df = ds.to_dataframe()

		#decomposição de loess
		stl = STL(df, seasonal = seasonality, period = period, robust = robust, seasonal_deg = seasonal_deg, trend_deg = trend_deg, low_pass_deg = low_pass_deg)
		res = stl.fit()

		#retorna resultado de acordo com o tipo de análise
		if type == 'trend':
			res = res.trend
		elif type == 'seasonality':
			res = res.seasonal
		elif type == 'observed':
			res = res.observed
		elif type == 'residual':
			res = res.resid

		loess_da = xr.DataArray.from_series(res)
		loess_da.name = var
		loess_ds = loess_da.to_dataset()
		loess_ds.attrs = ds.attrs
		loess_ds.attrs['analysis'] = 'loess ' + str(type)

		return dataCalc(loess_ds, self.source)


	def anom_ts(self, period=['1979-01-01', '2018-12-01'], plot = True, freq="default", series='all-time'):
		"""
		Returns an area averaged anomaly timeseries.

		Call signature ::

			self.anom_ts(**period, **plot, **freq, **series)

		**Kwargs
		----------
		period: list of dates
			
			Corresponds to the time window that will be shown in the x axis.

		plot: boolean, default = True.
			
			Selects whether the user wants to plot the timeseries or not. Returns a dataPlot object.

		freq: str, default = True
			
			Resamples data. Input string must follow xarray.resample() syntax.

		series: str, list. default = 'all-time'.
			
			Selects the season of the year. Input string must be a string or a list of months.

		Returns
		---------
		obj : <Climsy.dataCalc> or <Climsy.dataPlot>

		"""
		#slicing the dataset according to the input on period parameter
		#corresponds to the time window that will be shown in the x axis
		ds_format = self.formatter(time=[period[0], period[1]])

		level = str(ds_format.dataset.attrs['level'])

		#anomaly = series - climatology
		anom_ds = ds_format.dataset - ds_format.climatology().dataset

		#instancing anom_ds as a climsy object so climsy methods can be easily applied in it
		anom_ds = get_dataset(anom_ds.to_netcdf(), "NCEP")

		#calculating area averaged values of the anomaly
		med_anom_ds = anom_ds.formatter().aave()

		#setting data frequency according to freq parameter
		if freq != "default":
			med_anom_ds = med_anom_ds.dataset.resample(time=freq).mean()
			med_anom_ds = get_dataset(med_anom_ds.to_netcdf(), "NCEP").formatter()

		#setting time-slice according to series parameter
		if series != 'all-time':
			med_anom_ds = med_anom_ds.slicer(series=series)

		med_anom_ds.dataset.attrs['analysis'] = 'anomaly timeseries'

		if plot == False:
			return med_anom_ds

		elif plot == True:
			#stores coordinate values from the dataset
			latitude = [ds_format.dataset.lat[0].values.tolist(), ds_format.dataset.lat[-1].values.tolist()]
			longitude = [ds_format.dataset.lon[0].values.tolist(), ds_format.dataset.lon[-1].values.tolist()]

			med_anom_ds.data_view.cumulative(series = series, level = level, latitude=latitude)


	def climatology(self, time=None, level=None):
		'''
		Returns long term mean of a dataset given a time range (optional, as list) and isobaric level (optional, as integer).
		If the dataset is already dimension-selected by formatter() method from the parent class, time and level arguments should be ignored unless a new selection is desired.

		Call signature ::

			self.climatology(**time, **level)

		**Kwargs
		----------
		time: list
			
			Selects single time value or time range.

		level: int
			
			Selects isobaric level.

		Returns
		----------
		obj : <Climsy.dataCalc>

		'''
		data = self.dataset

		# Slicing dimensions 'time' and 'level'
		if time != None:
			data = data.sel(time = slice(time[0], time[1]))
		if level != None:
			data = data.sel(level = level)

		# grouping by monthly means
		climatology = data.groupby('time.month').mean()

		# Keeping the original dataset atributtes
		climatology.attrs = data.attrs
		climatology['time'] = data.time.values

		# Naming the analysis
		# Climatology analysis requires a time range >= 30 years
		# If time range < 30 years, the analysis is defined as 'averages'

		if pd.to_datetime(data.time.values[-1]).year - pd.to_datetime(data.time.values[0]).year >= 30:
			climatology.attrs['analysis'] = 'climatology'
		else:
			climatology.attrs['analysis'] = 'averages'

		# Creating a attribute for 'level'
		# Automates future procedures with 'level' dimensions
		if level != None:
			climatology.attrs['level'] = level

		return dataCalc(climatology, self.source)


	def anomaly(self, time, basetime, freq, level = None):
		'''
		Returns anomaly given two time ranges.
		If the dataset is already dimension-selected by formatter() method from the parent class, level argument should be ignored unless a new selection is desired.

		Call signature ::

			self.anomaly(time, basetime, **level = None)

		*Args:
		----------
		time: list
			
			Anomaly period.

		basetime: list
			
			Long term mean date range from which values should be compared.

		freq: {'monthly', 'seasonal'}
			
			Selects the type of time analysis. Stands for 'monthly' by default.

		**Kwargs
		----------
		level: int
			
			Selects isobaric level.

		Example:

			self.anomaly(time = ['2016-01-01', '2018-01-01'], basetime = ['1979-01-01', '2010-01-01'], level = 700, freq = 'seasonal')

		Returns
		----------
		obj : <Climsy.dataCalc>

		'''
		data = self.dataset

		# Slicing
		if level != None:
			data = data.sel(level = level)

		newest = data.sel(time = slice(time[0], time[1]))
		oldest = data.sel(time = slice(basetime[0], basetime[1]))

		# Grouping by monthly/seasonal means according to frequency required
		if freq == 'monthly':
			gpnewest = newest.groupby('time.month').mean()
			gpoldest = oldest.groupby('time.month').mean()
		elif freq == 'seasonal':
			gpnewest = newest.groupby('time.season').mean()
			gpoldest = oldest.groupby('time.season').mean()

		anomaly = gpnewest - gpoldest

		# Creating attributes in order not to lose relevant values (time, base period, analysis, level)
		# Automates future procedures with these values
		# Keeping the original dataset atributtes

		anomaly.attrs = data.attrs

		anomaly['time'] = newest.time.values
		anomaly.attrs['analysis'] = 'anomaly'
		anomaly.attrs['base period'] = str(pd.to_datetime(oldest.time.values[0]).date()) + ' to ' + str(pd.to_datetime(oldest.time.values[-1]).date())

		if level != None:
			# Creating a attribute of 'level' in order not to lose it's value after slicing
			anomaly.attrs['level'] = level

		return dataCalc(anomaly, self.source)


	def aave(self, lat = None, lon = None, time = None, level = None):
		'''
		Returns area average.
		If the dataset is already dimension-selected by formatter() method from the parent class, coordinates, time and level arguments should be ignored unless a new selection is desired.

		Call signature ::

			self.aave(**lat = None, **lon = None, **time = None, **level = None)

		**Kwargs
		----------

		lat: list
			
			Selects y coordinate.

		lon: list
			
			Selects x coordinate.

		time: list
			
			Selects single time value or time range.

		level: int
			
			Selects isobaric level.

		Example:

			self.aave(lat = [10, -40], time = ['2016-01-01', '2018-01-01'], level = 700)

		Returns
		----------
		obj : <Climsy.dataCalc>

		'''
		valores = []
		var = self.dataset

		# Slicing according to kwargs provided
		if level != None:
			var = var.sel(level = level)
		if time != None:
			var = var.sel(time = slice(time[0], time[1]))
		if lat != None:
			var = var.sel(lat = slice(lat[0], lat[1]))
		if lon != None:
			var = var.sel(lon = slice(lon[0], lon[1]))

		# Creating dataset for area averages
		for i, data in enumerate(var.time.values):
			valores.append(var.isel(time = i).map(np.mean))

		ds = xr.concat(valores, dim = 'time')

		ds.attrs = var.attrs

		# Creating attributes in order not to lose relevant values
		# Automates future procedures with these values
		ds.attrs['analysis'] = 'area average'
		ds.attrs['lat'] = str(var.lat.values[0]) + ' to ' + str(var.lat.values[-1]) + ' degrees'
		ds.attrs['lon'] = str(var.lon.values[0]) + ' to ' + str(var.lon.values[-1]) + ' degrees'

		if level != None:
			# Creating a attribute of 'level' in order not to lose it's value after slicing
			ds.attrs['level'] = level

		return dataCalc(ds, self.source)


	def seasonality(self, time = None):
		'''
		Returns seasonality for each month and year along the time range.
		If the dataset is already dimension-selected by formatter() method from the parent class, time argument should be ignored unless a new selection is desired.

		Call signature ::

			self.seasonality(**time = None)

		**Kwargs
		----------

		time: list
			
			Selects single time value or time range.

		Example:

			self.seasonality(time = ['2016-01-01', '2018-01-01'])

		Returns
		----------
		obj : <Climsy.dataCalc>

		'''
		data_list = []
		time_list = []
		data = self.dataset

		# Slicing 'time' dimension
		if time != None:
			data = data.sel(time = slice(time[0], time[1]))
		else:
			time = [str(data.time[0].values)[:10], str(data.time[-1].values)[:10]]

		# Creating a list of datasets grouped by season for each interval of one year in data's time range
		for i in range(0, len(data.time.values), 12):
			d_slice = data.isel(time = slice(i, i+12))
			d_season = d_slice.groupby('time.season').mean()
			data_list.append(d_season)
			time_list.append(pd.to_datetime(data.time.values[i]).year)

		# Concatenating datasets and indexing by year
		idx = pd.Index(time_list, name = 'time')
		ds = xr.concat(data_list, dim = idx)

		# Naming the analysis
		# Keeping the original dataset atributtes
		ds.attrs = data.attrs
		ds.attrs['analysis'] = 'seasonality by year'
		ds.attrs['period'] = str(ds.time[0].values) + ' to ' + str(ds.time[-1].values)

		return dataCalc(ds, self.source)


	def seasonality_clima(self, time = None):
		'''
		Returns seasonal climatology for each month along the time range.
		If the dataset is already dimension-selected by formatter() method from the parent class, time argument should be ignored unless a new selection is desired.

		Call signature ::

			self.seasonality_clima(**time = None)

		**Kwargs
		----------
		time: list
			
			Selects single time value or time range.

		Example:

			self.seasonality_clima(time = ['2016-01-01', '2018-01-01'])

		Returns
		----------
		obj : <Climsy.dataCalc>

		'''
		data = self.dataset

		# Slicing 'time' dimension
		if time != None:
			data = data.sel(time = slice(time[0], time[1]))
		else:
			time = [str(data.time[0].values)[:10], str(data.time[-1].values)[:10]]

		# grouping by seasonal means
		data_season = data.groupby('time.season').mean()

		# Creating attributes in order not to lose relevant values
		# Automates future procedures with these values
		# Keeping the original dataset atributtes
		data_season.attrs = data.attrs
		data_season['time'] = data.time.values

		if pd.to_datetime(data.time.values[-1]).year - pd.to_datetime(data.time.values[0]).year >= 30:
			data_season.attrs['analysis'] = 'climatological seasonality'
		else:
			data_season.attrs['analysis'] = 'seasonal averages'

		return dataCalc(data_season, self.source)


	def persistence(self, datas = None, how = 'continuous', freq = 'daily', save_netcdf = False):
		"""
		Persistence method generates a dataset containing the days of positive values of the atmospheric variable, if the positive values persists for over 3 consecutive days.
		From the third day and on, one day of blocking is computed based on the variable alone.
		If more than one variable is given, they are treated separately.

		Call signature ::

			persistence(*datas, **how, **freq, **save_netcdf)

		*Args:
		----------
		datas: list of <climsy.dataCalc>
		    
			List of datasets.

			Example: datas = [vort_850, div_850] <--> [div_850, vort_850] --> the order doesn't matter.

		**Kwargs
		----------
		how: str, {'continuous', 'total'}
		    
			Continuous how computes every single day; total how returns only the higher number at the end of the series of positive values until they reach a negative value.

			Example: (continuous)
		        
			If 1979-01-03 is the third consecutive day of positive vorticity value, for example, one day of blocking is computed.
			If 1979-01-04 is the fourth consecutive day of positive vorticity value, for example, this date receives the number 2 referring to the second day of atmospheric blocking.

			Example: (total)
		        
			Following the previous example, assuming 1979-01-05 shows a negative vorticity value, 1979-01-03 receives 0 (because 1 < 2) and 1979-01-04 receives 2. This method is recommended if the user wants to compile the number of blocking days by frequencies other than daily.

		freq: str, {'daily', 'monthly'}
		    
			Frequency of the series.
		        
			If *freq == 'daily'*, analysis will be daily.
			If *freq == 'monthly'*, analysis will be monthly.

			Example showing 'time' dimension of a dataset:
			
			If *freq == 'daily' * --> 1979-01-01, 1979-01-02, 1979-01-03... day by day
			If *freq == 'monthly' * --> 1979-01-01, 1979-02-01, 1979-03-01... month by month

		save_netcdf: bool
			
			Chooses whether a netcdf containing the results will be exported or not.

		Returns
		-------
		obj : <climsy.dataCalc>
		    
			Dataset containing the persistence of positive values.
			Variable values of the dataset will represent the days of blocking computed for each day or month of the timeseries.
		    OBS.:  Negative days receive NaN values.
		"""
		d_arrays = []
		level = []
		variables = []
		objects = [self]

		# Even if only one dataset is given it is handled as a list to execute the loop
		if datas != None:
			if type(datas) != list:
				datas = [datas]
			for data in datas:
				objects.append(data)

		# Unlimited number of objects --> allows analysis for different levels --> creates a variable for each level (var_name) into the same dataset
		for obj in objects:

			start = pd.to_datetime(obj.dataset.time.values[0])
			end = pd.to_datetime(obj.dataset.time.values[-1])

			# original variable name
			ext_name = obj.dataset.to_array().isel(variable = 0)
			var = str(ext_name['variable'].values)

			# calculating area average using 'Climsy' function
			aave = obj.aave().dataset

			# selecting only positive values for the index calculation
			dropped = aave.where(aave[var]>0, drop = True)

			# counters and dict 'persistence' return a dataset with the number days of persistence
			# counter 'count' counts only consecutive days
			# counter 'pst_count' counts days of persistence --> 3+ consecutive days of positive vort/div
			count = 0
			pst_count = 0
			persistence = {}

			for ix, day in enumerate(pd.to_datetime(dropped.time.values)):
				
				# First time.day --> counter is starting --> count += 1 --> as count is initially = 0 --> count = 1
				# 'pst_count' keeps reset because, to be counted as a persistence, it has to be at least 3 consecutive days
				# As it is the first day of analysis, it is impossible to be computed as a persistence
				
				if ix == 0:
					count += 1
					pst_count = 0
					persistence.update({day: pst_count})

				else:
					# Checking if the next value of time.day with positive values of div/vort is a consecutive day  (EX: 1979-01-01, 1979-01-02 --> consecutive days )
					if (day == pd.to_datetime(dropped.time.values[ix-1]) + datetime.timedelta(days = 1)) and (day.day != 1):

						count += 1

						if count >= 3:
							# If time.day is 3+ consecutive day of positive vort/div --> counts 1 day of persistence --> pst_count += 1
							pst_count += 1

						if how == 'total':
							
							# Only coherent if Freq == 'M' or Freq == 'S'
							# As the total value of persistence in month/season is desired, it counts only the total days of persistence, not the consecutive days itselves
							# Values different from ZERO only appear on the last days of the persistence

							# EX: if VORT > 0 in 1979-01-01, 1979-01-02, 1979-01-03 and 1979-01-04 --> 4 consecutive days of positive vorticity --> 2 days of persistence

							# If how == 'total' --> appears number 2 in 1979-01-04 --> there were 2 total days of persistence, ending the sequence in 1979-01-04
							# Value 2 indicates a sequence of days in the month/season that computed 2 days of persistence --> sequence of 4 consecutive days with positive vort/div
							
							persistence.update({day - datetime.timedelta(days = 1): 0})
							persistence.update({day: pst_count})

						else:
							
							# If how == 'continuos' --> adobj value 1 in 1979-01-03 and 2 in 1979-01-04 --> these values indicate the sequential numbering of days with positive vort/div
							# Value 1 indicates the first day of persistence <--> third consecutive day of positive vort/div (!!!)
							# Value 2 indicates the second day of persistence <--> fourth consecutive day of positive vort/div (!!!)
							
							persistence.update({day: pst_count})
					else:
						
						# As the next value of time.day with positive values for div/vort is not a consecutive day (EX: 1979-01-01, 1979-02-12 --> not consecutive days), the counter 'count' restarts ( = 1)
						# 'count' restarts and 'pst_count' resets because the time.day is not a consecutive day
						
						pst_count = 0
						persistence.update({day: pst_count})

			# values of level to be the attributes
			# if there is a combination of variables from different levels, all values of level will be provided
			level.append(obj.dataset.attrs['level'])

			# variable name to be assigned --> var_name + level
			var_name = var + ' ' + str(obj.dataset.attrs['level'])
			variables.append(var_name)

			# creating a DataArray from dict 'persistence'
			d_arrays.append(xr.DataArray(list(persistence.values()), coords = [list(persistence.keys())], dims = ['time'], name = var_name))

		# merging the DataArray objects and converting them into a Dataset
		pst = xr.merge(d_arrays)
		pst = pst.reindex({'time': pd.date_range(start = start, end = end, freq = 'D')})

		# keeping attributes from the original dataset
		pst.attrs = obj.dataset.attrs

		# creating new attributes for variable names
		pst.attrs['vars by level'] = [[var for var in variables][i] for i in range(len(variables))]

		if freq == 'daily':
		# Set the resample frequency and attributes
			pst = pst
			pst.attrs['level'] = level

		elif freq == 'monthly' and how == 'total':
		# Set the resample frequency and attributes
			pst = pst.resample(time = 'MS').sum()

			# keeping attributes from the original dataset --> after resample attributes get lost
			pst.attrs = obj.dataset.attrs
			pst.attrs['level'] = level

		elif freq == 'seasonal' and how == 'total':
		# Set the resample frequency and attributes
			pst = pst.resample(time = 'QS-DEC').sum()

			# keeping attributes from the original dataset --> after resample attributes get lost
			pst.attrs = obj.dataset.attrs
			pst.attrs['level'] = level

		if save_netcdf == True:

			if pst.attrs['lat'] == '-10.0 to -25.0 degrees':
				area = "Total area"
			elif pst.attrs['lat'] == '-10.0 to -17.5 degrees':
				area = "Northern section"
			elif pst.attrs['lat'] == '-17.5 to -25.0 degrees':
				area = "Southern section"
			else:
				if lat in pst.attrs and lon in pst.attrs:
					area = pst.attrs['lat'] + ' north - ' + pst.attrs['lon'] + ' west'
				elif latitude in pst.attrs and longitude in pst.attrs:
					area = pst.attrs['latitude'] + 'north - ' + pst.attrs['longitude'] + 'west'

			# saving the dataset as a NETCDF path
			net_name = 'persistence_'

			for ix, name in enumerate(pst.attrs['vars by level']):
				if ix == 0:
					name = name[0:4] + '_' + name[5:8] + '_'
				else:
					name = name[0:4] + '_' + name[5:8]

				net_name += name

			net_name += area + '.nc'
			netcdf = pst.to_netcdf(net_name)

		return dataCalc(pst, self.source)


	def blockix(self, datas = None, how = 'continuos', freq = 'daily', save_netcdf = False):
		"""
		Merges the results passed by climsy.dataCalc.persistence method to compute the occurence of a 3-day minimum positive values event, considering all variables related to atmospheric blocking, which are given by the user.
		Its implementation calls climsy.dataCalc.persistence.
		If more than one variable given, both of them must be positive for the counter to start. The first day of atmospheric blocking is computed after the third consecutive day of positive value for both of them at the same time.
		Thus one day of atmospheric blocking occurs only if there are 3 consecutive days of positive values for all variables together.

		Call signature.:

			blockix(*datas, **how, **freq, **save_netcdf)

		*Args:
		----------
		datas: list of <climsy.dataCalc>
		    
			List of datasets.

			Example: datas = [vort_850, div_850] <--> [div_850, vort_850] --> the order doesn't matter.

		**Kwargs
		----------
		how: str, {'continuous', 'total'}
		    
			Continuous how computes every single day; total how returns only the higher number at the end of the series of positive values until they reach a negative value.

			Example: (continuous)
			
			If 1979-01-03 is the third consecutive day of positive vorticity value, for example, one day of blocking is computed.
			If 1979-01-04 is the fourth consecutive day of positive vorticity value, for example, this date receives the number 2 referring to the second day of atmospheric blocking.

			Example: (total)
		        
			Following the previous example, assuming 1979-01-05 shows a negative vorticity value, 1979-01-03 receives 0 (because 1 < 2) and 1979-01-04 receives 2. This method is recommended if the user wants to compile the number of blocking days by frequencies other than daily.

		freq: str, {'daily', 'monthly'}
		    
			Frequency of the series.
		        
			If *freq == 'daily'*, analysis will be daily.
			If *freq == 'monthly'*, analysis will be monthly.

			example showing 'time' dimension of a dataset:
			
			If *freq == 'daily' * --> 1979-01-01, 1979-01-02, 1979-01-03... day by day
			If *freq == 'monthly' * --> 1979-01-01, 1979-02-01, 1979-03-01... month by month

		save_netcdf: bool
			
			Chooses whether a netcdf containing the results will be exported or not.

		Returns
		-------
		obj : <climsy.dataCalc>
		    
			Dataset containing the number of days based on one or more variables.
		"""
		# Creating dict 'blocking' to be converted into a DataArray afterwards
		# Selecting only positive values from indexing dataset returned from 'persistence' --> positive values from persistence (persistence) indicate blocking days
		
		blocking = {}
		positives = self.persistence(datas, how = 'continuos', freq = 'daily', save_netcdf = False).dataset
		new_positives = positives.to_array().dropna('time', 'any')
		block_count = 0

		# All the processes bellow follow the logical framework from persistence --> check the comments up there
		# The difference is that 'blockix' calculates the blocking days based on 'persistence' dataset of persistence
		for ix, day in enumerate(pd.to_datetime(new_positives.time.values)):

			if ix == 0:
				if (all(new_positives.sel(time = day).values) != 0) == True:
					block_count += 1
				else:
					block_count = 0

				blocking.update({day: block_count})

			if (day == pd.to_datetime(new_positives.time.values[ix-1]) + datetime.timedelta(days = 1)) and (day.day != 1):

				if (all(new_positives.sel(time = day).values) != 0) == True:

					block_count += 1

					if how == 'total':
						blocking.update({day - datetime.timedelta(days = 1): 0})
						blocking.update({day: block_count})

					elif how == 'continuos':
						blocking.update({day: block_count})
				else:
					block_count = 0
					blocking.update({day: block_count})

			else:
				if (all(new_positives.sel(time = day).values) != 0) == True:
					block_count = 1
				else:
					block_count = 0

				blocking.update({day: block_count})

		# creating a DataArray from dict 'blocking' and converting into Dataset
		index = xr.DataArray(list(blocking.values()), coords = [list(blocking.keys())], dims = ['time'], name = 'blocking')
		index = index.reindex({'time': pd.date_range(start = positives.time.values[0], end = positives.time.values[-1], freq = 'D')})
		index = index.fillna(0)
		index = index.to_dataset()

		if freq == 'daily':
			# Set the resample frequency and attributes
			index = index
			# keeping attributes from the original dataset
			index.attrs = positives.attrs

		elif freq == 'monthly' and how == 'total':
			# Set the resample frequency and attributes
			index = index.resample(time = 'MS').sum()
			# keeping attributes from the original dataset --> after resample attributes get lost
			index.attrs = positives.attrs

		elif freq == 'seasonal' and how == 'total':
			# Set the resample frequency and attributes
			index = index.resample(time = 'QS-DEC').sum()
			# keeping attributes from the original dataset --> after resample attributes get lost
			index.attrs = positives.attrs

		if save_netcdf == True:

			if index.attrs['lat'] == '-10.0 to -25.0 degrees':
				area = "Total area"
			elif index.attrs['lat'] == '-10.0 to -17.5 degrees':
				area = "Northern section"
			elif index.attrs['lat'] == '-17.5 to -25.0 degrees':
				area = "Southern section"
			else:
				if lat in index.attrs and lon in index.attrs:
					area = index.attrs['lat'] + ' north - ' + index.attrs['lon'] + ' west'
				elif latitude in index.attrs and longitude in index.attrs:
					area = index.attrs['latitude'] + 'north - ' + index.attrs['longitude'] + 'west'

			# saving the dataset as a NETCDF path
			net_name = 'index_'

			for ix, name in enumerate(index.attrs['vars by level']):
				if ix == 0:
					name = name[0:4] + '_' + name[5:8] + '_'
				else:
					name = name[0:4] + '_' + name[5:8]
				net_name += name

			net_name += area + '.nc'
			netcdf = index.to_netcdf(net_name)

		return dataCalc(index, self.source)


def data_view(data, source):
	'''
	Convert xarray.Dataset objects into dataPlot objects.

	Call signature::

		Climsy.data_view(path, source)

	*Args:
	----------
	path : str, Path, file
		
		File directory.

	source : str
		
		Data source.

		- 'ERA-5'
		- 'NCEP'

	Returns
	-------
	obj : <Climsy.dataPlot>

	'''
	# instantiating
	dataset = data

	return dataPlot(dataset, source)

class dataPlot():
	'''
	Plotting class designed for dataCalc objects.

	Call signature ::

		Climsy.dataPlot()

	Attributes
	----------
	singleplot

	multiplots

	season

	bars

	directories

	export

	attributes

	plot

	'''
	def __init__(self, data, source):

		self.dataset = data
		self.source = source
		self.variables = data.variables

		if 'level' in data:
			self.level = self.dataset.level.values

	def attributes(self, name = None, unit = None, clevs = None, cmap = None):
		'''
		Defines dataset attributes for supported variables. If unsupported variable given, returns default settings from the dataset itself.

		'''
		# The arguments are kwargs from data_view methods and allow the user to give desired values as input when calling data_view methods such as multiplots
		# If kwargs are not provided, those are determined automatically from the input.
		dataset = self.dataset
		dataset = dataset.to_array().isel(variable = 0)

		var = str(dataset['variable'].values)
		source = self.source

		# If not provided, predefining KWARGS for known variables
		# Even with predefined values the user can define those by providing the corresponding KWARGS
		if var == 'vort':

			if name == None:
				name = 'Vorticity'
			if cmap == None:
				cmap = 'RdBu_r'

			dataset = dataset * 10**6
			unit = '$10^{-6}$ $s^{-1}$'

		if var == 'div':

			if name == None:
				name = 'Divergence'
			if cmap == None:
				cmap = 'RdBu_r'

			dataset = dataset * 10**6
			unit = '$10^{-6}$ $s^{-1}$'

		elif var == 'hgt' or var == 'z':

			if name == None:
				name = 'Geopotential height'
			if cmap == None:
				cmap = 'Blues'

			if source == 'NCEP':

				dataset = dataset / 10

				if unit == None:
					unit = 'm x 10¹'

				if clevs == None:
					if self.dataset.attrs['level'] == 500:
						clevs = np.arange(586, 592)
					elif self.dataset.attrs['level'] == 700:
						clevs = np.arange(316, 321, 7)
					elif self.dataset.attrs['level'] == 850:
						clevs = np.arange(154, 160, 7)

			elif source == 'ERA-5':

				dataset = dataset / 100

				if unit == None:
					unit = 'm x 10²'

				if clevs == None:
					if self.dataset.attrs['level'] == 500:
						clevs = np.arange(575.5, 578.5, 0.5)
					elif self.dataset.attrs['level'] == 700:
						clevs = np.arange(310, 313.5, 0.5)
					elif self.dataset.attrs['level'] == 850:
						clevs = np.arange(150, 157, 1)

		elif var == 'sst':

			if name == None:
				name = 'Sea surface temperature'
			if cmap == None:
				cmap = 'coolwarm'
			if unit == None:
				unit = '°C'

			if source == 'NCEP':

				dataset = dataset

				if 'base period' in self.dataset.attrs:
					if clevs == None:
						# predefining anomaly clevs
						clevs = np.arange(-2, 2.5, 0.5)
				else:
					if clevs == None:
						clevs = np.arange(-5, 32.5, 2.5)

			elif source == 'ERA-5':

				if 'base period' in self.dataset.attrs:
					# Kelvin to celsius conversion is not needed when analyzing anomaly --> analogous scaling between celsius and kelvin
					dataset = dataset
					# predefining anomaly clevs
					if clevs == None:
						clevs = np.arange(-2, 2.5, 0.5)
				else:
					# Kelvin to celsius conversion
					dataset = dataset - 273

					if clevs == None:
						clevs = np.arange(-5, 32.5, 2.5)

		elif var == 'magnitude':

			if name == None:
				name = 'Wind magnitude'
			if unit == None:
				unit = 'm/s'
			if cmap == None:
				cmap = 'jet_r'
			dataset = dataset

			if clevs == None:
				if 'base period' in self.dataset.attrs:
					# compatible for each level (700, 850, 975, 1000) hPa
					clevs = np.linspace(-6, 6, 13)
				else:
					if self.dataset.attrs['level'] == 975 or self.dataset.attrs['level'] == 1000:
						clevs = np.arange(0, 16.25, 1.25)
					elif self.dataset.attrs['level'] == 700 or self.dataset.attrs['level'] == 850:
						clevs = np.arange(0, 21.25, 1.25)

		elif var == 'uwnd':

			if name == None:
				name = 'Zonal wind'
			if unit == None:
				unit = 'm/s'
			if cmap == None:
				cmap = 'jet_r'

			dataset = dataset

			if clevs == None:
				if 'base period' in self.dataset.attrs:
					# compatible for each level (700, 850, 975, 1000) hPa
					clevs = np.linspace(-3, 3, 9)
				else:
					clevs = np.arange(0, 14, 0.5)

		elif var == 'olr':

			if name == None:
				name = 'Outgoing longwave radiation'
			if unit == None:
				unit = 'W / m²'
			if cmap == None:
				cmap = 'jet'

			if source == 'ERA-5':
				# Due to frame of reference, OLR values are negative for ERA-5 source.
				dataset = dataset*-1
			elif source == 'NCEP':
				# Not needed any conversion for NCEP's dataset
				dataset = dataset

			if clevs == None:
				if 'base period' in self.dataset.attrs:
					clevs = np.arange(-30, 35, 5)
				else:
					clevs = np.arange(180, 305, 5)

		elif var == 'blocking':

			name = 'blocking days'
			unit = ''
			level = None
			cmap = None

		else:
			# If not provided, 'name' and 'unit' are determined automatically from the input.
			dataset = dataset

			if name == None:
				name = var.upper()
			if unit == None:
				if 'units' in self.dataset[var].attrs:
					unit = self.dataset[var].units

		return name, clevs, cmap, unit, dataset


	def series(self, ax, stats = True, series='rainy season', title=None, color=None):
		"""
		Receives the area average of a variable, slicing the time series according to the slice kwarg (see climsy.dataCalc.slicer).
		Returns a plot of the timeseries.

		Call Signature::

			timeseries(*ax, **stats, **series, **title, **color)

		*Args:
		---------
			*ax

		**Kwargs:
		---------
		**stats: bool. Default: True
			
			stats is an optional kwarg that returns the mean value of the time series and +- 2*standard deviation, enlightening higher or lower
			peaks.

		**series: 'all-time', 'rainy season' or list. Default: 'rainy season'
			
			Time-slicing method. (see climsy.dataCalc.slicer)

		**title: str
			
			Title of the axes.

		**color: str
			
			Line color.
		"""
		ds = self.dataset

		#Checking the variable's name for data manipulation
		var = str(ds.to_array().isel(variable = 0)['variable'].values)

		#Checking the area extracting latitude attributes. Refers to atmospheric blocking study. If latitude values
		#are not among the supported values, the area variable receives both longitude and latitude values to be
		#displayed in the plot title
		try:
			if ds.attrs['lat'] == '-10.0 to -25.0 degrees':
				area = "Total area | "
			elif ds.attrs['lat'] == '-10.0 to -17.5 degrees':
				area = "Northern section | "
			elif ds.attrs['lat'] == '-17.5 to -25.0 degrees':
				area = "Southern section | "
			else:
				if lat in ds.attrs and lon in ds.attrs:
					area = ds.attrs['lat'] + ' north - ' + ds.attrs['lon'] + ' west | '
				elif latitude in ds.attrs and longitude in ds.attrs:
					area = ds.attrs['latitude'] + 'north - ' + ds.attrs['longitude'] + 'west | '
		except:
			area = ''

		try:
			level = "at " + str(ds.attrs['level']) + " hPa | "
		except:
			level = ''

		#Setting variable's name, units and scale (if needed).
		#Vorticity and divergence values are exponentially small (~10⁵)

		name, clevs, cmap, unit, dataset = self.attributes()
		ds[var] = dataset

		####################TEMPORARY#############################
		if name.lower() == 'geopotential height':
			name = 'geopotential height anomaly'

		#Resampling daily data to monthly mean reduces noise while preserving information
		dsmean = ds.resample(time="MS").mean()
		dsmean = set_calc(dsmean, self.source).slicer(series=series).dataset

		#Removes level values from the dataset (if exists)
		if 'level' in dsmean:
			dsmean = dsmean.drop('level')
		if 'variable' in dsmean:
			dsmean = dsmean.drop('variable')

		dfmean = dsmean.to_dataframe()
		dfmean.index = dfmean.index.strftime('%Y-%d-%m')

		if ax != None:
			try:
				ax.set_ylabel(name + " (" + unit + ")", fontsize = 25)
			except:
				pass

			if color == None:
				plot_color = 'teal' #cor do gráfico
			elif color != None:
				plot_color = color
			
			ax_color = 'white' #cor do axes
			av_color = 'teal' #cor da linha média
			thresh_color = 'teal' #cor dos limites
			year_color = '#E63946' #análise de ano

			ax.patch.set_color(ax_color) #axes na cor ax_color
			ax.patch.set_alpha(1) #transparência do axes

			ax.plot(dfmean, color = plot_color, label=name)

			if stats == True:
				average = np.mean(dfmean)
				std = np.std(dfmean)
				maximum = average + 2*std
				minimum = average - 2*std

				stats = {'min': minimum.values[0], 'mean': average.values[0], 'max': maximum.values[0]}

				ax.axhline(y = stats['mean'], color = av_color, linestyle = '-', label = 'média')
				ax.axhline(y = stats['max'], color = thresh_color, linestyle = ':', linewidth = 2, label = 'média + 2σ')
				ax.axhline(y = stats['min'], color = thresh_color, linestyle = ':', linewidth = 2, label = 'média - 2σ')

			plt.setp(ax.get_xmajorticklabels(), rotation = 45, ha = 'right', fontsize = 20)
			plt.setp(ax.get_ymajorticklabels(), fontsize = 20)
			ax.grid(True, color = 'grey', alpha = 0.7, axis = 'x', linestyle = '-.', linewidth = 1)

			if series == 'all-time':
				ax.xaxis.set_major_locator(mdates.YearLocator(3))
				years_fmt = mdates.DateFormatter('%Y')
				ax.xaxis.set_major_formatter(years_fmt)
			else:
				start, end = ax.get_xlim()
				ax.xaxis.set_ticks(np.arange(start, end, 8))

			if title == None:
				ax.set_title("Area averaged " + name.lower() + " | " + level + area + series.capitalize() + "\n", fontsize=30, style = 'oblique')
			elif title != None:
				ax.set_title(title + "\n", fontsize=30, style='oblique')

		self.export(plots = "timeseries", ax = ax)

		return ax

	def cumulative(self, series='all-time', level=None, latitude=None, stats=True):
		"""
		Returns a plot containing both anomaly and cumulative anomaly of a given dataset along with its threshold statistics.
		The threshold is given by an average horizontal line inside the plot referring to the mean value of the timeseries, and the maximum and minimum limits given by its mean +- 2 times the standard deviation.
		Timeseries can be sliced to months or known periods (i.e. rainy season).

		Call Signature:

			cumulative(**series, **level, **latitude, **stats)

		**Kwargs:
		---------
		**series: 'all-time', 'rainy season' or list. Default: 'all-time'
			
			Time-slicing method. (see climsy.dataCalc.slicer)

		**level: int. Default: None
			
			Isobaric level of analysis. When cumulative method is called through dataCalc.anom_ts() the level value of the dataset is automatically passed as parameter.

		**latitude: list. Default: None.
			
			Latitude of analysis. When cumulative method is called through dataCalc.anom_ts() the level value of the dataset is automatically passed as parameter.

		**stats: bool. Default: True
			
			Stats is an optional kwarg that returns the mean value of the time series and +- 2*standard deviation, enlightening higher or lower peaks.

		"""
		#converts dataset to dataframe
		anom_df = self.dataset.to_dataframe()

		#if series doesn't correspond to the historical series, the index of the dataframe must be categorical
		#it makes more easy to plot, once it overcomes any gaps along the x axis, avoiding interpolation
		if series != "all-time":
			anom_df.index = anom_df.index.strftime("%Y-%m-%d")
		elif series == 'all-time':
			series = 'timeseries'
		try:
			anom_df.drop("level", axis=1, inplace = True)
		except:
			pass

		#SETTINGS
		bbox = (0, 0.75, 1, 0.5)

		if str(anom_df.columns[0]) == 'hgt':
			scale = 1
			name = 'geopotential height'
		elif str(anom_df.columns[0]) == 'vort':
			scale = 10**4
			name = 'vorticity'
		elif str(anom_df.columns[0]) == 'div':
			scale = 10**4
			name = 'divergence'

		anom_df = anom_df*scale

		corfundo = '#0081A7'
		coranom = '#ED5145'
		av_color = '#004E66' #mean color
		thresh_color = '#004E66' #threshold color

		fig, ax = plt.subplots(figsize=(20, 4))

		#AX
		anom, = ax.plot(anom_df, label='Anomaly', color = corfundo)
		#roll, = ax.plot(anom_df.rolling(3).mean(), color = corfundo, label='Média móvel')
		ax.plot(np.nan, coranom, label = 'Cumulative anomaly')

		#TWINX AX
		ax2 = ax.twinx()
		#calculating cumulative anomaly from the dataframe of anomalies
		anom_df_cusum = np.cumsum(anom_df)
		cusum = ax2.plot(anom_df_cusum, color=coranom, label = 'Cumulative anomaly')

		#STATISTICS
		if stats == True:
			average = np.mean(anom_df)
			std = np.std(anom_df)
			maximum = average + 2*std
			minimum = average - 2*std
			stats = {'min': minimum.values[0], 'mean': average.values[0], 'max': maximum.values[0]}
			ax.axhline(y=stats['mean'], color=av_color, linestyle="-", label = 'mean')
			ax.axhline(y=stats['max'], color=thresh_color, linestyle=":", linewidth = 2, label='mean + 2σ')
			ax.axhline(y=stats['min'], color=thresh_color, linestyle=":", linewidth = 2, label='mean - 2σ')

		#PLOT DESIGN
		if anom_df.index.dtype == 'object':
			start, end = ax2.get_xlim()
			ax2.xaxis.set_ticks(np.arange(start, end, 8))

		plt.setp(ax.get_xmajorticklabels(), rotation=45, ha="right", fontsize=10)
		ax.legend(loc='upper left', bbox_to_anchor=bbox, fontsize=15, fancybox=True, framealpha=0.5, mode='expand', ncol=5)
		ax.tick_params(axis='y', colors='black')
		ax2.tick_params(axis='y', colors=coranom)
		ax.set_ylabel(str(anom_df.columns[0]) + ' anomaly (x '+ "{:.0e}".format(scale) + ')', color='black', fontsize = 12)
		ax2.set_ylabel('Cumulative anomaly (x '+ "{:.0e}".format(scale) + ')', fontsize = 12, labelpad = 20, color=coranom, rotation=-90)
		ax2.grid(False)
		ax.grid(axis='x', color = 'gainsboro')

		ax.set_title(name.capitalize() + ' on ' + level + ' hPa | ' + str(int(abs(latitude[0]))) + 'S to ' + str(int(abs(latitude[1]))) + 'S' + " | " + series.capitalize() + " | " + self.dataset.attrs['source'] + '\n\n', fontsize = 20, style="oblique" )
		fig.tight_layout()

		self.export(plots = "timeseries", ax = ax)


	def plot(self, ax, freq, clevs, cmap, unit, dataset, month=None, season=None, xlocator=None, ylocator=None, subplot_title_size=20, cbar_labelpad=20, ticks_size=10, cbar_ticks_size=15, cbar_label_size=15, cbar_pad=0.15):
		'''
		Geoaxes plot.
		'''
		# Clevs, cmap and unit are kwargs from attributes method, but arguments from plot method.
		# Formatting plots according to frequency provided
		if freq == 'monthly':

			var, lonu = add_cyclic_point(dataset.sel(month = month), coord = dataset['lon'][:])

			if clevs != None:
				cf = ax.contourf(lonu, dataset['lat'][:], var, clevs, cmap = cmap, extend = 'both')
			else:
				cf = ax.contourf(lonu, dataset['lat'][:], var, cmap = cmap, extend = 'both')

			cbar = plt.colorbar(cf, orientation = 'horizontal', pad = cbar_pad, ax = ax, shrink = 1.1, aspect = 40)
			cbar.ax.tick_params(labelsize = cbar_ticks_size)

			if unit != None:
				cbar.set_label(f'{unit}', fontsize = cbar_label_size, labelpad = cbar_labelpad, style = 'oblique')

			gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 1, color = 'black', alpha = 0.3, linestyle = '--')
			gl.xlabels_top = False
			gl.ylabels_left = True
			gl.ylabels_right = False
			gl.ylines = True
			gl.xlines = True
			gl.xformatter = LONGITUDE_FORMATTER
			gl.yformatter = LATITUDE_FORMATTER

			if xlocator == None:
				gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
			else:
				gl.xlocator = mticker.FixedLocator(xlocator)

			if ylocator == None:
				gl.ylocator = mticker.FixedLocator(np.arange(-180, 180, 5))
			else:
				gl.ylocator = mticker.FixedLocator(ylocator)

			gl.xlabel_style = {'size': ticks_size}
			gl.ylabel_style = {'size': ticks_size}

			ax.coastlines('50m')
			ax.get_extent(crs = ccrs.PlateCarree())
			ax.set_title(calendar.month_abbr[month], fontdict = {'fontsize': subplot_title_size}, loc = 'right', style = 'oblique')

		elif freq == 'seasonal':

			# user can provide string or int for season, but the plotting procedures only receive int.
			# Converting strings into int
			if type(season) != int:
				if season == 'DJF':
					season_index = 0
				elif season == 'JJA':
					season_index = 1
				elif season == 'MAM':
					season_index = 2
				elif season == 'SON':
					season_index = 3
			else:
				season_index = season

			# Clevs, cmap and unit are kwargs from attributes method, but arguments from plot method.
			# Formatting plots according to frequency provided
			var, lonu = add_cyclic_point(dataset.isel(season = season_index), coord = dataset['lon'][:])

			if clevs != None:
				cf = ax.contourf(lonu, dataset['lat'][:], var, clevs, cmap = cmap, extend = 'both')
			else:
				cf = ax.contourf(lonu, dataset['lat'][:], var, cmap = cmap, extend = 'both')

			cbar = plt.colorbar(cf, orientation = 'horizontal', pad = cbar_pad, ax = ax, shrink = 1.1, aspect = 40)
			cbar.ax.tick_params(labelsize = cbar_ticks_size)

			if unit != None:
				cbar.set_label(f'{unit}', fontsize = cbar_label_size, labelpad = cbar_labelpad, style = 'oblique')

			gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 1, color = 'black', alpha = 0.3, linestyle = '--')
			gl.xlabels_top = False
			gl.ylabels_left = True
			gl.ylabels_right = False
			gl.ylines = True
			gl.xlines = True
			gl.xformatter = LONGITUDE_FORMATTER
			gl.yformatter = LATITUDE_FORMATTER

			if xlocator == None:
				gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
			else:
				gl.xlocator = mticker.FixedLocator(xlocator)

			if ylocator == None:
				gl.ylocator = mticker.FixedLocator(np.arange(-180, 180, 5))
			else:
				gl.ylocator = mticker.FixedLocator(ylocator)

			gl.xlabel_style = {'size': ticks_size}
			gl.ylabel_style = {'size': ticks_size}

			ax.coastlines('50m')
			ax.get_extent(crs = ccrs.PlateCarree())
			ax.set_title(self.dataset.season[season_index].values, loc = 'left', fontsize = subplot_title_size, style = 'oblique')


	def multiplots(self, name = None, unit = None, clevs = None, cmap = None, export = False, xlocator = None, ylocator = None, figsize = None, suptitle_size = 27, hspace = 0.15, wspace = 0.2, subplot_title_size = 20, cbar_labelpad = 20, ticks_size = 10, cbar_ticks_size = 15, cbar_label_size = 15, cbar_pad = 0.15):
		'''
		Plots a figure with subplots for every month of the year.

		Call signature:

			self.multiplots(**kwargs)

		**Kwargs
		--------
		name: str

			Figure title. If the user doesn't give a string of choice to be displayed as title, tries to infer by supported variables. If unsupported, returns name straight from the dataset, displaying the array name as title.
			
			Check supported variables through Climsy's help method.

			Supported variables and its titles:
			* sst --> Sea Surface Temperature
			* olr --> Outgoing longwave radiation
			* uwnd --> Vento zonal
			* magnitude --> Wind magnitude
			* div --> Divergence
			* vort --> Vorticity
			* hgt --> Geopotential height

		unit: str

			Units for analyzed variable to be placed at the colorbar. Should be given if scaling operations where made when formatting the dataset, explicitly indicating that mathematical operations where performed. If **unit = None** (default) and the unit is unknown by Climsy, tries to infer the unit straight from dataset attributes, except there are no unit data within the dataset (text label blank on colorbar).
			
			Check supported variables through Climsy's help method.

			Supported variables and its units:
			* sst --> °C
			* olr --> W / m²
			* uwnd --> m/s
			* magnitude --> m/s
			* div --> 10^{-6} s^{-1}
			* vort --> 10^{-6} s^{-1}
			* hgt --> m x 10¹ (if source == 'NCEP') / m x 10² (if source == 'ERA-5')

		clevs: array or list

			Receives a numpy array or list as limits and steps for the color levels of the contour fill. The smaller the step, the bigger the precision for analysis. If **clevs == None** (default), limits and steps are inferred.

			Usage Example:
			
			* np.arange(start, end + step, step) --> generates an array from **start** to **end** by **step**

			Ex: np.arange(1, 4, 0.5) --> [1, 1.5, 2, 2.5, 3, 3.5]

			* np.linspace(start, end, number_of_steps) --> generates an array from start to end but with more control of the number of steps.

			Ex: np.linspace(1, 3.5, 6) --> [1, 1.5, 2, 2.5, 3, 3.5]

		cmap : str

			Color map for each variable plot according to matplotlib's palettes. If **cmap = None**, cmap receives default palettes from matplotlib or those setted at attributes method, according to the variable.
			
			Check supported variables through Climsy's help method.

			Supported variables and its suggested palettes:
			* sst --> 'coolwarm'
			* olr --> 'jet'
			* uwnd --> 'jet_r'
			* magnitude --> 'jet_r'
			* div --> 'RdBu_r'
			* vort --> 'RdBu_r'
			* hgt --> 'blues'

			Although there are suggested cmaps, the user can also explicitly give a desired one as input (cmap = 'cmap_name').

		export : bool

			Save figure in png format (see 'export' method). If 'True', the png file is directed to a predefined directory. If 'False' (default), this method doesn't export.

		xlocator, ylocator: list

			List of possible locations for ticks. Stands [-180, 180, 10] for xlocator and [-180, 180, 5] for ylocator by default. Check <matplotlib.ticker.FixedLocator>.

			Example: 
			
			xlocator = [-180, 180, 10] --> step of 10 --> xcoordinate's ticks varying from 10° to 10°
			ylocator = [-180, 180, 5] --> step of 5 --> ycoordinate's ticks varying from 5° to 5°

		figsize : (float, float)

			Size of figure (Width & height in inches). If not provided, stands (32,22) by Default. Check <matplotlib.pyplot.figure>.

		suptitle_size: float or str

			Font size of figure's suptitle. In points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 27 by default. Check <matplotlib.pyplot.suptitle> and <matplotlib.text.Text>.

		hspace: float

			The amount of height reserved for space between subplots, expressed as a fraction of the average axis height. If not provided, stands 0.15 by default. Check <matplotlib.pyplot.subplots_adjust>.

		wspace: float

			The amount of width reserved for space between subplots, expressed as a fraction of the average axis width. If not provided, stands 0.20 by default. Check <matplotlib.pyplot.subplots_adjust>.

		subplot_title_size: float or str

			Font size of subplot's title. In points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 20 by default. Check <matplotlib.pyplot.title> and <matplotlib.text.Text>.

		cbar_labelpad: int

			Space between colobar and it's label. If not provided, stands 20 by Default. Check <matplotlib.axes.Axes.set_label>.

		ticks_size: float or str

			Tick label font size in points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 10 by Default. Check <matplotlib.text.Text> and <cartopy.mpl.gridliner.Gridliner>.

		cbar_ticks_size: float or str

			Colorbar's tick label font size in points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 15 by Default. Check <matplotlib.axes.Axes.tick_params>.

		cbar_label_size: int

			Colorbar's label font size in points. If not provided, stands 15 by Default. Check <matplotlib.axes.Axes.set_label>.

		cbar_pad: float

		 	Fraction of original axes between colorbar and new image axes. If not provided, stands 0.15 by Default. Check <matplotlib.pyplot.colorbar>.

		'''
		# defining figure and subplots
		if figsize == None:
			fig, ax = plt.subplots(nrows = 3, ncols = 4, subplot_kw = dict(projection = ccrs.PlateCarree()), figsize = (32,22))
		else:
			fig, ax = plt.subplots(nrows = 3, ncols = 4, subplot_kw = dict(projection = ccrs.PlateCarree()), figsize = figsize)

		# adjusting space between subplots
		plt.subplots_adjust(hspace = hspace, wspace = wspace)

		# naming 'plots' path (check 'export' and 'directories' functions)
		plots = 'plots by year'

		# attributes for 'plot' function
		name, clevs, cmap, unit, dataset = self.attributes(name, unit, clevs, cmap)

		# time attribute
		time = [self.dataset.attrs['period'][0:7], self.dataset.attrs['period'][14:21]]

		# creating titles according to Attributes
		if self.dataset.attrs['analysis'] == 'climatology':

			if 'level' in self.dataset.attrs:
				plt.suptitle(f"{name} on {self.dataset.attrs['level']} hPa\n\nClimatology from {time[0][:4]} to {time[1][:4]}" + " | " + self.source, fontsize=suptitle_size, weight="medium",style="oblique",va="top", ha="center")
			else:
				plt.suptitle(f"{name}\n\nClimatology from {time[0][:4]} to {time[1][:4]}" + " | " + self.source, fontsize=suptitle_size, weight="medium",style="oblique",va="top", ha="center")

		# creating titles according to Attributes
		elif self.dataset.attrs['analysis'] == 'averages':

			if 'level' in self.dataset.attrs:
				plt.suptitle(f"{name} on {self.dataset.attrs['level']} hPa\n\nMonthly averages from {time[0][:4]}/{time[0][5:7]} to {time[1][:4]}/{time[1][5:7]}"  + " | " + self.source, fontsize=suptitle_size, weight="medium",style="oblique",va="top", ha="center")
			else:
				plt.suptitle(f"{name}\n\nMonthly averages from {time[0][:4]}/{time[0][5:7]} to {time[1][:4]}/{time[1][5:7]}" + " | " + self.source, fontsize=suptitle_size, weight="medium",style="oblique",va="top", ha="center")

		# creating titles according to Attributes
		elif self.dataset.attrs['analysis'] == 'anomaly':
			# basetime attribute for anomaly
			basetime = [self.dataset.attrs['base period'][0:7], self.dataset.attrs['base period'][14:21]]

			if 'level' in self.dataset.attrs:
				plt.suptitle(f"{name} on {self.dataset.attrs['level']} hPa\n\nAnomaly averaged from " + time[0][:4] + "/" + time[0][5:7] + " to " + time[1][:4] + "/" + time[1][5:7] + " relative to the long term averages from " + basetime[0][:4] + " to " + basetime[1][:4] + " | " + self.source, fontsize=suptitle_size, weight="medium",style="oblique",va="top", ha="center")
			else:
				plt.suptitle(f"{name}\n\nAnomaly averaged from " + time[0][:4] + "/" + time[0][5:7] + " to " + time[1][:4] + "/" + time[1][5:7] + " relative to the long term averages from " + basetime[0][:4] + " to " + basetime[1][:4]  + " | " + self.source, fontsize=suptitle_size, weight="medium",style="oblique",va="top", ha="center")


		# list of months for axes loop
		month = np.arange(1, 13)

		# defining the axes for subplots
		k = 0
		for i in range(0, 3):
			for j in range(0, 4):
				try:
					if type(clevs) != list:
						self.plot(ax[i,j], 'monthly', clevs.tolist(), cmap, unit, dataset, month[k], None, xlocator, ylocator, subplot_title_size, cbar_labelpad, ticks_size, cbar_ticks_size, cbar_label_size, cbar_pad)
					else:
						self.plot(ax[i,j], "monthly", clevs, cmap, unit, dataset, month[k], None, xlocator, ylocator, subplot_title_size, cbar_labelpad, ticks_size, cbar_ticks_size, cbar_label_size, cbar_pad)
				except:
					self.plot(ax[i,j], "monthly", clevs, cmap, unit, dataset, month[k], None, xlocator, ylocator, subplot_title_size, cbar_labelpad, ticks_size, cbar_ticks_size, cbar_label_size, cbar_pad)
				k += 1

		# exporting image for directories (check 'export' and 'directories' functions)
		if export == True:
			self.export(plots)


	def seasons(self, name = None, unit = None, clevs = None, cmap = None, export = False, xlocator = None, ylocator = None, figsize = None, suptitle_size = 15, hspace = 0.2, wspace = 0.2, subplot_title_size = 20, cbar_labelpad = 20, ticks_size = 10, cbar_ticks_size = 15, cbar_label_size = 15, cbar_pad = 0.15):
		'''
		Seasonal plot for each trimester of the year.

		Call signature:

			self.seasons(**kwargs)

		**Kwargs
		--------
		name: str

			Figure title. If the user doesn't give a string of choice to be displayed as title, tries to infer by supported variables. If unsupported, returns name straight from the dataset, displaying the array name as title.
			
			Check supported variables through Climsy's help method.

			Supported variables and its titles:
			* sst --> Sea Surface Temperature
			* olr --> Outgoing longwave radiation
			* uwnd --> Vento zonal
			* magnitude --> Wind magnitude
			* div --> Divergence
			* vort --> Vorticity
			* hgt --> Geopotential height

		unit: str

			Units for analyzed variable to be placed at the colorbar. Should be given if scaling operations where made when formatting the dataset, explicitly indicating that mathematical operations where performed. If **unit = None** (default) and the unit is unknown by Climsy, tries to infer the unit straight from dataset attributes, except there are no unit data within the dataset (text label blank on colorbar).
			
			Check supported variables through Climsy's help method.

			Supported variables and its units:
			* sst --> °C
			* olr --> W / m²
			* uwnd --> m/s
			* magnitude --> m/s
			* div --> 10^{-6} s^{-1}
			* vort --> 10^{-6} s^{-1}
			* hgt --> m x 10¹ (if source == 'NCEP') / m x 10² (if source == 'ERA-5')

		clevs: array or list

			Receives a numpy array or list as limits and steps for the color levels of the contour fill. The smaller the step, the bigger the precision for analysis. If **clevs == None** (default), limits and steps are inferred.

			Usage Example:
			
			* np.arange(start, end + step, step) --> generates an array from **start** to **end** by **step**

			Ex: np.arange(1, 4, 0.5) --> [1, 1.5, 2, 2.5, 3, 3.5]

			* np.linspace(start, end, number_of_steps) --> generates an array from start to end but with more control of the number of steps.

			Ex: np.linspace(1, 3.5, 6) --> [1, 1.5, 2, 2.5, 3, 3.5]

		cmap : str

			Color map for each variable plot according to matplotlib's palettes. If **cmap = None**, cmap receives default palettes from matplotlib or those setted at attributes method, according to the variable.
			
			Check supported variables through Climsy's help method.

			Supported variables and its suggested palettes:
			* sst --> 'coolwarm'
			* olr --> 'jet'
			* uwnd --> 'jet_r'
			* magnitude --> 'jet_r'
			* div --> 'RdBu_r'
			* vort --> 'RdBu_r'
			* hgt --> 'blues'

			Although there are suggested cmaps, the user can also explicitly give a desired one as input (cmap = 'cmap_name').

		export : bool

			Save figure in png format (see 'export' method). If 'True', the png file is directed to a predefined directory. If 'False' (default), this method doesn't export.

		xlocator, ylocator: list

			List of possible locations for ticks. Stands [-180, 180, 10] for xlocator and [-180, 180, 5] for ylocator by default. Check <matplotlib.ticker.FixedLocator>.

			Example: 
			
			xlocator = [-180, 180, 10] --> step of 10 --> xcoordinate's ticks varying from 10° to 10°
			ylocator = [-180, 180, 5] --> step of 5 --> ycoordinate's ticks varying from 5° to 5°

		figsize : (float, float)

			Size of figure (Width & height in inches). If not provided, stands (20,12) by Default. Check <matplotlib.pyplot.figure>.

		suptitle_size: float or str

			Font size of figure's suptitle. In points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 27 by default. Check <matplotlib.pyplot.suptitle> and <matplotlib.text.Text>.

		hspace: float

			The amount of height reserved for space between subplots, expressed as a fraction of the average axis height. If not provided, stands 0.15 by default. Check <matplotlib.pyplot.subplots_adjust>.

		wspace: float

			The amount of width reserved for space between subplots, expressed as a fraction of the average axis width. If not provided, stands 0.20 by default. Check <matplotlib.pyplot.subplots_adjust>.

		subplot_title_size: float or str

			Font size of subplot's title. In points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 20 by default. Check <matplotlib.pyplot.title> and <matplotlib.text.Text>.

		cbar_labelpad: int

			Space between colobar and it's label. If not provided, stands 20 by Default. Check <matplotlib.axes.Axes.set_label>.

		ticks_size: float or str

			Tick label font size in points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 10 by Default. Check <matplotlib.text.Text> and <cartopy.mpl.gridliner.Gridliner>.

		cbar_ticks_size: float or str

			Colorbar's tick label font size in points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 15 by Default. Check <matplotlib.axes.Axes.tick_params>.

		cbar_label_size: int

			Colorbar's label font size in points. If not provided, stands 15 by Default. Check <matplotlib.axes.Axes.set_label>.

		cbar_pad: float

		 	Fraction of original axes between colorbar and new image axes. If not provided, stands 0.15 by Default. Check <matplotlib.pyplot.colorbar>.

		'''
		# naming 'plots' path (check 'export' and 'directories' functions)
		plots = 'seasonal plots'

		# defining figure and subplots
		if figsize == None:
			fig, ax = plt.subplots(ncols = 2, nrows = 2, subplot_kw = dict(projection = ccrs.PlateCarree()), figsize = (20, 12))
		else:
			fig, ax = plt.subplots(ncols = 2, nrows = 2, subplot_kw = dict(projection = ccrs.PlateCarree()), figsize = figsize)

		# adjusting space between subplots
		plt.subplots_adjust(hspace = hspace, wspace = wspace)

		# attributes for 'plot' function
		name, clevs, cmap, unit, dataset = self.attributes(name, unit, clevs, cmap)
		# time attribute
		time = [self.dataset.attrs['period'][0:7], self.dataset.attrs['period'][14:21]]

		# creating titles according to Attributes
		if self.dataset.attrs['analysis'] == 'climatological seasonality':

			if 'level' in self.dataset.attrs:
				plt.suptitle(f'{name} on {self.dataset.attrs["level"]} hPa\n\nClimatology from {str(int(time[0][0:4]))} to {time[1][:4]}', fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')

			else:
				plt.suptitle(f'{name}\n\nClimatology from {str(int(time[0][0:4]))} to {time[1][:4]}', fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')

		# creating titles according to Attributes
		elif self.dataset.attrs['analysis'] == 'seasonal averages':

			if 'level' in self.dataset.attrs:
				plt.suptitle(f'{name} on {self.dataset.attrs["level"]} hPa\n\nSeasonal averages from {str(int(time[0][0:4]))} to {time[1][:4]}', fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')

			else:
				plt.suptitle(f'{name}\n\nSeasonal averages from {str(int(time[0][0:4]))} to {time[1][:4]}', fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')

		# creating titles according to Attributes
		elif self.dataset.attrs['analysis'] == 'anomaly':

			basetime = [self.dataset.attrs['base period'][0:7], self.dataset.attrs['base period'][0:7]]

			if 'level' in self.dataset.attrs:
				plt.suptitle(f"{name} on {self.dataset.attrs['level']} hPa\n\nAnomaly averaged from " + str(int(time[0][0:4])) + " to " + time[1][:4] + " relative to the seasonal averages from " + str(int(basetime[0][0:4]) + 1) + " to " + basetime[1][:4] + " | " + self.source, fontsize=suptitle_size, weight="medium",style="oblique",va="top", ha="center")
			else:
				plt.suptitle(f"{name}\n\nAnomaly averaged from " + str(int(time[0][0:4])) + " to " + time[1][:4] + " relative to the seasonal averages from " + str(int(basetime[0][0:4]) + 1) + " to " + basetime[1][:4] + " | " + self.source, fontsize=suptitle_size, weight="medium",style="oblique",va="top", ha="center")

		# defining the axes for subplots
		k = 0
		for i in range (0, 2):
			for j in range(0, 2):
				if clevs != None:
					if type(clevs) != list:
						self.plot(ax[i,j], 'seasonal', clevs.tolist(), cmap, unit, None, k, xlocator, ylocator, subplot_title_size, cbar_labelpad, ticks_size, cbar_ticks_size, cbar_label_size, cbar_pad)
					else:
						self.plot(ax[i,j], 'seasonal', clevs, cmap, unit, dataset, None, k, xlocator, ylocator, subplot_title_size, cbar_labelpad, ticks_size, cbar_ticks_size, cbar_label_size, cbar_pad)
				else:
					self.plot(ax[i,j], 'seasonal', clevs, cmap, unit, dataset, None, k, xlocator, ylocator, subplot_title_size, cbar_labelpad, ticks_size, cbar_ticks_size, cbar_label_size, cbar_pad)
				k += 1

		# exporting image for directories (check function 'export' and 'directories')
		if export == True:
			self.export(plots)


	def singleplot(self, freq, month = None, season = None, name = None, unit = None, clevs = None, cmap = None, export = False, xlocator = None, ylocator = None, figsize = None, suptitle_size = 15, subplot_title_size = 20, cbar_labelpad = 20, ticks_size = 10, cbar_ticks_size = 15, cbar_label_size = 15, cbar_pad = 0.15):
		'''
		A single plot ment for a single month analysis.

		Call signature:

			self.singleplot(freq, month, season, **kwargs)

			Ex: Whole year loop (12 months):

			--> months = np.arange(1, 13)
			--> month in months:
			-->    self.singleplot(freq, month, season, **kwargs)

		*Args:
		----------
		freq: {'monthly', 'seasonal'}

			Selects the type of time analysis. Stands for 'monthly' by default.

		month: int in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

			Month of analysis from 1 (January) to 12 (December). It is only necessary if 'monthly' is given for 'freq'.

		season: int in [0, 1, 2, 3] or str in ['DJF', 'JJA', 'MAM', 'SON']

			Season of analysis. If given an int, it is convert into string following this dict: {'DJF': 0, 'JJA': 1, 'MAM': 2, 'SON': 3}. It is only necessary if 'seasonal' is given for 'freq'.

		**Kwargs
		--------
		name: str

			Figure title. If the user doesn't give a value to be displayed as title, tries to infer by supported variables. If unsupported, returns name straight from the dataset, displaying the array name as title.
			
			Check supported variables through Climsy's help method.

			Supported variables and its titles:
			* sst --> Sea Surface Temperature
			* olr --> Outgoing longwave radiation
			* uwnd --> Vento zonal
			* magnitude --> Wind magnitude
			* div --> Divergence
			* vort --> Vorticity
			* hgt --> Geopotential height

		unit: str

			Units for analyzed variable to be placed at the colorbar. Should be given if scaling operations where made when formatting the dataset, explicitly indicating that mathematical operations where performed. If **unit = None** (default) and the unit is unknown by Climsy, tries to infer the unit straight from dataset attributes, except there are no unit data within the dataset (text label blank on colorbar).
			
			Check supported variables through Climsy's help method.

			Supported variables and its units:
			* sst --> °C
			* olr --> W / m²
			* uwnd --> m/s
			* magnitude --> m/s
			* div --> 10^{-6} s^{-1}
			* vort --> 10^{-6} s^{-1}
			* hgt --> m x 10¹ (se source == 'NCEP') / m x 10² (se source == 'ERA-5')

		clevs: array or list

			Receives a numpy array or list as limits and steps for the color levels of the contour fill. The smaller the step, the bigger the precision for analysis. If **clevs == None** (default), limits and steps are inferred.

			Usage Example:
			
			* np.arange(start, end + step, step) --> generates an array from **start** to **end** by **step**

			Ex: np.arange(1, 4, 0.5) --> [1, 1.5, 2, 2.5, 3, 3.5]

			* np.linspace(start, end, number_of_steps) --> generates an array from start to end but with more control of the number of steps.

			Ex: np.linspace(1, 3.5, 6) --> [1, 1.5, 2, 2.5, 3, 3.5]

		cmap : str

			Color map for each variable plot according to matplotlib's palettes. If **cmap = None**, cmap receives default palettes from matplotlib or those setted at attributes method, according to the variable.
			
			Check supported variables through Climsy's help method.

			Supported variables and its suggested palettes:
			* sst --> 'coolwarm'
			* olr --> 'jet'
			* uwnd --> 'jet_r'
			* magnitude --> 'jet_r'
			* div --> 'RdBu_r'
			* vort --> 'RdBu_r'
			* hgt --> 'blues'

			Although there are suggested cmaps, the user can also explicitly give a desired one as input (cmap = 'cmap_name').

		export : bool

			Save figure in png format (see 'export' method). If 'True', the png file is directed to a predefined directory. If 'False' (default), this method doesn't export.

		xlocator, ylocator: list

			List of possible locations for ticks. Stands [-180, 180, 10] for xlocator and [-180, 180, 5] for ylocator by default. Check <matplotlib.ticker.FixedLocator>.

			Example: 
			
			xlocator = [-180, 180, 10] --> step of 10 --> xcoordinate's ticks varying from 10° to 10°
			ylocator = [-180, 180, 5] --> step of 5 --> ycoordinate's ticks varying from 5° to 5°

		figsize : (float, float)

			Size of figure (Width & height in inches). If not provided, stands (16, 12) by Default. Check <matplotlib.pyplot.figure>.

		suptitle_size: float or str

			Font size of figure's suptitle. In points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 27 by default. Check <matplotlib.pyplot.suptitle> and <matplotlib.text.Text>.

		subplot_title_size: float or str

			Font size of subplot's title. In points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 20 by default. Check <matplotlib.pyplot.title> and <matplotlib.text.Text>.

		cbar_labelpad: int

			Space between colobar and it's label. If not provided, stands 20 by Default. Check <matplotlib.axes.Axes.set_label>.

		ticks_size: float or str

			Tick label font size in points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 10 by Default. Check <matplotlib.text.Text> and <cartopy.mpl.gridliner.Gridliner>.

		cbar_ticks_size: float or str

			Colorbar's tick label font size in points or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}. If not provided, stands 15 by Default. Check <matplotlib.axes.Axes.tick_params>.

		cbar_label_size: int

			Colorbar's label font size in points. If not provided, stands 15 by Default. Check <matplotlib.axes.Axes.set_label>.

		cbar_pad: float

		 	Fraction of original axes between colorbar and new image axes. If not provided, stands 0.15 by Default. Check <matplotlib.pyplot.colorbar>.

		'''
		# naming 'plots' path (check 'export' and 'directories' functions) according to frequency provided
		if freq == 'monthly':
			plots = 'plots by month'
		elif freq == 'seasonal':
			plots = 'plots by season'

		# defining figure and subplots
		if figsize == None:
			fig, ax = plt.subplots(subplot_kw = dict(projection = ccrs.PlateCarree()), figsize = (16, 12))
		else:
			fig, ax = plt.subplots(subplot_kw = dict(projection = ccrs.PlateCarree()), figsize = figsize)

		# attributes for 'plot' function
		name, clevs, cmap, unit, dataset = self.attributes(name, unit, clevs, cmap)
		# time attribute
		time = [self.dataset.attrs['period'][0:7],self.dataset.attrs['period'][14:21]]

		# creating titles according to Attributes
		if self.dataset.attrs['analysis'] == 'climatology':

			if 'level' in self.dataset.attrs:
				plt.suptitle(f'{name} on {self.dataset.attrs["level"]} hPa\n\nClimatology from {time[0][:4]} to {time[1][:4]}', fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')
			else:
				plt.suptitle(f'{name}\n\nClimatology from {time[0][:4]} to {time[1][:4]}', fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')

		# creating titles according to Attributes
		elif self.dataset.attrs['analysis'] == 'averages':

			if 'level' in self.dataset.attrs:
				plt.suptitle(f'{name} on {self.dataset.attrs["level"]} hPa\n\nMonthly average from {time[0][:4]}/{time[0][5:7]} to {time[1][:4]}/{time[1][5:7]}', fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')

			else:
				plt.suptitle(f'{name}\n\nMonthly average from {time[0][:4]}/{time[0][5:7]} to {time[1][:4]}/{time[1][5:7]}', fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')

		# creating titles according to Attributes
		elif self.dataset.attrs['analysis'] == 'anomaly':
			basetime = [self.dataset.attrs['base period'][0:7], self.dataset.attrs['base period'][14:21]]

			if 'level' in self.dataset.attrs:
				plt.suptitle(f'{name} on {self.dataset.attrs["level"]} hPa\n\nAnomaly averaged from ' + time[0][:4] + '/' + time[0][5:7] + ' to ' + time[1][:4] + '/' + time[1][5:7] + ' relative to the long term average from ' + basetime[0][:4] + ' to ' + basetime[1][:4], fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')

			else:
				plt.suptitle(f'{name}\n\nAnomaly averaged from ' + time[0][:4] + '/' + time[0][5:7] + ' to ' + time[1][:4] + '/' + time[1][5:7] + ' relative to the long term average from ' + basetime[0][:4] + ' to ' + basetime[1][:4], fontsize = suptitle_size, weight = 'medium',style = 'oblique',va = 'top', ha = 'center')

		if clevs != None:
			if type(clevs) != list:
				self.plot(ax, freq, clevs.tolist(), cmap, unit, dataset, month, season, xlocator, ylocator, subplot_title_size, cbar_labelpad, ticks_size, cbar_ticks_size, cbar_label_size, cbar_pad)
			else:
				self.plot(ax, freq, clevs, cmap, unit, dataset, month, season, xlocator, ylocator, subplot_title_size, cbar_labelpad, ticks_size, cbar_ticks_size, cbar_label_size, cbar_pad)
		else:
			self.plot(ax, freq, clevs, cmap, unit, dataset, month, season, xlocator, ylocator, subplot_title_size, cbar_labelpad, ticks_size, cbar_ticks_size, cbar_label_size, cbar_pad)

		if export == True:
			# exporting image for directories (check function 'export' and 'directories') according to frequency provided
			if freq == 'seasonal':
				self.export(plots, None, season)

			elif freq == 'monthly':
				self.export(plots, month, None)


	def barplot(self, ax, coords, time, basetime, basetime_2 = None, level = None):

		# Gráfico comparativo de barras
		# Gráfico pode comparar até 3 períodos ao Mesmo tempo
		# coords = [LONGITUDE, LATITUDE] -> respeitar esta ordem
		# O KWARG 'LEVEL' SÓ É NECESSÁRIO CASO O OBJETO.DATASET NÃO ESTEJA RECORTADO PARA O VALOR 'LEVEL'
		# SE O OBJETO.DATASET ESTIVER RECORTADO (recortado no formatter(), por ex), O VALOR DE 'LEVEL' É BUSCADO DIRETAMENTE NO .DATASET

		name, clevs, cmap, unit, dataset = self.attributes()
		month = []
		for i in range(1, 13):
			month.append(calendar.month_abbr[i])

		data = self.dataset.sel(lat = coords[1], lon = coords[0])

		# INSTANCIANDO OBJETO dataCalc PARA PODER CALCULAR A climatology PELA FUNÇÃO 'climatology' DA CLASSE 'dataCalc' (!!)
		obj_medias = set_calc(data, self.source)

		x_bar = np.arange(1, 13, 1)

		if 'level' in self.dataset:
			if level == None:
				level = int(self.level)
				y_bar_1 = (obj_medias.climatology(time)).dataset.to_array().isel(variable = 0)
				y_bar_2 = (obj_medias.climatology(basetime)).dataset.to_array().isel(variable = 0)
			else:
				y_bar_1 = (obj_medias.climatology(time, level)).dataset.to_array().isel(variable = 0)
				y_bar_2 = (obj_medias.climatology(basetime, level)).dataset.to_array().isel(variable = 0)
		else:
			y_bar_1 = (obj_medias.climatology(time)).dataset.to_array().isel(variable = 0)
			y_bar_2 = (obj_medias.climatology(basetime)).dataset.to_array().isel(variable = 0)

		if basetime_2 != None:
		# Gráfico com 3 barras mensais (comparando 3 períodos)
			if 'level' in self.dataset:
				if level == None:
					level = int(self.level)
					y_bar_3 = (obj_medias.climatology(time)).dataset.to_array().isel(variable = 0)
				else:
					y_bar_3 = (obj_medias.climatology(basetime_2, level)).dataset.to_array().isel(variable = 0)
			else:
				y_bar_3 = (obj_medias.climatology(time)).dataset.to_array().isel(variable = 0)

			label = [f'{time[0][:4]} a {time[1][:4]}', f'{basetime[0][:4]} a {basetime[1][:4]}', f'{basetime_2[0][:4]} a {basetime_2[1][:4]}']
			width = 0.26
			bar1 = ax.bar(x_bar, y_bar_1, color = 'lightsalmon', width = width, label = label[0], align = 'center', edgecolor = 'white')
			bar2 = ax.bar(x_bar - (width + 0.05), y_bar_2, color = 'cornflowerblue', width = width, label = label[1], align = 'center', edgecolor = 'white')
			bar3 = ax.bar(x_bar + (width + 0.05), y_bar_3, color = 'mediumspringgreen', width = width, label = label[2], align = 'center', edgecolor = 'white')

		else:
		# Gráfico com 2 barras mensais (comparando 2 períodos)
			label = [f'{time[0][:4]} a {time[1][:4]}', f'{basetime[0][:4]} a {basetime[1][:4]}']
			width = 0.3
			bar1 = ax.bar(x_bar, y_bar_1, color = 'lightsalmon', width = width, label = label[0], align = 'edge', edgecolor = 'white', linewidth = 2.5)
			bar2 = ax.bar(x_bar, y_bar_2, color = 'cornflowerblue', width = width*-1, label = label[1], align = 'edge', edgecolor = 'white', linewidth = 2.5)

		ax.set_xticks(x_bar)
		ax.set_xticklabels(month)
		ax.set_ylabel(f'{unit}', labelpad = 15, fontsize = 15, style = 'oblique')
		ax.set_xlabel('Mês', labelpad = 15, fontsize = 15, x = 0.5, style = 'oblique')
		ax.set_title('LAMMOC-UFF', fontsize = 8, loc = 'left', style = 'italic')
		ax.legend(bbox_to_anchor = (0.858, 1.045), loc = 'upper center', ncol = 3, prop = font_manager.FontProperties(style = 'oblique', size = 9), fancybox = True, shadow = True)

		def autolabel(bars):

			# Função que plota os valores em y exatos acima de cada barra (com 2 casa decimais)
			for bar in bars:
				height = bar.get_height()
				x = Decimal(height)
				y = round(x, 2)
				ax.annotate(f'{y}',
							xy = (bar.get_x() + bar.get_width() / 1.8, y),
							xytext = (0, 12),
							textcoords = 'offset points',
							ha = 'center', va = 'top', fontsize = 8)

		autolabel(bar1)
		autolabel(bar2)

		if basetime_2 != None:
			autolabel(bar3)


	def bars(self, coords, time, basetime, basetime_2 = None, level = None, export = False):
		'''
		Função que gera um gráfico comparativo de barras, recebendo no mínimo *dois* e no máximo *três* períodos para comparação. As séries de dados são comparadas mês a mês.

		Call signature:

			self.bars(coords, time, basetime, basetime_2, level, export)

		*Args:
		----------
		coords : list, len(list == 2)

			Recebe a lista de coordenadas -- na ordem [longitude, latitude] -- do local onde está sendo feita a análise. O parâmetro recebe coordenadas exatas (len(list) == 2) e não àquelas equivalentes a uma região (len(list) == 4).

			Ex.: coords = [-55, -5] (longitude: -55 W, latitude: -5 S) --> coordenadas exatas --> CORRETO
			Ex.: coords = [-55, -45, -5, -10] (intervalo de longitude e de latitude) --> coordenadas de uma região --> ERRADO

		time : list, str

			Intervalo temporal do *primeiro* período de análise.

			Ex: time = ['2005-01-01', '2010-12-01']

		basetime: list, str

			Intervalo temporal do *segundo* período de análise.

			Ex: time = ['2005-01-01', '2010-12-01'] e basetime = ['1979-01-01', '2010-12-01'].

		**Kwargs
		--------
		level: int

			Caso os dados contenham dimensão 'level', é necessário especificar para qual valor de level a análise está sendo gerada. O valor numérico inteiro é convertido em string e especificado no título da figura.

			Ex: 'Wind magnitude comparison on 850 hPa' --> título da figura , onde '850': str(level)

		export : bool

			Salvar a figura (formato '.png') em pastas e diretórios com caminhos pré-definidos pela biblioteca (conferir docstring da função 'export'). Se 'True', a figura é salva na pasta final definida. Se 'False' (Padrão) a imagem não é salva.

			Nessa função, a imagem gerada será salva na seguinte sequência de pastas:

			'climatologyS' -> 'name_banco_de_dados' -> 'name_variável' -> 'bar plots' -> 'name_imagem.png'

			OBS.: name da imagem varia de acordo com os valores fornecidos para o *parâmetro* 'time' e 'basetime' e para os *kwargs* 'basetime_2' e 'level'
			(Conferir docstring da função 'export')

		'''
		# Função que define a figura, título e subplot do gráfico de barras
		# Se o usuário quiser comparar 3 período ao Mesmo tempo ----> DEFINIR O PARÂMETRO 'basetime2' (Default: None)

		plots = 'bar plots'
		name, clevs, cmap, unit, dataset = self.attributes()
		fig, ax = plt.subplots(figsize = (16, 7))

		if 'level' in self.dataset:
			if level == None:
				if 'level' in self.dataset:
					level_title = int(self.level)
				else:
					level_title = level

			plt.suptitle(f'{name} Comparison on {level_title} hPa', ha = 'center', va = 'top', weight = 'bold', fontsize = 20, y = 1.05)

		else:
			plt.suptitle(f'{name} Comparison', ha = 'center', va = 'top', weight = 'bold', fontsize = 20, y = 1.05)

		if basetime_2 != None:
			plt.title(f'Monthly Averages for {time[0][:4]} to {time[1][:4]} relative to the Monthly Averages for {basetime[0][:4]} to {basetime[1][:4]} and for {basetime_2[0][:4]} to {basetime_2[1][:4]}', fontsize = 14, y = 1.06, x = 0.485)

		else:
			plt.title(f'Monthly Averages for {time[0][:4]} to {time[1][:4]} relative to the Monthly Averages for {basetime[0][:4]} to {basetime[1][:4]}', fontsize = 14, y = 1.06, x = 0.475)

		self.barplot(ax, coords, time, basetime, basetime_2, level)

		# melhor ajuste para a figura
		fig.tight_layout()

		if export != False:
			if basetime_2 != None:
				self.export('climatology', plots, time, None, level, None)
			else:
				self.export('climatology', plots, time, basetime, level, None)


	def directories(self, array, plots):
		'''
		Creates directories following the order: data_source > variable > number_of_plots > saved_image

		Example:
			
			anomaly > ERA-5 > uwnd > seasonal plots > figname.png

		obs: figname is returned by dataPlot.export() method. (See export)

		'''
		filepath = os.path.join(os.getcwd(), array) #where array stands for analysis. ex: anomaly, climatology, etc...

		dataset = self.dataset
		dataset = dataset.to_array().isel(variable = 0)

		var = str(dataset['variable'].values)
		source = self.source

		try:
		#at first, directories try to create a path in the same directory as the running code, followed by the analysis
			os.mkdir(filepath)
		
		except FileExistsError:
		#if the directory already exists, move on to next loop
			pass

		try:
		#tries to add a source directory to the path. Supported sources are 'NCEP' and 'ERA-5'
			os.mkdir(os.path.join(filepath, self.source))
		
		except FileExistsError:
		#if path already exists, pass
			pass

		path = os.path.join(filepath, self.source, var)

		try:
		#tries to add the variable name to the path
			os.mkdir(path)
		
		except FileExistsError:
		#unless it already exists too
			pass

		path = os.path.join(filepath, self.source, var, plots)

		try:
		#at last, tries to create a directory with all data specifications so the user can easily access data output
			os.mkdir(path)
		
		except FileExistsError:
			pass

		#returns full path where image will be saved in
		return path


	def export(self, plots, month=None, season=None, ax=None):
		'''
		Saves and exports figure to predefined directories:

		'array' --> 'source' --> 'variable' --> 'plot type' --> 'image_name.png', where 'ARRAY' and 'PLOTS' are arguments.

		OBS.: The image name ('image_name.png') feature may vary according to the variable and its attributes.

		OBS_2.: Folders are saved on the same path as the one from which the .py file is running.

		Example: 
		
			if 'filename.py' ou 'filename.ipynb' (jupyter notebook) is running on *C:/Users/name_user/variable*, all folders will be saved there as well.

		Call signature:

			self.export(array, plots, time, basetime, level, month)

			OBS.: This method is executed through **export = True** kwarg of plotting methods, which automates exporting processes.

		*Args:
		----------
		plots: {'bar plots', 'plots by year', 'seasonal plots', 'plots by month'}

			Type of analysis. When export function is called within plotting fuctions (**export = True**), plot is automatically inferred.

		**Kwargs
		--------
		month: int in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

			Month of analysis, It is received as int, but displayed as a string containing the name of the month through calendar's month_abbr method. (Ex: 3 --> 'MAR').

			The name of the month is written on the file name. (see above)

			Ex: olr_79_10_Mar, where 'olr': variable, '79_10': abbreviated period and 'Mar': str(month)

		season: int in [0, 1, 2, 3] or str in ['DJF', 'JJA', 'MAM', 'SON']

			Season of analysis. If given an int, it is convert into string following this dict: {'DJF': 0, 'JJA': 1, 'MAM': 2, 'SON': 3}

			Ex: olr_79_10_DJF, where 'olr': variable, '79_10': abbreviated period and 'DJF': season (or str(0) if it is an int)

		'''
		#creates a directory based on analysis type and plot type
		path = self.directories(self.dataset.attrs['analysis'], plots)

		dataset = self.dataset
		dataset = dataset.to_array().isel(variable = 0)

		var = str(dataset['variable'].values)

		#checking which trimester data is referencing if seasonal
		if season != None:
			if type(season) != int:
				if season == 'DJF':
					season = 0
				elif season == 'JJA':
					season = 1
				elif season == 'MAM':
					season = 2
				elif season == 'SON':
					season = 3
			else:
				season = season

		#if we're dealing with anomalous data, this block ensures the base period enters figname
		if self.dataset.attrs['analysis'] == 'anomaly':

			if month != None:
			#if data refers to monthy anomaly

				if 'level' in self.dataset.attrs: #and if there's an isobaric level in its attributes
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + self.dataset.attrs['base period'][2:4] + '_' + self.dataset.attrs['base period'][16:18] + '_' + calendar.month_abbr[month] + '_' + str(self.dataset.attrs["level"]) + '.png'), bbox_inches = 'tight')

				else: #data without level supported because of SST and MSLP variables
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + self.dataset.attrs['base period'][2:4] + '_' + self.dataset.attrs['base period'][16:18] + '_' + calendar.month_abbr[month] + '.png'), bbox_inches = 'tight')

			elif season != None:
			#if data refers to seasonal anomaly

				if 'level' in self.dataset.attrs:
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + self.dataset.attrs['base period'][2:4] + '_' + self.dataset.attrs['base period'][16:18] + '_' + str(self.dataset.season[season].values) + '_' + str(self.dataset.attrs["level"]) + '.png'), bbox_inches = 'tight')

				else:
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + self.dataset.attrs['base period'][2:4] + '_' + self.dataset.attrs['base period'][16:18] + '_' + str(self.dataset.season[season].values) + '.png'), bbox_inches = 'tight')

			else:
			#if data is not seasonal or monthly anomaly, export assumes it is yearly

				if 'level' in self.dataset.attrs:
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + self.dataset.attrs['base period'][2:4] + '_' + self.dataset.attrs['base period'][16:18] + '_' + str(self.dataset.attrs["level"]) + '.png'), bbox_inches = 'tight')

				else:
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + self.dataset.attrs['base period'][2:4] + '_' + self.dataset.attrs['base period'][16:18] + '.png'), bbox_inches = 'tight')

		elif self.dataset.attrs['analysis'] == 'anomaly timeseries':
			plt.savefig(os.path.join(path, ax.get_title() + ".png"), bbox_inches="tight")


		else:
		#if data has no 'base period', we're dealing with averaged data or climatology
		#all following conditions share the same specifications with the conditions above

			if month != None:

				if 'level' in self.dataset.attrs:
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + calendar.month_abbr[month] + '_' + str(self.dataset.attrs["level"]) + '.png'), bbox_inches = 'tight')

				else:
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + calendar.month_abbr[month] + '.png'), bbox_inches = 'tight')

			elif season != None:

				if 'level' in self.dataset.attrs:
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + str(self.dataset.season[season].values) + '_' + str(self.dataset.attrs["level"]) + '.png'), bbox_inches = 'tight')

				else:
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + str(self.dataset.season[season].values) + '.png'), bbox_inches = 'tight')

			else:

				if 'level' in self.dataset.attrs:
					plt.savefig(os.path.join(path, var + '_' + self.dataset.attrs['period'][2:4] + '_' + self.dataset.attrs['period'][16:18] + '_' + str(self.dataset.attrs["level"]) + '.png'), bbox_inches = 'tight')

				else:
					plt.savefig(os.path.join(path, var + "_" + self.dataset.attrs['period'][2:4] + "_" + self.dataset.attrs['period'][16:18] + ".png"), bbox_inches="tight")


def normalize(ds):
	'''
	Receives a xarray.Dataset object and returns a normalized one.
	'''
	minimum = []
	maximum = []

	for index in ds.data_vars:
		minimum.append(ds.data_vars[index].values.min())
		maximum.append(ds.data_vars[index].values.max())
		datamin = np.min(minimum)
		datamax = np.max(maximum)

	norm = (ds-datamin)/(datamax-datamin)

	return norm
