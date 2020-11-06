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

def dataset(path, bd):

	'''
	Transforma objetos 'xarray.Dataset' em objetos 'lammoc_dataset'.

	Call signature::

		nb.dataset(path, bd) , onde 'nb' é abreviação para 'newBlock'

	Parameters
	----------
	path : str, Path, file, list
		Endereço do arquivo NetCDF.
		
		OBS.: Aceita lista de 2 valores, retornando dois objetos 'lammoc_dataset'.
			
		Exemplo:
			
		dataset_1, dataset_2 = nb.dataset(['C:/arquivo_1.nc', 'C:/arquivo_2.nc'], bd)

	bd : str
		Fonte de dados utilizada.
		
		- 'ERA-5' 
		- 'NCEP'

	Returns
	-------
	obj : lammoc_dataset
	
	'''

	if type(path) is list and len(path) == 2:
		dados_1 = xr.open_dataset(path[0])
		dados_2 = xr.open_dataset(path[1])

		return lammoc_dataset(dados_1, bd) , lammoc_dataset(dados_2, bd)
	else:
		dados = xr.open_dataset(path) 

		return lammoc_dataset(dados, bd)

class lammoc_dataset:
	'''
	Classe responsável pela formatação, operações com múltiplas variáveis (magnitude, divergência e vorticidade) e padronização dos dados.
	É recomendado que sua instância seja feita pela função 'nb.dataset()'.
	
	Call signature ::
		
		nb.lammoc_dataset(dados, bd), onde 'nb' é abreviação para 'newBlock'
		
	Parameters
	----------
	dados : xarray.open_dataset(path)
		Exige o uso da função 'xarray.open_dataset' com o endereço do arquivo NetCDF em seus parâmetros.
		
	bd : str
		Fonte de dados utilizada.
		
		- 'ERA-5' 
		- 'NCEP'
	
	Exemplo: nb.lammoc_dataset(xarray.open_dataset('C:/arquivo_1.nc'), 'NCEP'
	
	OBS.: É fortemente recomendado o uso da função 'nb.dataset()' para a instância, de outro modo é necessária a chamada da função 'xarray.open_dataset()'.
	
	Attributes
	----------
	dataset: 'xarray.dataset'
		Dados no formato 'xarray.dataset', facilita a visualização dos arrays.
	
	variables: 'xarray.core.utils.Frozen'
		Acesso às variáveis do dataset.
		
	bd: str
		Fonte de dados do dataset. ('ERA-5 ou 'NCEP)
	
	new_plot
	
	formatar
	'''


	def __init__(self, dados, bd):

		if bd == 'ERA-5':
			self.dataset = dados
			self.variables = dados.variables
			self.bd = bd
			self.new_plot = new_plot(dados, bd)
					 
			if "expver" in self.dataset.coords:
			#eliminando as variações do expver para os dados ERA-5
				self.dataset = self.dataset.isel(expver=0)   
				self.dataset = self.dataset.drop('expver')

			# padronizando latitude, longitude, U e V do ERA-5 para ficar igual aos do NCEP (lat, lon, uwnd, vwnd)
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

		elif bd == 'NCEP':
			self.dataset = dados
			self.variables = dados.variables
			self.new_plot = new_plot(dados, bd)
			self.bd = bd
			
		
	def formatar(self, opr=None, fator=None, data_2=None, level=None, lat=None, lon=None, time=None):
		'''
		Função que possibilita recorte dos dados, operações com múltiplas variáveis e operações de divisão e multiplicação.
		Retorna um objeto 'lammoc_medias'.
		
		Call signature:: 
		
			self.formatar(**opr=None,
			**fator=None,
			**data_2=None, 
			**level=None,
			**lat=None,
			**lon=None,
			**time=None)
						
			
		Todos os parâmetros da função são opcionais, sua seleção define o objetivo da função.
		
		Para as operações com múltiplas variáveis, é possível que seja utilizado um 'lammoc_dataset' com as duas variáveis, assim como também é possível utilizar um 'lammoc_dataset' com determinada variável e chamar outro 'lammoc_dataset' no kwarg '**data_2' com a variável complementar independente da ordem.
		
		Exemplo:  Chamando a função com 2 objetos separados e uma variável em cada
			
			dataset_Vwind.formatar(data_2 = dataset_Uwind, opr = 'magnitude')
			
													---------------								
			
		Exemplo:  Chamando a função com 1 objeto com as 2 variáveis
			
			dataset_Uwind_e_Vwind.formatar(opr = 'magnitude')
			
													---------------
			
		**Kwargs
		----------
		data_2 : lammoc_dataset	
			Objeto complementar necessário para operações com múltiplas variáveis (magnitude, divergência, vorticidade) caso estejam em objetos 'lammoc_dataset' diferentes.
		
			Exemplo:
			
			dataset_Vwind.formatar(data_2 = dataset_Uwind, opr = 'magnitude')
		
		opr : str
			Seleciona o tipo de operação, podendo ser 'magnitude', 'vorticidade', 'divergência', 'divisão' ou 'multiplicação'
			
		fator: int or float
			No caso de operações de divisão ou multiplicação, é necessário selecionar o fator da operação.
			
		level : int
			Recorta o dataset para o level escolhido.
			
		lat : list
			Recorta o dataset para a latitude escolhida.
			
			Exemplo:
			
			lammoc_dataset.formatar(lat = [10, -50])
			
		lon : list
			Recorta o dataset para a longitude escolhida.
			
			Exemplo:
			
			lammoc_dataset.formatar(lon = [-60, 10])
			
		time: list
			Recorta o dataset para o tempo escolhido.
			
			Exemplo:
			
			lammoc_dataset.formatar(time = ['1979-01-01', '2010-12-01'])
			

		Returns
		-------
		obj : lammoc_medias
		
		'''
		# Data_2 recebe outro objeto lammoc_dataset (!!)
		# Função que formata os dados originais em datasets "base" para outras classes e operações
		# Permite recortes simultâneos de diferentes variáveis
		databases = []
		data_base = self.dataset
		databases.append(data_base)
		if data_2 != None:
			data_base_2 = data_2.dataset
			databases.append(data_base_2)
		
		datas = []
		for data in databases:
		# Recortando o(s) dataset(s)
			if level != None:
				data = data.sel(level=level)
			if lat != None:
				if type(lat) == list:
					data = data.sel(lat=slice(lat[0], lat[1]))
				else:
					data = data.sel(lat=lat)  
			if lon != None:
				if self.bd == "NCEP":
					if type(lon) == list:
						if lon[0] > lon [1]:
							w = data.sel(lon=slice(lon[0], 360))
							e = data.sel(lon=slice(0, lon[1]))
							e['lon'] = e['lon'] + 360
							data = xr.merge([e,w])
						else:
							data = data.sel(lon=slice(lon[0], lon[1]))
					else:
						data = data.sel(lon=lon)
				else:
					if type(lon) == list:
						data = data.sel(lon=slice(lon[0], lon[1]))
					else:
						data = data.sel(lon=lon)
			if time != None:
				if type(time) == list:
					data = data.sel(time=slice(time[0], time[1]))
				else:
					data = data.sel(time=time)
					
			datas.append(data)
				
		if opr == "magnitude" or opr == "mag":
		# Formatando o dataset para magnitude
			if len(datas) == 1:
				data = datas[0]
				data["magnitude"] = (data.uwnd**2 + data.vwnd**2)**0.5
				#magnitude = data.drop(["uwnd", "vwnd"]).to_dataset(name='magnitude')
				magnitude = data.drop(["uwnd", "vwnd"])
			else:
				if 'uwnd' in datas[0]:
					dataset = ((datas[0].uwnd)**2 + (datas[1].vwnd)**2)**0.5
					magnitude = dataset.to_dataset(name='magnitude')
				else:
					dataset = ((datas[1].uwnd)**2 + (datas[0].vwnd)**2)**0.5
					magnitude = dataset.to_dataset(name='magnitude')
					
			return new_medias(magnitude, self.bd)
		
		elif opr == "divergência" or opr == "div":
		# Formatando o dataset para divergencia
			if len(datas) == 1:
				dataset = datas[0]
				d = []
				dx, dy = mpcalc.lat_lon_grid_deltas(dataset.variables['lon'][:], dataset.variables['lat'][:])

				for i, data in enumerate(dataset.variables['time'][:]):
					div = mpcalc.divergence(dataset.uwnd.isel(time=i),
															dataset.vwnd.isel(time=i),
															dx, dy, dim_order='yx')
					d.append(xr.DataArray(div.m,
								  dims=['lat', 'lon'],
								  coords={'lat': dataset.variables['lat'][:], 'lon': dataset.variables['lon'][:], 'time': dataset.variables['time'][:][i]}, name = 'div'))

				divergencia = xr.concat(d, dim = 'time').to_dataset()
		
			else:
				d = []
				dx, dy = mpcalc.lat_lon_grid_deltas(datas[0].variables['lon'][:], datas[0].variables['lat'][:])

				for i, data in enumerate(datas[0].variables['time'][:]):
					div = mpcalc.divergence(datas[0].uwnd.isel(time=i),
															datas[1].vwnd.isel(time=i),
															dx, dy, dim_order='yx')
					d.append(xr.DataArray(div.m,
								  dims=['lat', 'lon'],
								  coords={'lat': datas[0].variables['lat'][:], 'lon': datas[0].variables['lon'][:], 'time': datas[0].variables['time'][:][i]}, name = 'div'))

				divergencia = xr.concat(d, dim = 'time').to_dataset()
				
			return new_medias(divergencia, self.bd)
		
		elif opr == "vorticidade" or opr == "vort":
		# Formatando o dataset para vorticidade
			if len(datas) == 1:
				dataset = datas[0]
				v = []
				dx, dy = mpcalc.lat_lon_grid_deltas(dataset.variables['lon'][:], dataset.variables['lat'][:])

				for i, data in enumerate(dataset.variables['time'][:]):
					div = mpcalc.vorticity(dataset.uwnd.isel(time=i),
															dataset.vwnd.isel(time=i),
															dx, dy, dim_order='yx')
					v.append(xr.DataArray(vort.m,
								  dims=['lat', 'lon'],
								  coords={'lat': dataset.variables['lat'][:], 'lon': dataset.variables['lon'][:], 'time': dataset.variables['time'][:][i]}, name = 'vort'))

				vorticidade = xr.concat(v, dim = 'time').to_dataset()
		
			else:
				v = []
				dx, dy = mpcalc.lat_lon_grid_deltas(datas[0].variables['lon'][:], datas[0].variables['lat'][:])

				for i, data in enumerate(datas[0].variables['time'][:]):
					vort = mpcalc.vorticity(datas[0].uwnd.isel(time=i),
															datas[1].vwnd.isel(time=i),
															dx, dy, dim_order='yx')
					v.append(xr.DataArray(vort.m,
								  dims=['lat', 'lon'],
								  coords={'lat': datas[0].variables['lat'][:], 'lon': datas[0].variables['lon'][:], 'time': datas[0].variables['time'][:][i]}, name = 'vort'))

				vorticidade = xr.concat(v, dim = 'time').to_dataset()
			
			return new_medias(vorticidade, self.bd)
		
		elif opr == "divisão" or opr == "/":
			data = datas[0]
			new_data = data / fator
			return new_medias(new_data, self.bd)
		
		elif opr == "multiplicação" or opr == "x":
			data = datas[0]
			new_data = data * fator
			return new_medias(new_data, self.bd)
				
		else:
			data_base = datas[0]
			return new_medias(data_base, self.bd)

def new_medias(dados, bd):
	
	# instanciando objeto lammoc_medias
	dataset = dados
	return lammoc_medias(dataset, bd)
		
		
class lammoc_medias(lammoc_dataset):  # Herança simples da classe lammoc_dataset ---> OBJETOS terão os MESMOS atributos.
	'''
	Classe responsável pelas operações de médias com dataset; climatologias, levantamentos, anomalias, medias sazonais, medias regionais, etc...
	A classe é 'lammoc_medias' é  uma subclasse da 'lammoc_dataset', portanto herda todos os seus atributos.
	
	Call signature ::
		
		nb.lammoc_medias(dados, bd), onde 'nb' é abreviação para 'newBlock'
		
	Parameters
	----------
	dados : xarray.open_dataset(path)
		Exige o uso da função 'xarray.open_dataset' com o endereço do arquivo NetCDF em seus parâmetros.
		
	bd : str
		Fonte de dados utilizada.
		
		- 'ERA-5' 
		- 'NCEP'
	
	Exemplo: nb.lammoc_dataset(xarray.open_dataset('C:/arquivo_1.nc'), 'NCEP'
	
	
	Attributes
	----------
	dataset: 'xarray.dataset'
		Dados no formato 'xarray.dataset', facilita a visualização dos arrays.
	
	variables: 'xarray.core.utils.Frozen'
		Acesso às variáveis do dataset.
		
	bd: str
		Fonte de dados do dataset. ('ERA-5 ou 'NCEP)
	
	new_plot
	
	formatar
	'''

	def climatologia(self, periodo=None, level=None):  
	
		'''
		Função que calcula a climatologia dados o intervalo de tempo e level.
		Se o 'dataset' já estiver recortado (pela função formatar(), por exemplo), não há necessidade de explicitar periodo e level.
		Caso seja apontado um novo recorte, a função retornará a climatologia em cima deste.
		
		Call signature :: 
		
			self.climatologia(**periodo, **level)
		
		**Kwargs
		----------
		periodo: list
			Recorta o dataset para o tempo escolhido.
			
		level: int
			Recorta o dataset para o level escolhido.
			
		Returns
		----------
		obj : lammoc_medias
		
		'''
		dataset = self.dataset
		if periodo != None:
			dataset = dataset.sel(time=slice(periodo[0], periodo[1]))
		if level != None:
			dataset = dataset.sel(level=level)
			
		climatologia = dataset.groupby("time.month").mean()
		
		return lammoc_medias(climatologia, self.bd)
	
	def anomalia(self, periodo, periodobase, level=None):
		'''
		Função que calcula a anomalia selecionados o intervalo de tempo analisado e o período base.
		
		Call signature :: 
		
			self.anomalia(periodo, periodobase, **level=None)
			
		Parameters
		----------
		periodo: list
			Período escolhido para análise.
			
		periodobase: list
			Período relativo à base que está sendo comparada.
			
		**Kwargs
		----------
		level: int
			Recorta o dataset para o level escolhido.
			
		Exemplo:
		
			self.anomalia(periodo = ['2016-01-01', '2018-01-01'], periodobase = ['1979-01-01', '2010-01-01'])
			
		Returns
		----------
		obj : lammoc_medias
		'''
	
		climatologia = self.climatologia(periodo, level)
		climabase = self.climatologia(periodobase, level)
			
		anomalia = climatologia.dataset - climabase.dataset

		return lammoc_medias(anomalia, self.bd)

	def media_regional(self, latitude=None, longitude=None, periodos=None, level=None):
		
		valores = []
		var = self.dataset
		if level != None:
			var = var.sel(level=level)
		if periodos != None:
			var = var.sel(time=slice(periodos[0], periodos[1]))
		if latitude != None:
			var = var.sel(lat=slice(latitude[0], latitude[1]))
		if longitude != None:
			var = var.sel(lon=slice(longitude[0], longitude[1]))
		for i, data in enumerate(var.time.values):
			valores.append(var.isel(time=i).map(np.mean))
		ds = xr.concat(valores, dim='time')
		return lammoc_medias(ds, self.bd)
	
	def saz(self, periodo=None):
		
		# Nesta função o dataset(ds) original é recortado para o período de análise e retorna uma LISTA DE DATASETS e UM DATASET
		# A lista 'data_list' retornada contém os datasets.season de cada intervalo de UM ANO dentro do RANGE do 'periodo de recorte'
		# EX:: ASSUMINDO QUE 'PERIODO' É [1979-12-01, 2010-11-01] <--> 'DATA_LIST[0]' retornará o dataset.season do intervalo de ..
		# ... (Continuando)... >> [1979-12-01 até 1980-11-01] <<< EQUIVALENTE AO INTERVALO UM ANO (DATA_LIST[0])
	
		data_list = []	  
		tempo = []
		data = self.dataset
		if periodo != None:
			data = data.sel(time=slice(periodo[0], periodo[1]))
		for i in range(0, len(data.time.values), 12):
			d_slice = data.isel(time=slice(i, i+12))
			d_season = d_slice.groupby("time.season").mean()
			data_list.append(d_season)
			tempo.append(pd.to_datetime(data.time.values[i]).year)
		idx = pd.Index(tempo, name="years")
		ds = xr.concat(data_list, dim=idx)
		
		return lammoc_medias(ds, self.bd)

	def sazclima(self, periodo=None):
		
		# O dataset 'data_season' retornado é o dataset.season de TODO o RANGE do 'periodo de recorte'
		# Logo, o 'data_season' é a média.SEASON do intervalo completo do dataset recortado
		# EX:: ASSUMINDO QUE PERIODO >> [1979-12-01, 2010-11-01] << --> 'DATA_SEASON' retornará o dataset.season do PERIODO

		data = self.dataset
		if periodo != None:
			data = data.sel(time=slice(periodo[0], periodo[1]))
		data_season = data.groupby("time.season").mean()
		
		return lammoc_medias(data_season, self.bd)
	
def new_plot(dados, bd):
	
	'''
	Transforma objetos 'xarray.Dataset' em objetos 'lammoc_plot'.

	Call signature::

		nb.new_plot(path, bd), onde 'nb' é abreviação para 'newBlock'

	Parameters
	----------
	path : str, Path, file
		Endereço do arquivo NetCDF.
		
	bd : str
		Fonte de dados utilizada.
		
		- 'ERA-5' 
		- 'NCEP'

	Returns
	-------
	obj : lammoc_plot
	
	'''
	# Instanciando objeto lammoc_plot
	dataset = dados
	return lammoc_plot(dataset, bd)
	
class lammoc_plot():
	'''
	Classe responsável pela geração de imagens e gráficos dos dados.
	É recomendado que sua instância seja feita pelo atributo 'lammoc_medias.new_plot', pois a maioria das funções só fazem sentido a partir desse objeto.
	
	Call signature ::
		
		nb.lammoc_plot(), onde 'nb' é abreviação para 'newBlock'
			
	Attributes
	----------
	singleplot
	
	multiplots
	
	season
	
	bars
	
	diretorios
	
	exportar
		
	atributos
	
	plot
	'''	
	
	def __init__(self, dados, bd):
		
		self.dataset = dados
		self.bd = bd
		self.variables = dados.variables
		if "level" in dados:
			self.level = self.dataset.level.values
	
	def atributos(self, periodobase=None, level=None, nome=None, unidade=None, clevs=None, cmap=None):
		
		# Função para definir os atributos do dataset
		# Função já conta com atributos fixos a variáveis 'conhecidas'.
		# Caso a varíavel seja 'desconhecida', os atributos seguem os valores default.
		# Os args aqui são **kwargs das funções finais de plot (multiplots, por ex)  (!!!)
		# Os **kwargs permitem ao usuário formatar seu subplot, caso a variável utilizada não tenha atributos fixos
		# Tais kwargs serão definidos apenas nas funções finais de plot (multiplots, por exemplo)
		
		dataset = self.dataset
		dataset = dataset.to_array().isel(variable=0)
		var = str(dataset["variable"].values)
		bd = self.bd
		
		if var == 'vort':
			if nome == None:
				nome = 'Vorticity'
			if cmap == None:
				cmap = "RdBu_r"
			dataset = dataset * 10**6
			unidade = "$10^{-6}$ $s^{-1}$"
		
		if var == 'div':
			if nome == None:
				nome = 'Divergence'
			if cmap == None:
				cmap = "RdBu_r"
			dataset = dataset * 10**6
			unidade = "$10^{-6}$ $s^{-1}$"
			
		elif var == "hgt" or var == "z":
			
			if nome == None:
				nome = "Geopotential height"
			if cmap == None:
				cmap = "Blues"
			
			if bd == "NCEP":
				dataset = dataset / 10
				if unidade == None:
					unidade = "m x 10¹"
				if clevs == None:
					if level == 500:
						clevs = np.arange(586, 592)
					elif level == 700:
						clevs = np.arange(316, 321, 7)
					elif level == 850:
						clevs = np.arange(154, 160, 7)
					
			elif bd == "ERA-5":
				dataset = dataset / 100
				if unidade == None:
					unidade = "m x 10²"
				if clevs == None:
					if level == 500:
						clevs = np.arange(575.5, 578.5, 0.5)
					elif level == 700:
						clevs = np.arange(310, 313.5, 0.5)
					elif level == 850:
						clevs = np.arange(150, 157, 1)

		elif var == "sst":
			
			if nome == None:
				nome = "Sea surface temperature"
			if cmap == None:
				cmap = "coolwarm"
			if unidade == None:
				unidade = "°C"
			
			if bd == "NCEP":
				dataset = dataset
				if periodobase != None:
					if clevs == None:
						clevs = np.arange(-2, 2.5, 0.5)
				else:
					if clevs == None:
						clevs = np.arange(-5, 32.5, 2.5)
					
			elif bd == "ERA-5":
			# Forma de definir Clevs das ANOMALIAS --> (periodobase != None)
				if periodobase != None:  
					dataset = dataset
					if clevs == None:
						clevs = np.arange(-2, 2.5, 0.5)
				else:
				# Conversão para Kelvin para os dados do ERA-5
				# A conversão NÃO é necessária quando fizer ANOMALIA --> variação em Kelvin <-> variação em Celsius
					dataset = dataset - 273  
					if clevs == None:
						clevs = np.arange(-5, 32.5, 2.5)

		elif var == "magnitude":
			
			if nome == None:
				nome = "Wind magnitude"
			if unidade == None:
				unidade = "m/s"
			if cmap == None:
				cmap = "jet_r"
			dataset = dataset
			
			if clevs == None:
				if periodobase != None:
					# coerente para todo level (700, 850, 975, 1000) hPa
					clevs = np.linspace(-6, 6, 13)  
				else:
					if level == 975 or level == 1000:
						clevs = np.arange(0, 16.25, 1.25)
					elif level == 700 or level == 850:
						clevs = np.arange(0, 21.25, 1.25)

		elif var == "uwnd":
			
			if nome == None:
				nome = "Zonal wind"
			if unidade == None:
				unidade = "m/s"
			if cmap == None:
				cmap = "jet_r"
			dataset = dataset
			
			if clevs == None:
				if periodobase != None:
				# coerente para todo level (700, 850, 975, 1000) hPa
					clevs = np.linspace(-3, 3, 9)	 
				else:
					clevs = np.arange(0, 14, 0.5)

		elif var == "olr":
			
			if nome == None:
				nome = "Outgoing longwave radiation"
			if unidade == None:
				unidade = "W / m²"
			if cmap == None:
				cmap = "jet"
			
			if bd == "ERA-5":
			# Por conta de referencial, valores de OLR no ERA-5 estão negativos
				dataset = dataset*-1 
			elif bd == "NCEP":
				dataset = dataset
				
			if clevs == None:
				if periodobase != None:
					clevs = np.arange(-30, 35, 5) 
				else:
					clevs = np.arange(180, 305, 5)
		
		else:
			dataset = dataset
			
			if nome == None:
				nome = var.upper()
			if cmap != None:
				cmap = cmap
			if clevs != None:
				clevs = clevs
			if unidade == None:
				if "units" in self.dataset[var].attrs:
					unidade = self.dataset[var].units
			
		return nome, clevs, cmap, unidade, dataset
	
	def plot(self, ax, kind, clevs, cmap, unidade, dataset, mes=None, season_index=None):
		
		# Função que formata e plota os GEOAXES (subplots --> mapas geográficos + colorbar)
		# Clevs, cmap, unidade são **Kwargs da função atributos, mas **Args da função plot
		# Caso clevs, cmap, unidade não sejam definidos na função atributos ---> assumem valores default já definidos
		
		if kind == "mensal":
			#var, lonu = dataset.sel(month=mes), dataset['lon'][:]
			var, lonu = add_cyclic_point(dataset.sel(month=mes), coord = dataset['lon'][:])
			if clevs != None:
				cf = ax.contourf(lonu, dataset['lat'][:], var, clevs, cmap=cmap, extend = "both")
			else:
				cf = ax.contourf(lonu, dataset['lat'][:], var, cmap=cmap, extend = "both")
			cbar = plt.colorbar(cf, orientation='horizontal', pad=0.15, ax=ax, shrink=1.1, aspect=40)
			cbar.ax.tick_params(labelsize=15)
			if unidade != None:
				cbar.set_label(f"{unidade}", fontsize=17, labelpad=20, style="oblique")

			gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color = 'black', alpha = 0.3, linestyle = '--')
			gl.xlabels_top = False
			gl.ylabels_left = True
			gl.ylabels_right = False
			gl.ylines = True
			gl.xlines = True
			gl.xformatter = LONGITUDE_FORMATTER
			gl.yformatter = LATITUDE_FORMATTER
			gl.xlocator = mticker.FixedLocator(np.arange(-180, 200, 10))
			gl.ylocator = mticker.FixedLocator(np.arange(-180, 200, 5))
			gl.xlabel_style = {'size': 10}

			ax.coastlines("50m")
			ax.get_extent(crs=ccrs.PlateCarree())																
			ax.set_title(calendar.month_abbr[mes], fontdict={'fontsize':20}, loc = 'right', style="oblique")
			
		elif kind == "sazonal":
			
			var, lonu = add_cyclic_point(dataset.isel(season=season_index), coord = dataset['lon'][:])
			if clevs != None:
				cf = ax.contourf(lonu, dataset['lat'][:], var, clevs, cmap=cmap, extend = "both")
			else:
				cf = ax.contourf(lonu, dataset['lat'][:], var, cmap=cmap, extend = "both")
			cbar = plt.colorbar(cf, orientation='horizontal', pad=0.15, ax=ax, shrink=1.1, aspect=40)
			cbar.ax.tick_params(labelsize=15)
			if unidade != None:
				cbar.set_label(f"{unidade}", fontsize=17, labelpad=20, style="oblique")

			gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color = 'black', alpha = 0.3, linestyle = '--')
			gl.xlabels_top = False
			gl.ylabels_left = True
			gl.ylabels_right = False
			gl.ylines = True
			gl.xlines = True
			gl.xformatter = LONGITUDE_FORMATTER
			gl.yformatter = LATITUDE_FORMATTER
			gl.xlocator = mticker.FixedLocator(np.arange(-180, 200, 10))
			gl.ylocator = mticker.FixedLocator(np.arange(-180, 200, 5))
			gl.xlabel_style = {'size': 10}
			gl.ylabel_style = {'size': 10}

			ax.coastlines("50m")
			ax.get_extent(crs=ccrs.PlateCarree())																
			ax.set_title(self.dataset.season[season_index].values, loc='left', fontsize=15, style="oblique")
			ax.set_title("LAMMOC-UFF", loc='right', fontsize=9, style="oblique")
	
	
	def multiplots(self, kind, periodo, periodobase=None, level=None, nome=None, unidade=None, clevs=None, cmap=None, export=False):
		
		'''
		Função que gera uma figura de análise anual com subplots para cada mês do ano.
		
		Call signature:
		
			self.multiplots(kind, periodo, periodobase, level, nome, unidade, clevs, cmap, export)
		
		Parameters
		----------
		kind : {"climatologia", "levantamento", "anomalia"}
		
			Esse parâmetro define o subtítulo da figura, especificando o intervalo temporal e o tipo da análise empregada.
		
			* climatologia --> caso a análise seja para um período de climatologia (Ex: climatologia de 1979 a 2010)  
			* levantamento --> caso a análise seja para um período de levantamento (Ex: levantamento de 2018 a 2020)
			* anomalia --> caso a análise seja para um período de anomalia (Ex: Anomalia de 2018 a 2020 em relação a 1979 a 2010) 
			
		periodo : list, str
		
			Intervalo temporal do período de análise.
			
			Ex: periodo = ["2005-01-01", "2010-12-01"]
			
		**Kwargs
		--------
		periodobase: list, str
		
			Intervalo temporal do *segundo* período de análise. Quando a análise é comparativa, são exigidos dois intervalos temporais: 'periodo' e 'periodobase'. Portanto, nessa função, 'periodobase' é somente necessário quando **kind == "anomalia" **. 
			
			Ex: periodo = ["2005-01-01", "2010-12-01"]  e periodobase = ["1979-01-01", "2010-12-01"]. Anomalia de 'periodo' em relação a 'periodobase'.
			
		level: int
			
			Caso os dados contenham dimensão 'level', é necessário especificar para qual valor de level a análise está sendo gerada. O valor numérico inteiro é convertido em string e especificado no título da figura.
			
			Ex: Zonal wind on 850 hPa --> Título da figura, onde '850': str(level)
			
		nome: str
		
			Título da figura que, por padrão, é o nome da variável analisada. Se **nome == None** (Padrão) e caso não seja uma variável conhecida pela biblioteca, a função buscará o nome da variável diretamente dos dados e retornará o valor encontrado como título. 
			
			Lista das variáveis e títulos pré-definidos pela biblioteca:
			
			* sst --> Sea Surface Temperature
			* olr --> Outgoing longwave radiation
			* uwnd --> Vento zonal
			* magnitude --> Wind magnitude
			* div --> Divergence
			* vort --> Vorticity
			* hgt --> Geopotential height
			
			Mesmo com tais títulos pré-definidos, cabe ao usuário definir o título da figura (basta explicitar nome = 'nome_do_título_desejado').
		
		unidade: str
		
			Unidade da variável analisada para a legenda do colorbar. Se **unidade = None** (Padrão) e caso não seja uma unidade conhecida pela biblioteca, a função buscará a unidade diretamente dos dados e retornará o valor encontrado como título (se não houver a unidade nos dados, o colorbar fica sem legenda).
			
			Lista das variáveis e unidades pré-definidas pela biblioteca:
			
			* sst --> °C
			* olr --> W / m²
			* uwnd --> m/s
			* magnitude --> m/s
			* div --> 10^{-6} s^{-1}
			* vort --> 10^{-6} s^{-1}
			* hgt --> m x 10¹ (se bd == "NCEP") / m x 10² (se bd == "ERA-5")
			
			Mesmo com tais unidades pré-definidas, cabe ao usuário definir a unidade do colorbar (basta explicitar unidade = 'nome_da_unidade').
			
		clevs: array
		
			Escala do colorbar ("color levels"). Define os intervalos de diferenciação por cores da série de dados. Quando menor o passo adotado, maior a variação da paleta e maior a precisão na análise gráfica. Se **clevs == None** (Padrão), a escala é gerada automaticamente a partir da série de dados.
			
			Recomendamos definir o 'clevs' usando a biblioteca numpy:
			
			* np.arange(valor_inicial, valor_final + passo, passo) --> cria um array de range [valor inicial , valor final] dividido em intervalos de largura **passo**
			
			Ex: np.arange(1, 4, 0.5) --> [1, 1.5, 2, 2.5, 3, 3.5]
			
			* np.linspace(valor_inicial, valor_final, número de intervalos desejado) --> cria um array de range [valor inicial , valor final] dividido em **número de intervalos desejados** intervalos.
			
			Ex: np.linspace(1, 3.5, 6) --> [1, 1.5, 2, 2.5, 3, 3.5]
			
			OBS.: Note que np.arange(1, 4, 0.5) e np.linspace(1, 3.5, 6) geram o MESMO ARRAY, deixando clara a diferença entre estes e cabendo ao usuário escolher o caminho mais conveniente para sua análise. 
			
		cmap : str
		
			Colormap adotado no subplot. Conferir a lista de colormaps da biblioteca 'Matplotlib'. Se **cmap = None** (Padrão), o cmap adotado é o default da biblioteca 'Matplotlib' ou default da biblioteca 'newBlock', isso dependerá da variável analisada. 
			
			Lista das variáveis e colormap pré-definidos pela biblioteca newBlock:
			
			* sst --> 'coolwarm'
			* olr --> 'jet'
			* uwnd --> 'jet_r'
			* magnitude --> 'jet_r'
			* div --> 'RdBu_r'
			* vort --> 'RdBu_r'
			* hgt --> 'blues'
			
			Mesmo com tais colormaps pré-definidos, cabe ao usuário definir o cmap (basta explicitar cmap = 'nome_do_cmap').
			
		export : bool
		
			Salvar a figura (formato '.png') em pastas e diretórios com caminhos pré-definidos pela biblioteca (conferir docstring da função 'exportar'). Se 'True', a figura é salva na pasta final definida. Se 'False' (Padrão) a imagem não é salva.
		
		Nessa função, a imagem gerada será salva na seguinte sequência de pastas:  
			
			'kind' -> 'nome_banco_de_dados' -> 'nome_variável' -> 'Multi_Plots' -> 'nome_imagem.png' 
			
			OBS.: Nome da imagem varia de acordo com os valores fornecidos para o *parâmetro* 'periodo' e para os *kwargs* 'periodobase' e 'level'.
			(Conferir doscstring da função 'exportar')
			
		'''
		
		# Função que define a figura, título e número de subplots
		# Periodo deve ser o mesmo intervalo do 'periodo' definido no dataset
		# self.dataset deve estar recortado anteriormente (função base, climatologia ou manipulado manualmente pelo próprio usuário)
		# Se self.dataset NÃO estiver recortado anteriormente, as funções de PLOTAGEM trabalharão com TODO o intervalo temporal (todos os 'times') do self.dataset (!!!)
		# Periodo explicitado nessa função é apenas para TÍTULO da figura e da imagem/pasta a ser exportada (função exportar)
		# Periodo explicitado nessa função NÃO RECORTA O DATASET (!!!) 
		# Para dados com LEVEL, explicitar o valor do level MESMO SE O DATASET JÁ ESTIVER RECORTADO (!!!)
		
		fig, ax = plt.subplots(nrows=3, ncols=4, subplot_kw=dict(projection=ccrs.PlateCarree()), figsize=(32, 22))
		plt.subplots_adjust(hspace=0.15)  # ajuste dos subplots
		
		plots = "Multi_Plots"
		nome, clevs, cmap, unidade, dataset = self.atributos(periodobase, level, nome, unidade, clevs, cmap)

		if kind == "climatologia":
			
			if level != None:
				plt.suptitle(f"{nome} on {level} hPa\n\nClimatology from {periodo[0][:4]} to {periodo[1][:4]}", fontsize=27, weight="medium",style="oblique",va="top", ha="center")				
			else:
				plt.suptitle(f"{nome}\n\nClimatology from {periodo[0][:4]} to {periodo[1][:4]}", fontsize=27, weight="medium",style="oblique",va="top", ha="center")
	
		elif kind == "levantamento":
			
			if level != None:
				plt.suptitle(f"{nome} on {level} hPa\n\nMonthly averages from {periodo[0][:4]}/{periodo[0][5:7]} to {periodo[1][:4]}/{periodo[1][5:7]}", fontsize=27, weight="medium",style="oblique",va="top", ha="center")		  
			else:
				plt.suptitle(f"{nome}\n\nMonthly averages from {periodo[0][:4]}/{periodo[0][5:7]} to {periodo[1][:4]}/{periodo[1][5:7]}", fontsize=27, weight="medium",style="oblique",va="top", ha="center")
			
		elif kind == "anomalia":
			
			if level != None:
				plt.suptitle(f"{nome} on {level} hPa\n\nAnomaly averaged from " + periodo[0][:4] + "/" + periodo[0][5:7] + " to " + periodo[1][:4] + "/" + periodo[1][5:7] + " relative to the long term averages from " + periodobase[0][:4] + " to " + periodobase[1][:4], fontsize=27, weight="medium",style="oblique",va="top", ha="center")	  
			else:
				plt.suptitle(f"{nome}\n\nAnomaly averaged from " + periodo[0][:4] + "/" + periodo[0][5:7] + " to " + periodo[1][:4] + "/" + periodo[1][5:7] + " relative to the long term averages from " + periodobase[0][:4] + " to " + periodobase[1][:4], fontsize=27, weight="medium",style="oblique",va="top", ha="center")
	   
		mes = np.arange(1, 13)
		
		k = 0
		for i in range(0, 3):
		# definindo a localização 'AX' de cada subplot mensal (ax[0,0] para mes[0], ax[0,1] para mes[1], ... , AX[i,j] para MES[k])
			for j in range(0, 4):
				self.plot(ax[i,j], "mensal", clevs, cmap, unidade, dataset, mes[k])
				k += 1
			
		#LABEL LAMMOC NO PRIMEIRO E ÚLTIMO SUBPLOT
		ax[0,0].set_title("LAMMOC-UFF", fontdict={'fontsize': 12}, style="italic", loc = 'left')
		ax[2,3].set_title("LAMMOC-UFF", fontdict={'fontsize': 12}, style="italic", loc = 'left')
		
		if export == True:
			self.exportar(kind, plots, periodo, periodobase, level, mes=None)
			
			
	def seasons(self, kind, periodo, periodobase=None, level=None, nome=None, unidade=None, clevs=None, cmap=None, export=False):
		
		'''
		Função que gera uma figura de análise sazonal com subplots para cada estação do ano.
		
		Call signature:
		
			self.seasons(kind, periodo, periodobase, level, nome, unidade, clevs, cmap, export)
		
		Parameters
		----------
		kind : {"climatologia", "levantamento", "anomalia"}
		
			Esse parâmetro define o subtítulo da figura, especificando o intervalo temporal e o tipo da análise empregada.
		
			* climatologia --> caso a análise seja para um período de climatologia (Ex: climatologia de 1979 a 2010)  
			* levantamento --> caso a análise seja para um período de levantamento (Ex: levantamento de 2018 a 2020)
			* anomalia --> caso a análise seja para um período de anomalia (Ex: Anomalia de 2018 a 2020 em relação a 1979 a 2010) 
			
		periodo : list, str
		
			Intervalo temporal do período de análise.
			
			Ex: periodo = ["2005-01-01", "2010-12-01"]
			
		**Kwargs
		--------
		periodobase: list, str
		
			Intrevalo temporal do *segundo* período de análise. Quando a análise é comparativa, são exigidos dois intervalos temporais: 'periodo' e 'periodobase'. Portanto, nessa função, 'periodobase' é somente necessário quando **kind == "anomalia" **. 
			
			Ex: periodo = ["2005-01-01", "2010-12-01"]  e periodobase = ["1979-01-01", "2010-12-01"]. Anomalia de 'periodo' em relação a 'periodobase'.
			
		level: int
			
			Caso os dados contenham dimensão 'level', é necessário especificar para qual valor de level a análise está sendo gerada. O valor numérico inteiro é convertido em string e especificado no título da figura.
			
			Ex: Zonal wind on 850 hPa --> Título da figura, onde '850': str(level)
			
		nome: str
		
			Título da figura que, por padrão, é o nome da variável analisada. Se **nome == None** (Padrão) e caso não seja uma variável conhecida pela biblioteca, a função buscará o nome da variável diretamente dos dados e retornará o valor encontrado como título. 
			
			Lista das variáveis e títulos pré-definidos pela biblioteca:
			
			* sst --> Sea Surface Temperature
			* olr --> Outgoing longwave radiation
			* uwnd --> Vento zonal
			* magnitude --> Wind magnitude
			* div --> Divergence
			* vort --> Vorticity
			* hgt --> Geopotential height
			
			Mesmo com tais títulos pré-definidos, cabe ao usuário definir o título da figura (basta explicitar nome = 'nome_do_título_desejado').
		
		unidade: str
		
			Unidade da variável analisada para a legenda do colorbar. Se **unidade == None** (Padrão) e caso não seja uma unidade conhecida pela biblioteca, a função buscará a unidade diretamente dos dados e retornará o valor encontrado como título (se não houver a unidade nos dados, o colorbar fica sem legenda).
			
			Lista das variáveis e unidades pré-definidas pela biblioteca:
			
			* sst --> °C
			* olr --> W / m²
			* uwnd --> m/s
			* magnitude --> m/s
			* div --> 10^{-6} s^{-1}
			* vort --> 10^{-6} s^{-1}
			* hgt --> m x 10¹ (se bd == "NCEP") / m x 10² (se bd == "ERA-5")
			
			Mesmo com tais unidades pré-definidas, cabe ao usuário definir a unidade do colorbar (basta explicitar unidade = 'nome_da_unidade').
			
		clevs: array
		
			Escala do colorbar ("color levels"). Define os intervalos de diferenciação por cores da série de dados. Quando menor o passo adotado, maior a variação da paleta e maior a precisão na análise gráfica. Se **clevs==None** (Padrão), a escala é gerada automaticamente a partir da série de dados.
			
			Recomendamos definir o 'clevs' usando a biblioteca numpy:
			
			* np.arange(valor_inicial, valor_final + passo, passo) --> cria um array de range [valor inicial , valor final] dividido em intervalos de largura **passo**
			
			Ex: np.arange(1, 4, 0.5) --> [1, 1.5, 2, 2.5, 3, 3.5]
			
			* np.linspace(valor_inicial, valor_final, número de intervalos desejado) --> cria um array de range [valor inicial , valor final] dividido em **número de intervalos desejados** intervalos.
			
			Ex: np.linspace(1, 3.5, 6) --> [1, 1.5, 2, 2.5, 3, 3.5]
			
			OBS.: Note que np.arange(1, 4, 0.5) e np.linspace(1, 3.5, 6) geram o MESMO ARRAY, deixando clara a diferença entre estes e cabendo ao usuário escolher o caminho mais conveniente para sua análise. 
			
		cmap : str
		
			Colormap adotado no subplot. Conferir a lista de colormaps da biblioteca 'Matplotlib'. Se **cmap==None** (Padrão), o cmap adotado é o default da biblioteca 'Matplotlib' ou default da biblioteca 'newBlock', isso dependerá da variável analisada. 
			
			Lista das variáveis e colormap pré-definidos pela biblioteca newBlock:
			
			* sst --> 'coolwarm'
			* olr --> 'jet'
			* uwnd --> 'jet_r'
			* magnitude --> 'jet_r'
			* div --> 'RdBu_r'
			* vort --> 'RdBu_r'
			* hgt --> 'blues'
			
			Mesmo com tais colormaps pré-definidos, cabe ao usuário definir o cmap (basta explicitar cmap = 'nome_do_cmap').
			
		export : bool
		
			Salvar a figura (formato '.png') em pastas e diretórios com caminhos pré-definidos pela biblioteca (conferir docstring da função 'exportar'). Se 'True', a figura é salva na pasta final definida. Se 'False' (Padrão) a imagem não é salva.
			
			Nessa função, a imagem gerada será salva na seguinte sequência de pastas:  
			
			'kind' -> 'nome_banco_de_dados' -> 'nome_variável' -> 'Season_Plots' -> 'nome_imagem.png' 
			
			OBS.: Nome da imagem varia de acordo com os valores fornecidos para o *parâmetro* 'periodo' e para os *kwargs* 'periodobase' e 'level'.
			(Conferir docstring da função 'exportar')
		
		'''
		
		plots = "Season_Plots"
		
		fig, ax = plt.subplots(ncols = 2, nrows = 2, subplot_kw=dict(projection=ccrs.PlateCarree()), figsize=(20, 12))
		nome, clevs, cmap, unidade, dataset = self.atributos(periodobase, level, nome, unidade, clevs, cmap)
		
		if kind == "climatologia":
			
			if level != None:
				plt.suptitle(f"{nome} on {level} hPa\n\nClimatology from {str(int(periodo[0][0:4]) + 1)} to {periodo[1][:4]}", fontsize=15, weight="medium",style="oblique",va="top", ha="center")				
			else:
				plt.suptitle(f"{nome}\n\nClimatology from {str(int(periodo[0][0:4]) + 1)} to {periodo[1][:4]}", fontsize=15, weight="medium",style="oblique",va="top", ha="center")
	
		elif kind == "levantamento":
			
			if level != None:
				plt.suptitle(f"{nome} on {level} hPa\n\nSeasonal averages from {str(int(periodo[0][0:4]) + 1)} to {periodo[1][:4]}", fontsize=15, weight="medium",style="oblique",va="top", ha="center")		  
			else:
				plt.suptitle(f"{nome}\n\nSeasonal averages from {str(int(periodo[0][0:4]) + 1)} to {periodo[1][:4]}", fontsize=15, weight="medium",style="oblique",va="top", ha="center")
			
		elif kind == "anomalia":
			
			if level != None:
				plt.suptitle(f"{nome} on {level} hPa\n\nAnomaly averaged from " + str(int(periodo[0][0:4]) + 1) + " to " + periodo[1][:4] + " relative to the seasonal averages from " + str(int(periodobase[0][0:4]) + 1) + " to " + periodobase[1][:4], fontsize=15, weight="medium",style="oblique",va="top", ha="center")	  
			else:
				plt.suptitle(f"{nome}\n\nAnomaly averaged from " + str(int(periodo[0][0:4]) + 1) + " to " + periodo[1][:4] + " relative to the seasonal averages from " + str(int(periodobase[0][0:4]) + 1) + " to " + periodobase[1][:4], fontsize=15, weight="medium",style="oblique",va="top", ha="center")
				
		k = 0
		for i in range (0, 2):
			for j in range(0, 2):
				self.plot(ax[i,j], 'sazonal', clevs, cmap, unidade, dataset, None, k)
				k += 1
		
		if export == True:
			self.exportar(kind, plots, periodo, periodobase, level)


	def singleplot(self, kind, mes, periodo, periodobase=None, level=None, nome=None, unidade=None, clevs=None, cmap=None, export=False):
		
		'''
		Função que gera uma figura de análise mensal com um único subplot correspondente ao mês indicado como parâmetro. 
		
		Call signature:
		
			self.singleplot(kind, mes, periodo, periodobase, level, nome, unidade, clevs, cmap, export)
			
			Obs: Se o usuário desejar gerar de vez uma figura para cada mês do ano (totalizando 12 ao final), recomenda-se executar o código abaixo:
		
			--> meses = np.arange(1, 13)
			--> mes in meses:
			-->    self.singleplot(kind, mes, periodo, periodobase, level, nome, unidade, clevs, cmap, export) 
		
		Parameters
		----------
		kind : {"climatologia", "levantamento", "anomalia"}
		
			Esse parâmetro define o subtítulo da figura, especificando o intervalo temporal e o tipo da análise empregada.
		
			* climatologia --> caso a análise seja para um período de climatologia (Ex: climatologia de 1979 a 2010)  
			* levantamento --> caso a análise seja para um período de levantamento (Ex: levantamento de 2018 a 2020)
			* anomalia --> caso a análise seja para um período de anomalia (Ex: Anomalia de 2018 a 2020 em relação a 1979 a 2010) 
			
		mes: int in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
			
			Mês em que a análise está sendo gerada. Parâmetro deve ser fornecido como valor numérico (dentro do intervalo supramencionado) e não como string (Ex: 'JAN').
			
		periodo : list, str
		
			Intervalo temporal da análise.
			
			Ex: periodo = ["2005-01-01", "2010-12-01"]
			
		**Kwargs
		--------
		periodobase: list, str
		
			Quando a análise é comparativa, são exigidos dois intervalos temporais: 'periodo' e 'periodobase'. Portanto, nessa função, 'periodobase' é somente necessário quando **kind == "anomalia" **. 
			
			Ex: periodo = ["2005-01-01", "2010-12-01"]  e periodobase = ["1979-01-01", "2010-12-01"]. Anomalia de 'periodo' em relação a 'periodobase'.
			
		level: int
			
			Caso os dados contenham dimensão 'level', é necessário especificar para qual valor de level a análise está sendo gerada. O valor numérico inteiro é convertido em string e especificado no título da figura.
			
			Ex: Zonal wind on 850 hPa --> Título da figura, onde '850': str(level)
			
		nome: str
		
			Título da figura que, por padrão, é o nome da variável analisada. Se **nome == None** (Padrão) e caso não seja uma variável conhecida pela biblioteca, a função buscará o nome da variável diretamente dos dados e retornará o valor encontrado como título. 
			
			Lista das variáveis e títulos pré-definidos pela biblioteca:
			
			* sst --> Sea Surface Temperature
			* olr --> Outgoing longwave radiation
			* uwnd --> Vento zonal
			* magnitude --> Wind magnitude
			* div --> Divergence
			* vort --> Vorticity
			* hgt --> Geopotential height
			
			Mesmo com tais títulos pré-definidos, cabe ao usuário definir o título da figura (basta explicitar nome = 'nome_do_título_desejado').
		
		unidade: str
		
			Unidade da variável analisada para a legenda do colorbar. Se **unidade == None** (Padrão) e caso não seja uma unidade conhecida pela biblioteca, a função buscará a unidade diretamente dos dados e retornará o valor encontrado como título (se não houver a unidade nos dados, o colorbar fica sem legenda).
			
			Lista das variáveis e unidades pré-definidas pela biblioteca:
			
			* sst --> °C
			* olr --> W / m²
			* uwnd --> m/s
			* magnitude --> m/s
			* div --> 10^{-6} s^{-1}
			* vort --> 10^{-6} s^{-1}
			* hgt --> m x 10¹ (se bd == "NCEP") / m x 10² (se bd == "ERA-5")
			
			Mesmo com tais unidades pré-definidas, cabe ao usuário definir a unidade do colorbar (basta explicitar unidade = 'nome_da_unidade').
			
		clevs: array
		
			Escala do colorbar ("color levels"). Define os intervalos de diferenciação por cores da série de dados. Quando menor o passo adotado, maior a variação da paleta e maior a precisão na análise gráfica. Se **clevs = None** (Padrão), a escala é gerada automaticamente a partir da série de dados.
			
			Recomendamos definir o 'clevs' usando a biblioteca numpy por *dois* diferentes caminhos:
			
			* np.arange(valor_inicial, valor_final + passo, passo) --> cria um array de range [valor inicial , valor final] dividido em intervalos de largura **passo**
			
			Ex: np.arange(1, 4, 0.5) --> [1, 1.5, 2, 2.5, 3, 3.5]
			
			* np.linspace(valor_inicial, valor_final, número de intervalos desejado) --> cria um array de range [valor inicial , valor final] dividido em **número de intervalos desejados** intervalos.
			
			Ex: np.linspace(1, 3.5, 6) --> [1, 1.5, 2, 2.5, 3, 3.5]
			
			OBS.: Note que np.arange(1, 4, 0.5) e np.linspace(1, 3.5, 6) geram o MESMO ARRAY, deixando clara a diferença entre estes e cabendo ao usuário escolher o caminho mais conveniente para sua análise. 
			
		cmap : str
		
			Colormap adotado no subplot. Conferir a lista de colormaps da biblioteca 'Matplotlib'. Se **cmap = None** (Padrão), o 'cmap' adotado é o default (Padrão) da biblioteca 'Matplotlib' ou default (Padrão) da biblioteca 'newBlock', isso dependerá da variável analisada. 
			
			Lista das variáveis e colormaps pré-definidos pela biblioteca newBlock:
			
			* sst --> 'coolwarm'
			* olr --> 'jet'
			* uwnd --> 'jet_r'
			* magnitude --> 'jet_r'
			* div --> 'RdBu_r'
			* vort --> 'RdBu_r'
			* hgt --> 'blues'
			
			Mesmo com tais colormaps pré-definidos, cabe ao usuário definir o cmap (basta explicitar cmap = 'nome_do_cmap').
			
		export : bool
		
			Salvar a figura (formato '.png') em pastas e diretórios com caminhos pré-definidos pela biblioteca (conferir docstring da função 'exportar'). Se 'True', a figura é salva na pasta final definida. Se 'False' (Padrão) a imagem não é salva.
			
			Nessa função, a imagem gerada será salva na seguinte sequência de pastas:  
			
			'kind' -> 'nome_banco_de_dados' -> 'nome_variável' -> 'Single_Plot' -> 'nome_imagem.png' 
			
			OBS.: Nome da imagem varia de acordo com os valores fornecidos para o *parâmetro* 'periodo' e 'mes' e para os *kwargs* 'periodobase' e 'level'.
			(Conferir docstring da função 'exportar')
			
		'''
		
		# Função que plota um único Plot (GeoAxes) correspondente ao MES (parametro da função --> *ARG) definido pelo usuário
		# Caso o usuário queira plotar os 12 singleplots (um para cada mês), recomendar o seguinte comando QUANDO FOR ''DAR RUN''(!!):
		
		# ---------- EXECUTAR LINHA DE CÓDIGO ABAIXO -----------> ONDE 'OBJ_PLOT' é o OBJETO 'LAMMOC_PLOT' e 'SINGLEPLOT' a FUNÇÃO (!!!)
		
		#							  meses = np.arange(1, 13)
		#							  for mes in meses:
		#								  OBJ_PLOT.singleplot(kind, mes, periodo, **KWARGS) 
		
		# -------------- (ESSE CÓDIGO PERMITE AO USUÁRIO GERAR 12 IMAGENS SINGLEPLOT, UMA PARA CADA MÊS) --------------------------
		
		plots = "Single_Plot"
		
		fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()), figsize=(16, 12))
		nome, clevs, cmap, unidade, dataset = self.atributos(periodobase, level, nome, unidade, clevs, cmap)
		
		if kind == "climatologia":
			
			if level != None:
				plt.suptitle(f"{nome} on {level} hPa\n\nClimatology from {periodo[0][:4]} to {periodo[1][:4]}", fontsize=15, weight="medium",style="oblique",va="top", ha="center")				
			else:
				plt.suptitle(f"{nome}\n\nClimatology from {periodo[0][:4]} to {periodo[1][:4]}", fontsize=15, weight="medium",style="oblique",va="top", ha="center")
	
		elif kind == "levantamento":
			
			if level != None:
				plt.suptitle(f"{nome} on {level} hPa\n\nMonthly average from {periodo[0][:4]}/{periodo[0][5:7]} to {periodo[1][:4]}/{periodo[1][5:7]}", fontsize=15, weight="medium",style="oblique",va="top", ha="center")		  
			else:
				plt.suptitle(f"{nome}\n\nMonthly average from {periodo[0][:4]}/{periodo[0][5:7]} to {periodo[1][:4]}/{periodo[1][5:7]}", fontsize=15, weight="medium",style="oblique",va="top", ha="center")
			
		elif kind == "anomalia":
			
			if level != None:
				plt.suptitle(f"{nome} on {level} hPa\n\nAnomaly averaged from " + periodo[0][:4] + "/" + periodo[0][5:7] + " to " + periodo[1][:4] + "/" + periodo[1][5:7] + " relative to the long term average from " + periodobase[0][:4] + " to " + periodobase[1][:4], fontsize=15, weight="medium",style="oblique",va="top", ha="center")	  
			else:
				plt.suptitle(f"{nome}\n\nAnomaly averaged from " + periodo[0][:4] + "/" + periodo[0][5:7] + " to " + periodo[1][:4] + "/" + periodo[1][5:7] + " relative to the long term average from " + periodobase[0][:4] + " to " + periodobase[1][:4], fontsize=15, weight="medium",style="oblique",va="top", ha="center")
				
		self.plot(ax, 'mensal', clevs, cmap, unidade, dataset, mes)
		ax.set_title("LAMMOC-UFF", fontdict={'fontsize': 8}, style="italic", loc = 'left')
		
		if export == True:
			self.exportar(kind, plots, periodo, periodobase, level, mes)
		
		
	def barplot(self, ax, coords, periodo, periodobase, periodobase_2=None, level=None):  
		
		# Gráfico comparativo de barras
		# Gráfico pode comparar até 3 períodos ao mesmo tempo
		# coords = [LONGITUDE, LATITUDE] -> respeitar esta ordem
		# O KWARG 'LEVEL' SÓ É NECESSÁRIO CASO O OBJETO.DATASET NÃO ESTEJA RECORTADO PARA O VALOR 'LEVEL'
		# SE O OBJETO.DATASET ESTIVER RECORTADO (recortado no formatar(), por ex), O VALOR DE 'LEVEL' É BUSCADO DIRETAMENTE NO .DATASET
		
		nome, clevs, cmap, unidade, dataset = self.atributos()
		mes = []
		for i in range(1, 13):
			mes.append(calendar.month_abbr[i])
		
		data = self.dataset.sel(lat=coords[1], lon=coords[0]) 
		
		# INSTANCIANDO OBJETO LAMMOC_MEDIAS PARA PODER CALCULAR A CLIMATOLOGIA PELA FUNÇÃO 'CLIMATOLOGIA' DA CLASSE 'LAMMOC_MEDIAS' (!!)
		obj_medias = new_medias(data, self.bd)
		
		x_bar = np.arange(1, 13, 1)
		
		if "level" in self.dataset:
			if level == None:
				level = int(self.level)
				y_bar_1 = (obj_medias.climatologia(periodo)).dataset.to_array().isel(variable=0)
				y_bar_2 = (obj_medias.climatologia(periodobase)).dataset.to_array().isel(variable=0)
			else:  
				y_bar_1 = (obj_medias.climatologia(periodo, level)).dataset.to_array().isel(variable=0)
				y_bar_2 = (obj_medias.climatologia(periodobase, level)).dataset.to_array().isel(variable=0)
		else:
			y_bar_1 = (obj_medias.climatologia(periodo)).dataset.to_array().isel(variable=0)
			y_bar_2 = (obj_medias.climatologia(periodobase)).dataset.to_array().isel(variable=0)
		
		if periodobase_2 != None: 
		# Gráfico com 3 barras mensais (comparando 3 períodos)
			if "level" in self.dataset:
				if level == None:
					level = int(self.level)
					y_bar_3 = (obj_medias.climatologia(periodo)).dataset.to_array().isel(variable=0)
				else:
					y_bar_3 = (obj_medias.climatologia(periodobase_2, level)).dataset.to_array().isel(variable=0)
			else:
				y_bar_3 = (obj_medias.climatologia(periodo)).dataset.to_array().isel(variable=0)
				
			label = [f"{periodo[0][:4]} a {periodo[1][:4]}", f"{periodobase[0][:4]} a {periodobase[1][:4]}", f"{periodobase_2[0][:4]} a {periodobase_2[1][:4]}"]
			width = 0.26
			bar1 = ax.bar(x_bar, y_bar_1, color="lightsalmon", width=width, label=label[0], align="center", edgecolor="white")
			bar2 = ax.bar(x_bar - (width + 0.05), y_bar_2, color="cornflowerblue", width=width, label=label[1], align="center", edgecolor="white")
			bar3 = ax.bar(x_bar + (width + 0.05), y_bar_3, color="mediumspringgreen", width=width, label=label[2], align="center", edgecolor="white")
				
		else:
		# Gráfico com 2 barras mensais (comparando 2 períodos)
			label = [f"{periodo[0][:4]} a {periodo[1][:4]}", f"{periodobase[0][:4]} a {periodobase[1][:4]}"]
			width = 0.3
			bar1 = ax.bar(x_bar, y_bar_1, color="lightsalmon", width=width, label=label[0], align="edge", edgecolor="white", linewidth=2.5)
			bar2 = ax.bar(x_bar, y_bar_2, color="cornflowerblue", width=width*-1, label=label[1], align="edge", edgecolor="white", linewidth=2.5)
	 
		ax.set_xticks(x_bar)
		ax.set_xticklabels(mes)
		ax.set_ylabel(f"{unidade}", labelpad=15, fontsize=15, style="oblique")
		ax.set_xlabel("Mês", labelpad=15, fontsize=15, x=0.5, style="oblique")
		ax.set_title("LAMMOC-UFF", fontsize=8, loc="left", style="italic")
		ax.legend(bbox_to_anchor=(0.858, 1.045), loc="upper center", ncol=3, prop=font_manager.FontProperties(style='oblique', size=9), fancybox=True, shadow=True)				  
	
		def autolabel(bars):   
			
			# Função que plota os valores em y exatos acima de cada barra (com 2 casa decimais)
			for bar in bars:
				height = bar.get_height()
				x = Decimal(height)
				y = round(x, 2)
				ax.annotate(f'{y}',
							xy=(bar.get_x() + bar.get_width() / 1.8, y),
							xytext=(0, 12),  
							textcoords="offset points",
							ha='center', va='top', fontsize=8)

		autolabel(bar1)
		autolabel(bar2)
		if periodobase_2 != None:
			autolabel(bar3)
			
			
	def bars(self, coords, periodo, periodobase, periodobase_2=None, level=None, export=False):
		
		'''
		Função que gera um gráfico comparativo de barras, recebendo no mínimo *dois* e no máximo *três* períodos para comparação. As séries de dados são comparadas mês a mês.
		
		Call signature:
		
			self.bars(coords, periodo, periodobase, periodobase_2, level, export)
			
		Parameters
		----------
		coords : list, len(list == 2)
		
			Recebe a lista de coordenadas -- na ordem [longitude, latitude] -- do local onde está sendo feita a análise. O parâmetro recebe coordenadas exatas (len(list) == 2) e não àquelas equivalentes a uma região (len(list) == 4).
			
			Ex.: coords = [-55, -5] (longitude: -55 W, latitude: -5 S) --> coordenadas exatas --> CORRETO
			Ex.: coords = [-55, -45, -5, -10] (intervalo de longitude e de latitude) --> coordenadas de uma região --> ERRADO
			
		periodo : list, str
		
			Intervalo temporal do *primeiro* período de análise.
			
			Ex: periodo = ["2005-01-01", "2010-12-01"]
			
		periodobase: list, str
		
			Intervalo temporal do *segundo* período de análise. 
			
			Ex: periodo = ["2005-01-01", "2010-12-01"] e periodobase = ["1979-01-01", "2010-12-01"]. 
			
		**Kwargs
		--------
		level: int
			
			Caso os dados contenham dimensão 'level', é necessário especificar para qual valor de level a análise está sendo gerada. O valor numérico inteiro é convertido em string e especificado no título da figura.
			
			Ex: 'Wind magnitude comparison on 850 hPa' --> título da figura , onde '850': str(level)
			
		export : bool
		
			Salvar a figura (formato '.png') em pastas e diretórios com caminhos pré-definidos pela biblioteca (conferir docstring da função 'exportar'). Se 'True', a figura é salva na pasta final definida. Se 'False' (Padrão) a imagem não é salva.
			
			Nessa função, a imagem gerada será salva na seguinte sequência de pastas:  
			
			'CLIMATOLOGIAS' -> 'nome_banco_de_dados' -> 'nome_variável' -> 'Bars_Plot' -> 'nome_imagem.png' 
			
			OBS.: Nome da imagem varia de acordo com os valores fornecidos para o *parâmetro* 'periodo' e 'periodobase' e para os *kwargs* 'periodobase_2' e 'level'
			(Conferir docstring da função 'exportar')
			
		'''
		
		# Função que define a figura, título e subplot do gráfico de barras
		# Se o usuário quiser comparar 3 período ao mesmo tempo ----> DEFINIR O PARÂMETRO 'periodobase2' (Default: None)
		
		plots = "Bars_Plot"
		nome, clevs, cmap, unidade, dataset = self.atributos()
		fig, ax = plt.subplots(figsize=(16, 7))
			
		if "level" in self.dataset:
			if level == None:
				if "level" in self.dataset:
					level_title = int(self.level)
				else:
					level_title = level

			plt.suptitle(f"{nome} Comparison on {level_title} hPa", ha="center", va="top", weight="bold", fontsize=20, y=1.05)
			
		else:
			plt.suptitle(f"{nome} Comparison", ha="center", va="top", weight="bold", fontsize=20, y=1.05)

		if periodobase_2 != None:
			plt.title(f"Monthly Averages for {periodo[0][:4]} to {periodo[1][:4]} relative to the Monthly Averages for {periodobase[0][:4]} to {periodobase[1][:4]} and for {periodobase_2[0][:4]} to {periodobase_2[1][:4]}", fontsize=14, y=1.06, x=0.485)
				
		else:
			plt.title(f"Monthly Averages for {periodo[0][:4]} to {periodo[1][:4]} relative to the Monthly Averages for {periodobase[0][:4]} to {periodobase[1][:4]}", fontsize=14, y=1.06, x=0.475)

		self.barplot(ax, coords, periodo, periodobase, periodobase_2, level)
			
		# melhor ajuste para a figura
		fig.tight_layout() 

		if export != False:
			if periodobase_2 != None:
				self.exportar("climatologia", plots, periodo, None, level, None)
			else:
				self.exportar("climatologia", plots, periodo, periodobase, level, None)	
		
		
	def diretorios(self, array, plots):   
		
		# criando pastas no diretório, seguindo sequência: Pasta BD -> Pasta VAR -> Pasta Plots -> imagens salvas
		# EX::  ANOMALIAS -> ERA-5 -> UWND -> Mes_Plot -> (**Nome da imagem salva**) 
		# LOGO, O CAMINHO DO EXEMPLO ACIMA É --> ANOMALIAS/ ERA-5/ UWND/ MES_PLOT/ (**Nome da imagem salva**) 
		# ONDE --> ANOMALIAS: anompath, ERA-5: bd, UWND: var, MES_PLOT: plots, (**Nome da imagem salva**): nome da imagem
		# os nomes que as imagens serão salvas são definidos na função 'EXPORTAR', seguindo o padrão lá explicado/comentado
		
		self.climapath = os.path.join(os.getcwd(), "CLIMATOLOGIA")
		self.anompath = os.path.join(os.getcwd(), "ANOMALIA")
		dataset = self.dataset
		dataset = dataset.to_array().isel(variable=0)
		var = str(dataset["variable"].values)
		bd = self.bd
		
		try:
			os.mkdir(self.anompath)
			os.mkdir(self.climapath)
		except FileExistsError:
			pass
		
		if array == "climatologia":
			try:
				os.mkdir(os.path.join(self.climapath, bd))
			except FileExistsError:
				pass
			path = os.path.join(self.climapath, bd, var)
			try:
				os.mkdir(path)
			except FileExistsError:
				pass
			path = os.path.join(self.climapath, bd, var, plots)
			try:
				os.mkdir(path)
			except FileExistsError:
				pass
						 
		elif array == "anomalia":
			try:
				os.mkdir(os.path.join(self.anompath, bd))
			except FileExistsError:
				pass
			path = os.path.join(self.anompath, bd, var)
			try:
				os.mkdir(path)
			except FileExistsError:
				pass
			path = os.path.join(self.anompath, bd, var, plots)
			try:
				os.mkdir(path)
			except FileExistsError:
				pass
						 
		return path
	
	
	def exportar(self, array, plots, periodo, periodobase=None, level=None, mes=None):
		
		'''
		Função que salva e exporta a imagem gerada para diretórios e pastas com caminhos pré-definidos:
		
		'array' --> 'nome_banco_de_dados' --> 'nome_da_variável' --> 'plots' --> 'nome_da_imagem.png'  , onde 'ARRAY' e 'PLOTS' são parâmetros da função.
		
		OBS.: Nome da imagem ('nome_da_imagem.png') varia de acordo com os valores fornecidos para o *parâmetro* 'periodo' e para os *kwargs* 'periodobase', 'level' e 'mes'.
		
		--> Conferir exemplos nas definições dos *parameters e *kwargs <--
		
		OBS_2.: As pastas serão salvas (em sequência) no mesmo caminho em que o arquivo do código está sendo executado e salvo. 
		
		Ex: Se o 'nome_código.py' ou 'nome_código.ipynb' (se for um jupyter notebook) é executado e salvo no caminho *C:/Users/nome_user/Vento*, a pasta inicial (e o restante da sequência de pastas) também será(ão) salva(s) nesse mesmo caminho. 
		
		Call signature:
		
			self.exportar(array, plots, periodo, periodobase, level, mes)
			
			OBS.: Para executar a função exportar, recomenda-se definir **export = True** como kwarg das funções de plot, pois assim a função é chamada dentro de seus escopos e seu processo de exportação é automatizado, poupando o usuário de inserir mais um linha de código para poder chamar a função 'exportar' e, então, salvar a imagem.
			
		Parameters
		----------
		array : {"climatologia", "anomalia"}
		
			Nome da primeira pasta da sequência criada.

			* climatologia --> cria uma pasta inicial "CLIMATOLOGIA"
			* anomalia --> cria uma pasta inicial "ANOMALIA"
		
		plots: {"Bars_Plot", "Multi_Plots", "Season_Plots", "Single_Plot"}
		
			Nome do tipo de análise gráfica gerada. Quando a função 'exportar' é chamada dentro das funções de plot (**export = True**), o parâmetro 'plots' é automaticamente definido.
			
		periodo : list, str
		
			Intervalo temporal do período análise. Com abreviações, 'periodo' é explicitado no *nome_da_imagem.png*.
			
			Ex: periodo = ["2005-01-01", "2010-12-01"] --> no *nome_da_imagem.png* é abreviado como *05_10* (2005 a 2010)
			
		**Kwargs
		--------
		periodobase: list, str
		
			Intervalo temporal do *segundo* período de análise. Com abreviações, 'periodobase' é explicitado no *nome_da_imagem.png* logo depois da abrevição de 'periodo'.
			
			Ex: periodo = ["2005-01-01", "2010-12-01"] e periodobase = ["1979-01-01", "2010-12-01"] --> no *nome_da_imagem.png* teremos duas abreviações *05_10* e 79_10* (VEJA ABAIXO)
			
			-->  olr_05_10_79_10 , onde 'olr': nome_variavel, '05_10': abreviação_periodo, '79_10': abreviação_periodobase 
			
		level: int
			
			Caso os dados contenham dimensão 'level', é necessário especificar para qual valor de level a análise está sendo gerada. O valor numérico inteiro é convertido em string e especificado no *nome_da_imagem.png*.
			
			Ex: uwnd_79_10_850 , onde 'uwnd': nome_variavel, '79_10': abreviação_periodo e '850': str(level)
		
		mes: int in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
			
			Mês em que a análise está sendo gerada, caso esta seja do tipo mensal. Parâmetro deve ser fornecido como valor numérico (dentro do intervalo supramencionado) e não como string (Ex: 'JAN'). A partir da biblioteca 'Calendar', é retornado o nome do mês correspondente ao número indicado no parâmetro (Ex: 3 --> 'MAR').
			
			A string retornada equivalente ao mês fornecido é especificada no *nome_da_imagem.png* (VEJA ABAIXO)
			
			Ex: olr_79_10_Mar  , onde 'olr': nome_variavel, '79_10': abreviação_periodo e 'Mar': str(mes)
			
			
		'''
		
		# Função que define as pastas aonde serão exportadas as saídas (*imagens geradas* .png) 
		# As pastas são aquelas criadas a partir da função 'DIRETÓRIOS'
		# Os nomes das imagens salvas seguem um padrão : VAR_PERIODO_PERIODOBASE(se houver)_MES(se houver)_LEVEL(se houver).PNG
		# EX:: uwnd_18_20_79_10_jan_850.png, onde ---> UWND: var, 18_20: periodo, 79_10: periodobase, JAN: mes, 850: level, .PNG: formato 
		
		path = self.diretorios(array, plots)
		dataset = self.dataset
		dataset = dataset.to_array().isel(variable=0)
		var = str(dataset["variable"].values)
					 
		if periodobase != None:
			if mes != None:
				if level != None:
					plt.savefig(os.path.join(path, var + "_" + str(periodo[0][2:4]) + "_" + str(periodo[1][2:4]) + "_" + str(periodobase[0][2:4]) + "_" + str(periodobase[1][2:4]) + "_" + calendar.month_abbr[mes] + "_" + str(level) + ".png"), bbox_inches="tight")
				else:
					plt.savefig(os.path.join(path, var + "_" + str(periodo[0][2:4]) + "_" + str(periodo[1][2:4]) + "_" + str(periodobase[0][2:4]) + "_" + str(periodobase[1][2:4]) + "_" + calendar.month_abbr[mes] + ".png"), bbox_inches="tight")
			else:
				if level != None:
					plt.savefig(os.path.join(path, var + "_" + str(periodo[0][2:4]) + "_" + str(periodo[1][2:4]) + "_" + str(periodobase[0][2:4]) + "_" + str(periodobase[1][2:4]) + "_" + str(level) + ".png"), bbox_inches="tight")
				else:
					plt.savefig(os.path.join(path, var + "_" + str(periodo[0][2:4]) + "_" + str(periodo[1][2:4]) + "_" + str(periodobase[0][2:4]) + "_" + str(periodobase[1][2:4]) + ".png"), bbox_inches="tight")
		
		else:
			if mes != None:
				if level != None:
					plt.savefig(os.path.join(path, var + "_" + str(periodo[0][2:4]) + "_" + str(periodo[1][2:4]) + "_" + calendar.month_abbr[mes] + "_" + str(level) + ".png"), bbox_inches="tight")
				else:
					plt.savefig(os.path.join(path, var + "_" + str(periodo[0][2:4]) + "_" + str(periodo[1][2:4]) + "_" + calendar.month_abbr[mes] + ".png"), bbox_inches="tight")
			else:
				if level != None:
					plt.savefig(os.path.join(path, var + "_" + str(periodo[0][2:4]) + "_" + str(periodo[1][2:4]) + "_" + str(level) + ".png"), bbox_inches="tight")
				else:
					plt.savefig(os.path.join(path, var + "_" + str(periodo[0][2:4]) + "_" + str(periodo[1][2:4]) + ".png"), bbox_inches="tight")
