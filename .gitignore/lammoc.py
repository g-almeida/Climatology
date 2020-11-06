import xarray as xr


def criar_dataset(path):
	dados = xr.open_dataset(path)
	return lammoc_dataset(dados)
	
class lammoc_dataset():
	def __init__(self, dados):
		self.xr = dados
		self.variables = self.xr.variables
	
	def testando(self, dados):
		print(dados)
			
	def climatologia_ERA(self, var, level, timeAnalise):
	#Função para calcular a climatologia dados o intervalo de tempo, level e a variável.
		data = self.xr
		dataset_var = data[var].isel(expver=0)   # Eliminando as variações do expver  
		if level != 0:
			dataset_var = dataset_var.sel(level=level)
		time_slice = slice(timeAnalise[0], timeAnalise[1])
		climatologia = dataset_var.sel(time=time_slice).groupby('time.month').mean()   #climatologia mensal no time_range determinado
    
		return climatologia
		
	def anomalia_ERA(self, var, level, timeAnalise, timeBase):
	# Função para calcular anomalia dados o intervalo de tempo analisado, intervalo de tempo da climatologia comparada, level e a variável.
		
		climaSelf = self.climatologia_ERA(var, level, timeAnalise)
		
	# Determinado o clima no primeiro time_range, vamos à base climatológica escolhida
		
		climatologia = self.climatologia_ERA(var, level, timeBase)
		 
	# Calculada a climatologia base e o clima no intervalo analisado, vamos à anomalia
		
		anomalia = climaSelf - climatologia
		
		return anomalia
		
	def media_regional(self, var, level, timeAnalise, latitudes, longitudes):
	# Função para calcular média regional dados intervalos de tempo, latitude, longitude, level e a variável.
	
		data = self.xr
		dataset_var = data[var].isel(expver=0)   # Eliminando as variações do expver
		if level != 0:
			dataset_var = dataset_var.sel(level=level)
		time_slice = slice(timeAnalise[0], timeAnalise[1])
		lat_slice = slice(latitudes[0], latitudes[1])
		lon_slice = slice(longitudes[0], longitudes[1])
		data_cortes = dataset_var.sel(latitude=lat_slice).sel(longitude=lon_slice)  #determinado os cortes de latitude e longitude
		climat_regional = data_cortes.sel(time=time_slice).groupby('time.month').mean()   #climatologia mensal no time_range determinado
    
		return climat_regional
		