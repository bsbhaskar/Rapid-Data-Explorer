import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Rapid_Data_Explorer:
	# this class automates the data exploration steps by building plots specific to each category
	# for numerical data - it creates scatter plots
	# creates multiple fields by year,  month etc for date fields
	# creates box plots for all categorical fields

	def __init__(self, df, sample=2000):
		# by default, the dataframe is sampled to improve the speed of building plots

		self.df = df
		if (self.df.values.shape[0] > sample):
			self.df_sampler = self.df.sample(n=sample)
		else:
			self.df_sampler = df


	def plot_scatter_matrix(self):
		# currently uses standard scatter_matrix

		pd.plotting.scatter_matrix(self.df_sampler, figsize=(20,20))

	def cat_to_numeric(self,catagorical):
		# converts categorical columns to numerical.Credit Matt D

		classes = catagorical.unique()
		classes_mapping = {cls: i for i, cls in enumerate(classes)}
		classes_inv_mapping = {i: cls for i, cls in enumerate(classes)}
		classes_numeric = catagorical.apply(lambda cls: classes_mapping[cls])
		return classes_numeric, classes_inv_mapping

	def add_date_columns(self,new_columns=['year','month','dayofweek','hour']):
		# looks for date columns and builds year, month etc columns

        date_columns = []
        for column in self.df_sampler.columns:
            if (self.df_sampler[column].dtype == 'datetime64[ns]'):
                    date_columns.append(column)

        for date_column in date_columns:
            for col in new_columns:
                if (col == 'year'):
                    self.df_sampler[date_column + '_year'] = self.df_sampler[date_column].apply(lambda x: x.year)
                if (col == 'month'):
                    self.df_sampler[date_column + '_month'] = self.df_sampler[date_column].apply(lambda x: x.month)
                if (col == 'dayofweek'):
                    self.df_sampler[date_column + '_dayofweek'] = self.df_sampler[date_column].apply(lambda x: x.dayofweek)
                elif (col == 'hour'):
                    self.df_sampler[date_column + '_hour'] = self.df_sampler[date_column].apply(lambda x: x.hour)

	def plot_all_columns(self,target_column):
		# builds box plots of all categorical columns. Credit Matt D

		date_columns = []
		for column in self.df_sampler.columns:
			if (self.df_sampler[column].dtype == 'datetime64[ns]'):
				date_columns.append(column)

		no_of_columns = len(self.df_sampler.columns) - len(date_columns)
		fig,axs = plt.subplots((1 + no_of_columns//3), 3, figsize=(12, 2 * (1 + (no_of_columns/3))))

		for i, column in enumerate(self.df_sampler.drop(date_columns, axis=1).columns):

			p = i // 3
			q = i % 3
			if (self.df_sampler[column].dtype == 'int64' or self.df_sampler[column].dtype == 'float64'):
				vertical_noise = np.random.uniform(-0.1, 0.1, size=self.df_sampler.shape[0])
				axs[p][q].scatter( self.df_sampler[column].values , self.df_sampler[target_column].values + vertical_noise,alpha=0.05)
			else:
				catagorical = self.df_sampler[column].fillna('NA')
				numeric, classes_mapping = self.convert_to_numeric(catagorical)
				noise = np.random.uniform(-0.3, 0.3, size=len(catagorical))
				vertical_noise = np.random.uniform(-0.1, 0.1, size=len(catagorical))
				axs[p][q].scatter(numeric + noise,  self.df_sampler[target_column].values + vertical_noise, color="grey", alpha=0.05)
				box_data = list(self.df_sampler[target_column].groupby(catagorical))
				axs[p][q].boxplot([data for _, data in box_data], positions=range(len(box_data)))
				axs[p][q].set_xticks(list(classes_mapping))
				axs[p][q].set_xticklabels(list(catagorical.unique()))
			axs[p][q].set_xlabel(column)
			axs[p][q].set_ylabel(target_column)
			plt.tight_layout()
		plt.show()
