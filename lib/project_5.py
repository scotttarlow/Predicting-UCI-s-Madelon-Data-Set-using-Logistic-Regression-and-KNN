from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data_from_database(url = 'joshuacook.me',user='dsi',
	password = 'correct horse battery staple',database = 'dsi',
	port = '5432', sql='SELECT * FROM Madelon'):
	'''This function loads a table from a database from remote server using sqlachmey
	 and returns a data frame'''

	engine = create_engine("postgresql://{}:{}@{}:{}/{}".format(user, password, url, port, database))
	return pd.read_sql(sql,engine)

def make_data_dict(x,y):
	''' This function takes a feature matrix and target vector 
	and returns a trainning and testing set in a data dictionary '''

	X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=3)
	return {'X_train': X_train,
	'X_test': X_test,
	'y_train': y_train,
	'y_test': y_test}

def general_transformer(transformer,data_dict):
	''' This function takes a data dictionary and fits a transformer
	such as a scaler onto the training set from the data dictionary.
	The output is a returned data dictionary with a "processes" list
	added to the dictionary.
	'''
	X_train = data_dict['X_train']
	X_test = data_dict['X_test']
	y_train = data_dict['y_train']
	y_test = data_dict['y_test']
	transform = transformer
	try:
		transform.fit(X_train)
		X_train = transform.transform(X_train)
		X_test = transform.transform(X_test)
	except:
		transform.fit(X_train,y_train)
		X_train = transform.transform(X_train)
		X_test = transform.transform(X_test)

	return {'processes':[transform],
    'X_train':X_train,
    'X_test':X_test,
    'y_train':y_train,
    'y_test':y_test}

def general_model(model,data_dict):
	'''takes in  a data dictionary with "processes" list, fits a model
	and returns the data dictionary with the model appended to the "processes"
	list and addes train/test scores of the model to the dictionary
	'''
	temp = model.fit(data_dict['X_train'],data_dict['y_train'])
	test_score = temp.score(data_dict['X_test'],data_dict['y_test'])
	train_score =  temp.score(data_dict['X_train'],data_dict['y_train'])
	data_dict['scores'] = [train_score,test_score]
	data_dict['prediction'] = temp.predict(data_dict['X_test'])
	if 'processes' in data_dict:
		data_dict['processes'].append(temp)
	else:
		data_dict['processes'] = temp
	return data_dict
