# -*- coding: utf-8 -*-
"""
Created on 15/04/2018

@author: JP
"""
"""Recommendation de films via K-Means, clusters, kNN."""
import numpy as np
import pandas as pd

#pour test kNN
from sklearn import neighbors

#Nombre de films souhaités
k=5


# Fonction listant les colonnes dont le titre contient des chaines de caractères
def col_rech_titre(df, fin = True, suffix =""):
    """Affiche le nom des colonnes d'un dataframe contenant une chaîne de caractères.

    - Args:
        df(pandas.dataframe): dataframe
        fin(boolean): flag pour indiquer un préfixe ou un suffixe
        suffix(str): chaîne de caractères recherchée
        
    - Returns:
        Liste contenant les colonnes contenant la chaîne de caractères.
    """
    liste_col = []
    if suffix !="":
        if fin == True:
            for col in df.columns:
                if col.endswith(suffix) == True:
                    liste_col.append(col)
        else:
            for col in df.columns:
                if col.startswith(suffix) == True:
                    liste_col.append(col)                  
    return liste_col 



def recommender (input_movie, with_cluster=True, movie_number = k):
	"""Recommendation de films via K-Means, clusters/sans clusters, kNN.
	- Args:
            input_movie(string): film dont on cherche des recommendations: code IMDB, code Index (du data frame), nom du film
            with_cluster(bool): avec / sans clustering
            movie_number(int): [de 1 à 50]: nombre de films recommendés souhaités
    - Returns:
            list_recommande (liste): les films recommendés 
			status (string): indique l'état de la demande
	"""
	
	mon_film = input_movie
	list_recommande = []
	status = ""
	
	if movie_number < 1 or movie_number > 10 :
		return list_recommande, "ERROR, le nombre de recommandation doit etre entre 1 et 50]"
	
	#lecture du fichier, obtention d'un DataFrame Pandas.
	movie = pd.read_csv('movie_metadata_Clusters_KM.csv', sep=",", encoding='utf_8', decimal = '.', low_memory=False)
	
	#########################
	#Détection si on a une input: Chaine/Imdb/Index
	#########################
	le_cluster = -1
	imdb = False
	index = False
	chaine = False
	if not mon_film.isdigit():
		chaine = True
	else:
		mon_film=int(mon_film)
		if mon_film > movie.shape[0] :
			imdb=True
		else:
			index = True
	print(f"imdb: {imdb}, index: {index}, chaine: {chaine}")
	
	#########################
	#Recherche du input_movie
	#########################
	if imdb:
		Liste_Index = movie.loc[movie.imdb_id == mon_film].index.values
		if len(Liste_Index) != 1 :
			print("Code Imdb Inconnu")
			resultatD = "Code Imdb Inconnu" 
			query_index = -1
		else :
			query_index = Liste_Index[0]

	if chaine:
		#Création d'une nouvelle colonne avec titres en minuscules et le caractère parasite dégagé !
		movie['titre_film'] = movie.movie_title.str.lower().str.replace(u'\xa0', u'').str.strip()
		#Recherche
		Liste_Index = movie.loc[movie.titre_film == mon_film.lower().strip()].index.values
		if len(Liste_Index) != 1 :
			print("Nom de film Inconnu")
			resultatD = "Nom de film Inconnu"
			query_index = -1
		else :
			query_index = Liste_Index[0]

	if index:
		if mon_film <= movie.shape[0] and mon_film>=0:
			query_index = index
		else:
			print(f"Le code Index doit être compris entre 0 et {movie.shape[0]} vous avez fourni {mon_film}")
			resultatD = "Le Code Index n'est pas correct)"
			query_index = -1

			
	if query_index != -1 and with_cluster:
		#Recherche du cluster du film_input
		le_cluster = movie.loc[movie.index.values ==  query_index]['Cluster_KM_V4'][query_index]
		print(f"Mon cluster du film Input: {le_cluster}")

		#Je crée un sous_df de ce cluster
		sous_df = movie.loc[movie.Cluster_KM_V4 == le_cluster]
		print(f"Taille du cluster {le_cluster} : {sous_df.shape}")

	#########################
	#Recommandation de k films
	#########################

	from sklearn.neighbors import NearestNeighbors

	if (le_cluster != -1 and with_cluster) or (query_index != -1 and not with_cluster):
		if with_cluster :
			#Création d'un DF ne contenant que les données de type  QUAL_*
			QUAL_sous_df = sous_df[col_rech_titre(movie, False, "QUAL_")]
			QUAL_sous_df.shape
		else:
			#on ne fait pas de clustering
			sous_df = movie
			QUAL_sous_df = movie[col_rech_titre(movie, False, "QUAL_")]
			QUAL_sous_df.shape

		#Lancement du kNN
		model_knn = NearestNeighbors(metric = 'euclidean', algorithm = 'auto')
		model_knn.fit(QUAL_sous_df)
		distances, indices = model_knn.kneighbors(QUAL_sous_df.loc[query_index, :].values.reshape(1, -1), n_neighbors = k+1)
		mon_indice = indices[0]

		resultatI = []
		resultatN = []
		resultatD = []
		for i in range(0, len(distances.flatten())):
			if i == 0:
				print (f"Recommandations pour {sous_df['movie_title'].loc[query_index]} (index = {query_index}):\n")
			else:
				if query_index != indices.flatten()[i]:
					print (f"{i}: {sous_df['movie_title'].iloc[indices.flatten()[i]]} (index = {indices.flatten()[i]}), avec une distance de {distances.flatten()[i]}:")
					resultatI.append(indices.flatten()[i])
					resultatN.append(sous_df['movie_title'].iloc[indices.flatten()[i]].strip())
					mon_dict  = {sous_df['movie_title'].iloc[indices.flatten()[i]].strip() : indices.flatten()[i] }
					resultatD.append(mon_dict)
		print(f"\nRésultats (index): {resultatI}")
		print(f"\nRésultats (noms): {resultatN}")
		print(f"\nRésultats (dict): {resultatD}")
		status = "OK"
	else:
		status = "ERROR, Probleme lors de la definition du traitement a effectuer"
	return resultatD, status
