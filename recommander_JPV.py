# -*- coding: utf-8 -*-
"""
Created on 26/03/2018

@author: JP
"""
"""Recommendation de films via MHA, clusters, kNN."""
import numpy as np
import pandas as pd

#pour test kNN
from sklearn import neighbors

#Nombre de films souhaités
k=5



def recommender (input_movie, movie_number = k):
	"""Recommendation de films via MHA, clusters, kNN.
	- Args:
            input_movie(string): film dont on cherche des recommendations
            movie_number(int): [de 1 à 50]: nombre de films recommendés souhaités
    - Returns:
            list_recommande (liste): les films recommendés 
			status (string): indique l'état de la demande
	"""
	
	list_recommande = []
	
	if movie_number < 1 or movie_number > 50 :
		return list_recommande, "ERROR, le nombre de recommandation doit etre entre 1 et 50]"
	
	#lecture du fichier, obtention d'un DataFrame Pandas.
	movie = pd.read_csv('movie_metadata_Num_NaN_Filled_Genres_Converted_avec_clusters_7_canberra_MHA_hclust.csv', sep=";", encoding='utf_8', decimal = ',', low_memory=False)
	
	#Je choisi d'utiliser une colonne avec mon_index, 
	#une indexation persistante à travers la création de susbset (sous data frame via df.loc)
	movie = movie.rename(index=str, columns={'Unnamed: 0': 'my_index'})
	
	#Création de la liste des features pris en compte
	col_movie_num = ['my_index','num_voted_users' ,'cast_total_facebook_likes' ,'imdb_score' ,'movie_facebook_likes' ,'color' ,'num_critic_for_reviews' ,'duration' ,'director_facebook_likes' ,'actor_3_facebook_likes' ,'actor_1_facebook_likes' ,'recette' ,'facenumber_in_poster' ,'num_user_for_reviews' ,'budget' ,'title_year' ,'actor_2_facebook_likes' ,'aspect_ratio' ,'genre_action' ,'genre_adventure' ,'genre_animation' ,'genre_biography' ,'genre_comedy' ,'genre_crime' ,'genre_documentary' ,'genre_drama' ,'genre_family' ,'genre_fantasy' ,'genre_film.noir' ,'genre_game.show' ,'genre_history' ,'genre_horror' ,'genre_music' ,'genre_musical' ,'genre_mystery' ,'genre_news' ,'genre_reality.tv' ,'genre_romance' ,'genre_sci.fi' ,'genre_short' ,'genre_sport' ,'genre_thriller' ,'genre_war' ,'genre_western']
	
	#Recherche du input_movie
	#########################
	#Création d'une nouvelle colonne avec titres en minuscules et le caractère parasite dégagé !
	movie['titre_film'] = movie.movie_title.str.lower().str.replace(u'\xa0', u'').str.strip()

	#Recherche d'un titre de film: input_movie
	subset_film = movie.loc[movie.titre_film ==  input_movie.lower().strip()]

	#recherche du cluster ou se trouve input_movie
	le_cluster=0
	if subset_film.shape[0] == 0 :
		status = "ERROR, Le film " + input_movie + " est inconnu dans la base de données. Process stoppé"
	elif subset_film.shape[0] > 1 :
		status = "ERROR, Le film " + input_movie + " est présent " + str(subset_film.shape[0]) + " fois dans la base de données. Process stoppé"
	else:
		status = "OK, Le film " + input_movie + " est dans le cluster: " + str(subset_film['cluster'].iloc[0])
		le_cluster = subset_film['cluster'].iloc[0]
		#Je crée un subset de ce cluster
		subset_cluster = movie.loc[movie.cluster == le_cluster]	
		#print(f"Taille du cluster {le_cluster} : {subset_cluster.shape}")

	#Recherche des recommandations
	##############################
	if le_cluster == 0 :
		return list_recommande, status
	else :
		#Création d'un sous_matrice avec les valeurs numériques
		subset_cluster_ref =  subset_cluster[col_movie_num]
		#Lancement du kNN
		knn = neighbors.NearestNeighbors (n_neighbors= movie_number+1, metric = 'canberra' )
		#Apprentissage
		knn.fit(subset_cluster_ref)
		#k recommendations
		distances, indices = knn.kneighbors(subset_film[col_movie_num])
		
		list_recommande=[]
		mon_indice = indices[0]
		for i in mon_indice[1:]:
			mon_index = subset_cluster_ref['my_index'].iloc[i]
			ma_rech_film = movie.loc[movie.my_index == mon_index]
			#ma_chaine = ma_rech_film['titre_film'].iloc[0] + " (mon_index: " + str(mon_index) +")"
			#list_recommande.append(ma_chaine)
			mon_dict  = {ma_rech_film['titre_film'].iloc[0] : mon_index }
			list_recommande.append(mon_dict)
		return list_recommande, status
