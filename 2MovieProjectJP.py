from flask import Flask, jsonify
import recommander_V2_JPV

#déclare le serveur flask
app = Flask(__name__)

#crée la route web de la racine du site 
#et la lie à la fonction hello
@app.route("/")
def hello():
    return "Bienvenue sur le site de recommandation de films de JP V\n\nPour lancer une recommandation sans cluster taper nocluster/<film>\nPour lancer une recommandation avec cluster taper cluster/<film>\n\A noter que le <film> peut etre une chaine, un code IMDB ou un index de dataframe pandas"

@app.route('/nocluster/<string:lefilm>', methods = ['GET'])
def return_movie_nocluster(lefilm):
    la_recommendation, letat = recommander_V2_JPV.recommender(input_movie = lefilm, with_cluster = False, movie_number = 5)
    print( letat)
    if "ERROR" in letat:
        return jsonify({'status' : str(letat) })
    else:
        return jsonify({'films recommandes' : str(la_recommendation) })

@app.route('/cluster/<string:lefilm>', methods = ['GET'])
def return_movie_cluster(lefilm):
    la_recommendation, letat = recommander_V2_JPV.recommender(input_movie = lefilm, with_cluster = True, movie_number = 5)
    print( letat)
    if "ERROR" in letat:
        return jsonify({'status' : str(letat) })
    else:
        return jsonify({'films recommandes' : str(la_recommendation) })
        
if __name__ == "__main__":
#lance le serveur Flask
    #Lance app sur le port 8080 en mode debug
    app.run(host="0.0.0.0",debug=True, port = 8080)
