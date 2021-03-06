from flask import Flask, jsonify
import recommander_V2_JPV

#déclare le serveur flask
app = Flask(__name__)

#crée la route web de la racine du site 
#et la lie à la fonction hello
@app.route("/")
def hello():
    return "Bienvenue sur le site de recommandation de films de JP V          Pour lancer une recommandation SANS cluster tapez /nocluster/lefilm          Pour lancer une recommandation AVEC cluster tapez /cluster/lefilm          A noter que lefilm peut etre une chaine ou un code IMDB"

@app.route("/nocluster/")
def hello_nocluster():
    return "Version sans cluster:  Entrez un titre de film ou un id IMDB"

@app.route('/nocluster/<string:lefilm>', methods = ['GET'])
def return_movie_nocluster(lefilm):
    la_recommandation, letat = recommander_V2_JPV.recommender(input_movie = lefilm, with_cluster = False, movie_number = 5)
    print( letat)
    if "ERROR" in letat:
        return jsonify({'status' : str(letat) })
    else:
        return jsonify({'films recommandes' : str(la_recommandation) })

@app.route("/cluster/")
def hello_cluster():
    return "Version avec cluster:  Entrez un titre de film ou un id IMDB"

@app.route('/cluster/<string:lefilm>', methods = ['GET'])
def return_movie_cluster(lefilm):
    la_recommandation, letat = recommander_V2_JPV.recommender(input_movie = lefilm, with_cluster = True, movie_number = 5)
    print( letat)
    if "ERROR" in letat:
        return jsonify({'status' : str(letat) })
    else:
        return jsonify({'films recommandes' : str(la_recommandation) })
        
if __name__ == "__main__":
#lance le serveur Flask
    #Lance app sur le port 8080 en mode debug
    app.run(host="0.0.0.0",debug=True, port = 8080)
