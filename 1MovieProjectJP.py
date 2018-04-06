from flask import Flask, jsonify
import recommander_JPV

#déclare le serveur flask
app = Flask(__name__)

#crée la route web de la racine du site 
#et la lie à la fonction hello
@app.route("/")
def hello():
    return "Hello World (JPV1)!"

@app.route('/recommend/<string:lefilm>', methods = ['GET'])
def return_movie(lefilm):
    la_recommendation, letat = recommander_JPV.recommender(input_movie = lefilm, movie_number = 5)
    print( letat)
    if "ERROR" in letat:
        return jsonify({'status' : str(letat) })
    else:
        return jsonify({'films recommandes' : str(la_recommendation) })
        
if __name__ == "__main__":
#lance le serveur Flask
    #Lance app sur le port 8080 en mode debug
    app.run(host="0.0.0.0",debug=True, port = 8080)
