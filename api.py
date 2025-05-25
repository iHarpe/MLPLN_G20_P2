#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from m09_model_deployment import predict_genres

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction API',
    description='API for predicting movie genres based on plot text')

ns = api.namespace('predict', 
     description='Movie Genre Classifier')
   
parser = api.parser()

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Movie plot text to analyze', 
    location='args')

parser.add_argument(
    'title', 
    type=str, 
    required=False, 
    help='Movie title (optional)', 
    location='args')

genre_model = api.model('Genre', {
    'genre': fields.String(description='Movie genre'),
    'probability': fields.Float(description='Probability score')
})

resource_fields = api.model('Resource', {
    'genres': fields.List(fields.Nested(genre_model))
})

@ns.route('/')
class GenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        plot = args['plot']
        title = args['title'] if args['title'] else ""
        
        genres = predict_genres(plot, title)
        
        return {
            "genres": genres
        }, 200
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)