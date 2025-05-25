#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from model_deployment import predict_genres

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='API de Predicción de Géneros de Películas',
    description='API para predecir los géneros de una película a partir de la sinopsis y el título (opcional).'
)

ns = api.namespace('predecir',
     description='Clasificador de Géneros de Películas')

parser = api.parser()

parser.add_argument(
    'sinopsis',
    type=str,
    required=True,
    help='Sinopsis o trama de la película a analizar',
    location='args'
)

parser.add_argument(
    'titulo',
    type=str,
    required=False,
    help='Título de la película (opcional)',
    location='args'
)

parser.add_argument(
    'top_n',
    type=int,
    required=False,
    default=3,
    help='Número de géneros principales a devolver (por defecto: 3)',
    location='args'
)

genre_model = api.model('Género', {
    'genre': fields.String(description='Género de la película'),
    'probability': fields.Float(description='Probabilidad')
})

resource_fields = api.model('Respuesta', {
    'generos': fields.List(fields.Nested(genre_model), description='Lista de géneros predichos con sus probabilidades')
})

@ns.route('/')
class GenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        sinopsis = args['sinopsis']
        titulo = args['titulo'] if args['titulo'] else ""
        top_n = args['top_n']

        generos = predict_genres(sinopsis, titulo, top_n)

        return {
            "generos": generos
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)