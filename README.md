# fastai-flask-api
Deploying a fastai model via a REST API with Flask

Run a Flask development server by typing:
```
$ FLASK_ENV=development FLASK_APP=app.py flask run
```


ToDo:
- Deploy the application on PythonAnywhere
- Deploy the application on GCP
- Run the Flask server in the production mode [link](https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/)
- The user may send non-image type files too. Since we are not handling errors, this will break our server. Adding an explicit error handing path that will throw an exception would allow us to better handle the bad inputs
- Add a UI by creating a page with a form which takes the image and displays the prediction [demo](https://pytorch-imagenet.herokuapp.com/) [source code](https://github.com/avinassh/pytorch-flask-api-heroku)



Links:
[Deploying Pytorch in Python via a REST API with Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)