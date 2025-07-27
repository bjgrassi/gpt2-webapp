# GPT2 Webapp
## 1. If you're cloning this repo for the first time
### 1.1. Check if you have python installed. 
`python --version`

It must be 3.11.0

### 1.2. (Windows) Create and activate the virtual environment 

```
py -3 -m venv .venv
.venv\Scripts\activate
```

### 1.3. Install the requirements (the libraries used on the app)
```
pip install -r requirements.txt
```

### 1.4 Run the application
```
flask --app main run --debug
```

Will show messages on the terminal. Look for the "Running on http://127.0.0.1:5000" to open the app.
Every change you do in the code, reload the browser.

## 2. (Windows) If you already have the app in your computer, just run the application
```
.venv\Scripts\activate
flask --app main run --debug
```

## If you added a new library, save on the requirements.txt
`pip3 freeze > requirements.txt`

